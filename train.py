#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch, torchvision
from random import randint
from utils.loss_utils import l2_loss, ssim, l1_loss, ScaleAndShiftInvariantLoss
from gaussian_renderer import render, network_gui, render_dyn, render_all, return_gaussians_boxes_and_box2worlds
from scene.cameras import augmentCamera
from PIL import Image
import sys
from scene import Scene, GaussianModel, GaussianBoxModel
from utils.general_utils import safe_state, Normal2Torch, check_objects_in_frame
from utils.norminit_utils import initialize_gaussians_with_window_normals
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, KITTI360DataParams, BoxModelParams, SDRegularizationParams
from kitti360scripts.helpers import labels as kittilabels
import random
import numpy as np
from utils.graphics_utils import normal_to_rot, cam_normal_to_world_normal, standardize_quaternion, matrix_to_quaternion, quaternion_to_matrix
from loss import loss_normal_guidance
import wandb

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.loss_utils import l2_loss
import copy

from scene.cameras import make_camera_like_input_camera
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from torch.linalg import norm
import math


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
torch.autograd.set_detect_anomaly(True)

def seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def training(dataset, opt, pipe, cfg_kitti, cfg_box, cfg_sd, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint_dir, debug_from, dyn_obj_list=['car'], exp_note='', run=None, args=None, output_dir=None):
    seed_all(dataset.seed)
    # add start / ending iteration of diffusion guidance for test
    first_iter = 0
    unique_str = prepare_output_and_logger(dataset, cfg_kitti, exp_note, output_dir=output_dir)
    if run is not None:
        run.tags += (unique_str, )
    
    # Set-up Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, cfg_kitti, cfg_box)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Initialize Gaussian covariances with monocular normals
    gaussians = initialize_gaussians_with_window_normals(gaussians, scene, pipe, background)

    for instanceId in scene.gaussian_box_models.keys():
        scene.gaussian_box_models[instanceId].training_setup(opt)

    if checkpoint_dir:
        (model_params, first_iter) = torch.load(checkpoint_dir + "/chkpnt" + str(checkpoint_iterations[-1]) + ".pth")
        gaussians.restore(model_params, opt)
        for instanceId in scene.gaussian_box_models.keys():
            (model_params, first_iter) = torch.load(checkpoint_dir + "/chkpnt" + str(checkpoint_iterations[-1]) + f"_inst_{instanceId}" +".pth")
            scene.gaussian_box_models[instanceId].restore(model_params, opt)

    # Load diffusion regularizer
    from loss import LoRADiffusionRegularizer
    sd_reg = LoRADiffusionRegularizer(dataset, cfg_kitti, cfg_sd, opt.iterations)


    if cfg_sd.perceptual_loss:
        from loss import VGGPerceptualLoss
        perceptual_loss = VGGPerceptualLoss().cuda()


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
   

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        for instanceId in scene.gaussian_box_models.keys():
            scene.gaussian_box_models[instanceId].update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            for instanceId in scene.gaussian_box_models.keys():
                scene.gaussian_box_models[instanceId].oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        all_bboxes = scene.getTrainBboxes() # This includes all existing vehicle bboxes in a scene that's "DYNAMIC".

        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # Retrieve frame, instance information on this camera
        frame = viewpoint_cam.frame
        this_frame_includes_objects, insts_in_frame = check_objects_in_frame(frame, all_bboxes) 
        
        # Retrieve GT image
        gt_image = viewpoint_cam.original_image
        
        # Render dynamic scene 
        if this_frame_includes_objects:
            bboxes = all_bboxes[frame]
            gaussians_boxes, box_models, box2worlds = return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame)
            render_pkg = render_all(viewpoint_cam, gaussians, gaussians_boxes, box2worlds, pipe, background)
        
        # Render static scene
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        
        image, viewspace_point_tensor, visibility_filter, radii, cov_quat, cov_scale = render_pkg["render"], \
                                                                                       render_pkg["viewspace_points"], \
                                                                                       render_pkg["visibility_filter"], \
                                                                                       render_pkg["radii"], \
                                                                                       render_pkg['render_cov_quat'], \
                                                                                       render_pkg['render_cov_scale']         



        # Photometric loss  
        Ll1 = l1_loss(image, gt_image)
        lambda_dssim = opt.lambda_dssim
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Normal guidance loss
        Lng = loss_normal_guidance(viewpoint_cam, cov_quat, cov_scale)
        loss += opt.lambda_dnormal * Lng


        # Diffusion guidance loss
        if iteration > cfg_sd.start_guiding_from_iter and iteration < cfg_sd.end_guiding_at_iter:
            # [1] Augment viewpoints
            viewpoint_cam_aug, yaw, pitch, t_y, aug_dir = augmentCamera(viewpoint_cam, cfg_sd)
            
            # [2] Render augmented viewpoints
            image_aug = render(viewpoint_cam_aug, gaussians, pipe, background)["render"]

            # [3] Random crop renderings from augmented view.
            h_aug, w_aug = image_aug.shape[1], image_aug.shape[2]
            if cfg_sd.global_crop:
                w_crop_start = randint(0, w_aug-h_aug)
            else:
                if aug_dir == -1: # Look right
                    w_crop_start = randint((w_aug-h_aug)//2, w_aug-h_aug)
                else: # Look left
                    w_crop_start = randint(0, (w_aug-h_aug) // 2)

            image_aug = image_aug[None, ..., w_crop_start:w_crop_start+h_aug]

            # [3] Compute guidance loss
            loss_guidance = sd_reg(image_aug, iteration)
            loss += loss_guidance


        loss.backward()

        ## Do not update nan gradients for box optimizers 
        ## TODO: Why is this happening?
        if this_frame_includes_objects:
            for bm in box_models:
                if torch.any(torch.isnan(bm.delta_r.grad)) or torch.any(torch.isnan(bm.delta_s.grad)):
                    bm.delta_r.grad = torch.zeros_like(bm.delta_r.grad)
                    bm.delta_s.grad = torch.zeros_like(bm.delta_s.grad)
                    bm.delta_t.grad = torch.zeros_like(bm.delta_t.grad)
            
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss (L1)": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            scalar_kwargs={}
            scalar_kwargs["loss"] = loss.item()
            scalar_kwargs["loss_ema"] = ema_loss_for_log
            scalar_kwargs["l1_loss"] = Ll1.item()
            scalar_kwargs["normal_loss"] = Lng.item()

            if iteration > cfg_sd.start_guiding_from_iter and iteration < cfg_sd.end_guiding_at_iter:
                scalar_kwargs[f"{cfg_sd.guidance_mode}_loss_guidance"] = loss_guidance.item()

                # Record box refinment information 
                deltas = []
                if this_frame_includes_objects:
                    for box_model in box_models:
                        deltas.append(box_model.get_deltas())
                    deltas = torch.mean(torch.Tensor(deltas), dim=0)
                    scalar_kwargs["delta_r_norm"] = deltas[0].item()
                    scalar_kwargs["delta_s_norm"] = deltas[1].item()
                    scalar_kwargs["delta_t_norm"] = deltas[2].item()


        # Log and save
        with torch.no_grad():
            if not args.no_wandb:
                save_dir = None
                if dataset.save_results_as_images:
                    save_dir = scene.model_path
                wandb.log(scalar_kwargs, step=iteration)
                training_report(iteration, testing_iterations, scene, all_bboxes, gaussians, pipe, background, dyn_obj_list, cfg_sd=cfg_sd, scalar_kwargs=scalar_kwargs, save_dir=save_dir)
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        end_idx = gaussians.get_xyz.shape[0]
        cur_viewspace_point_tensor = slice_with_grad(viewspace_point_tensor, 0, end_idx)
        densification_and_optimization(gaussians, opt, cfg_sd, iteration, cur_viewspace_point_tensor, visibility_filter[:end_idx], scene, pipe, radii[:end_idx], dataset)
        
        if this_frame_includes_objects:
            start_idx=end_idx
            for gaussians_box in gaussians_boxes:
                # Optimize box gaussians
                idx_length = gaussians_box.get_xyz.shape[0]
                cur_viewspace_point_tensor = slice_with_grad(viewspace_point_tensor, start_idx, start_idx+idx_length)
                densification_and_optimization(gaussians_box, 
                                               opt,
                                               cfg_sd, 
                                               iteration, 
                                               cur_viewspace_point_tensor, 
                                               visibility_filter[start_idx:start_idx+idx_length], 
                                               scene, 
                                               pipe, 
                                               radii[start_idx:start_idx+idx_length],
                                               dataset,
                                               box=True)
                start_idx += idx_length

            # Optimize bounding boxes
            for box_model in box_models:
                box_model.optimizer.step()
                box_model.optimizer.zero_grad()
                box_model.regularize(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                for instanceId in scene.gaussian_box_models.keys():
                    torch.save((scene.gaussian_box_models[instanceId].capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + f"_inst_{instanceId}" +".pth")


def slice_with_grad(tensor, start, end):
    out = tensor[start:end]
    out.grad = tensor.grad[start:end]
    return out

def densification_and_optimization(gaussians, opt, cfg_sd, iteration, viewspace_point_tensor, visibility_filter, scene, pipe, radii, dataset, box=False):
    # Densification
    if box:
        condition = iteration < opt.densify_until_iter_box
    else:
        condition = iteration < opt.densify_until_iter
    if condition: 
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            densify_grad_threshold = opt.densify_grad_threshold
            if box:
                densify_grad_threshold *= 0.5
                if size_threshold is not None:
                    size_threshold *= 0.5
            # do_prune = (iteration < cfg_sd.start_guiding_from_iter) and cfg_sd.do_prune
            # gaussians.densify_and_prune(densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, prune=do_prune)
            gaussians.densify_and_prune(densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        
        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()

    # Optimizer step
    if iteration < opt.iterations:
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)


def prepare_output_and_logger(args, cfg_kitti, exp_note, output_dir): 
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join(output_dir, f"{cfg_kitti.seq}_{str(cfg_kitti.start_frame).zfill(10)}_{str(cfg_kitti.end_frame).zfill(10)}", f"{unique_str[0:10]}_{exp_note}")
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    return unique_str

def render_novelview_image(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, add_xrot_val=0, add_zrot_val=0, add_tz = 0):
    frame = viewpoint.frame
    this_frame_includes_objects, insts_in_frame = check_objects_in_frame(frame , all_bboxes)
    
    image_full = None
    viewpoint_new = make_camera_like_input_camera(viewpoint, add_xrot_val=add_xrot_val, add_zrot_val=add_zrot_val, add_tz = add_tz)    

    if this_frame_includes_objects:
        bboxes = all_bboxes[frame]
        gaussians_boxes, box_models, box2worlds = return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame)
        image_full = torch.clamp(render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render"], 0.0, 1.0)
        image = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render"], 0.0, 1.0)
    else: 
        image = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render"], 0.0, 1.0)    

    return image, image_full

def render_novelview_rotaxis(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, idx_best='min_scale', add_xrot_val=0, add_yrot_val=0, add_zrot_val=0, add_tx = 0, add_ty = 0, add_tz = 0):
    frame = viewpoint.frame
    normal_gt = viewpoint.original_normal.reshape(3,1,-1).permute(2,0,1)

    this_frame_includes_objects, insts_in_frame = check_objects_in_frame(frame, all_bboxes)

    cov_rot_full = None
    _, H, W = viewpoint.original_normal.shape    


    viewpoint_new = make_camera_like_input_camera(viewpoint, add_xrot_val=add_xrot_val, add_zrot_val=add_zrot_val, add_tz = add_tz)    

    if this_frame_includes_objects:
        bboxes = all_bboxes[frame]
        gaussians_boxes, box_models, box2worlds = return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame)
        cov_quat_full = render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render_cov_quat"]
        cov_quat = render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"]
        cov_scale_full = render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render_cov_scale"]
        cov_scale = render(viewpoint_new, gaussians, pipe, background)["render_cov_scale"]

        cov_quat_full = cov_quat_full.permute(1,2,0).reshape(-1, 4).contiguous()
        cov_rot_full = quaternion_to_matrix(cov_quat_full)                # cov_rot.shape: torch.Size([n_pix, 3, 3])
        cov_scale_full = cov_scale_full.permute(1,2,0).reshape(-1,3).contiguous()        
    else: 
        cov_quat = render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"]
        cov_scale = render(viewpoint_new, gaussians, pipe, background)["render_cov_scale"]
    
    cov_quat = cov_quat.permute(1,2,0).reshape(-1, 4).contiguous() # -> npix x 4
    cov_rot = quaternion_to_matrix(cov_quat)                # cov_rot.shape: torch.Size([n_pix, 3, 3])
    cov_scale = cov_scale.permute(1,2,0).reshape(-1,3).contiguous() # -> n_pix, 3

    R_world2cam = torch.from_numpy(viewpoint.R).transpose(-1, -2).to(device=cov_rot.device).type_as(cov_rot)
    R_world2cam = R_world2cam[None] # -> 1x3x3
    norm_like = (R_world2cam @ cov_rot) # npix x 3 x 3    

    if idx_best == 'gt_like':
        idx_best = torch.argmax(torch.sum(normal_gt * norm_like, dim=1), dim=1)[:,None, None].repeat(1,3,1)    
    elif idx_best == 'min_scale':
        idx_best = torch.argmin(cov_scale, dim=1)[:,None, None].repeat(1,3,1)
    else:
        raise RuntimeError(f'unknown idx_best:{idx_best}')

    norm_like_best = norm_like.gather(dim=2, index = idx_best).squeeze().permute(1,0)
    norm_like_best = ((( norm_like_best.reshape(-1, H, W)*-1)+1)/2)*255
    norm_like_best = torch.clip(norm_like_best, min=0, max=255)
    norm_like_best = norm_like_best.to(torch.uint8)  

    norm_like_best_full = None
    if this_frame_includes_objects:
        norm_like_full = (R_world2cam @ cov_rot) # npix x 3 x 3    
        idx_best = torch.argmax(torch.sum(normal_gt * norm_like_full, dim=1), dim=1)[:,None, None].repeat(1,3,1)
        norm_like_best_full = norm_like.gather(dim=2, index = idx_best).squeeze().permute(1,0)
        norm_like_best_full = ((( norm_like_best_full.reshape(-1, H, W)*-1)+1)/2)*255
        norm_like_best_full = torch.clip(norm_like_best_full, min=0, max=255)
        norm_like_best_full = norm_like_best_full.to(torch.uint8)   

    return norm_like_best, norm_like_best_full

def render_novelview_bestrotaxis(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, add_xrot_val=0, add_yrot_val=0, add_zrot_val=0, add_tx = 0, add_ty = 0, add_tz = 0):
    frame = viewpoint.frame
    normal_gt = viewpoint.original_normal.reshape(3,1,-1).permute(2,0,1)

    this_frame_includes_objects, insts_in_frame = check_objects_in_frame(frame, all_bboxes)

    cov_rot_full = None
    _, H, W = viewpoint.original_normal.shape    

   
    viewpoint_new = make_camera_like_input_camera(viewpoint, add_xrot_val=add_xrot_val, add_zrot_val=add_zrot_val, add_tz = add_tz)    
    
    if this_frame_includes_objects:
        bboxes = all_bboxes[frame]
        gaussians_boxes, boxmodels, box2worlds = return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame)
        cov_quat_full = torch.clamp(render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render_cov_quat"], -1.0, 1.0)
        cov_quat = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"], -1.0, 1.0)

        cov_quat_full = cov_quat_full.permute(1,2,0).reshape(-1, 4).contiguous()
        cov_rot_full = quaternion_to_matrix(cov_quat_full)                # cov_rot.shape: torch.Size([n_pix, 3, 3])
    else: 
        cov_quat = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"], -1.0, 1.0)    
    
    cov_quat = cov_quat.permute(1,2,0).reshape(-1, 4).contiguous() # -> npix x 4
    cov_rot = quaternion_to_matrix(cov_quat)                # cov_rot.shape: torch.Size([n_pix, 3, 3])

    R_world2cam = torch.from_numpy(viewpoint.R).transpose(-1, -2).to(device=cov_rot.device).type_as(cov_rot)
    R_world2cam = R_world2cam[None] # -> 1x3x3

    norm_like = (R_world2cam @ cov_rot) # npix x 3 x 3    
    idx_best = torch.argmax(torch.sum(normal_gt * norm_like, dim=1), dim=1)[:,None, None].repeat(1,3,1)
    norm_like_best = norm_like.gather(dim=2, index = idx_best).squeeze().permute(1,0)
    norm_like_best = ((( norm_like_best.reshape(-1, H, W)*-1)+1)/2)*255
    norm_like_best = torch.clip(norm_like_best, min=0, max=255)
    norm_like_best = norm_like_best.to(torch.uint8)  

    norm_like_best_full = None
    if this_frame_includes_objects:
        norm_like_full = (R_world2cam @ cov_rot) # npix x 3 x 3    
        idx_best = torch.argmax(torch.sum(normal_gt * norm_like_full, dim=1), dim=1)[:,None, None].repeat(1,3,1)
        norm_like_best_full = norm_like.gather(dim=2, index = idx_best).squeeze().permute(1,0)
        norm_like_best_full = ((( norm_like_best_full.reshape(-1, H, W)*-1)+1)/2)*255
        norm_like_best_full = torch.clip(norm_like_best_full, min=0, max=255)
        norm_like_best_full = norm_like_best_full.to(torch.uint8)   

    return norm_like_best, norm_like_best_full

def render_novelview_rotaxis_onebyone(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, add_xrot_val=0, add_yrot_val=0, add_zrot_val=0, add_tx = 0, add_ty = 0, add_tz = 0):
    frame = viewpoint.frame
    this_frame_includes_objects, insts_in_frame = check_objects_in_frame(frame, all_bboxes)

    cov_rot_full = None
    _, H, W = viewpoint.original_normal.shape    


    viewpoint_new = make_camera_like_input_camera(viewpoint, add_xrot_val=add_xrot_val, add_zrot_val=add_zrot_val, add_tz=add_tz)    

    if this_frame_includes_objects:
        bboxes = all_bboxes[frame]
        gaussians_boxes, boxmodels, box2worlds = return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame)
        cov_quat_full = torch.clamp(render_all(viewpoint_new, gaussians, gaussians_boxes, box2worlds, pipe, background)["render_cov_quat"], -1.0, 1.0)
        cov_quat = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"], -1.0, 1.0)

        cov_quat_full = cov_quat_full.permute(1,2,0).reshape(-1, 4).contiguous()
        cov_rot_full = quaternion_to_matrix(cov_quat_full)                # cov_rot.shape: torch.Size([n_pix, 3, 3])
    else: 
        cov_quat = torch.clamp(render(viewpoint_new, gaussians, pipe, background)["render_cov_quat"], -1.0, 1.0)    
    
    cov_quat = cov_quat.permute(1,2,0).reshape(-1, 4).contiguous()
    cov_rot = quaternion_to_matrix(cov_quat)                # cov_rot.shape: torch.Size([n_pix, 3, 3])

    cov_rot = cov_rot.permute(1,2,0)                # -> shape: torch.Size([3, 3, n_pix])
    R_world2cam = torch.from_numpy(viewpoint.R).transpose(-1, -2).to(device=cov_rot.device).type_as(cov_rot)

    cov_axis_list = []
    for i in range(3):
        norm_like_cam = (R_world2cam @ cov_rot[:, i, :])
        cov_axis = ((( norm_like_cam.reshape(-1, H, W)*-1)+1)/2)*255
        cov_axis = torch.clip(cov_axis, min=0, max=255)
        cov_axis = cov_axis.to(torch.uint8)  
        cov_axis_list.append(cov_axis) 

    cov_axis_y = (( (R_world2cam @ cov_rot[:, 1, :]).reshape(-1, H, W)*-1)+1)/2
    cov_axis_z = (( (R_world2cam @ cov_rot[:, 2, :]).reshape(-1, H, W)*-1)+1)/2

    cov_axis_full_list = [None, None, None]    
    if this_frame_includes_objects:
        cov_rot_full  = cov_rot_full.permute(1,2,0)  # -> shape: torch.Size([3, 3, n_pix])
        for i in range(3):
            norm_like_cam = (R_world2cam @ cov_rot_full[:, i, :])
            cov_axis_full = ((( norm_like_cam.reshape(-1, H, W)*-1)+1)/2) * 255
            cov_axis_full = torch.clip(cov_axis_full, min=0, max=255)
            cov_axis_full = cov_axis_full.to(torch.uint8)
            cov_axis_full_list.append(cov_axis_full) 

    return cov_axis_list, cov_axis_full_list


def training_report(iteration, testing_iterations, scene : Scene, all_bboxes, gaussians, pipe, background, dyn_obj_list, cfg_sd=None, viewpoint_stack=None, scalar_kwargs=None, save_dir=None):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        if viewpoint_stack is not None:
            validation_configs = ({'name': 'test', 'cameras' : viewpoint_stack}, 
                                  {'name': 'train', 'cameras' : viewpoint_stack})

        # add augmented image to log candidates
        # Look left/right by 30/60 deg
                          #Rx  Rz  Tz
        cam_aug_params = [[0,  30,  0], \
                          [0, -30,  0], \
                          [0, 60,  0], \
                          [0, -60,  0]]
        
        # view down & move up 
        cam_aug_params += [[-i, 0, i/15*1.5] for i in range(10)]


        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if save_dir is not None and iteration in testing_iterations:
                    viz_types = ['render_rgb', 'render_axis_min_scale', 'render_axis_gt_like', 'render_rgb_aug', 'render_rgb_aug_all']
                    viz_dirs = {}
                    for viz_type in viz_types:
                        viz_dir = os.path.join(save_dir, 'results', config['name'], viz_type, str(iteration))
                        viz_dirs[viz_type] = viz_dir
                        os.makedirs(viz_dir, exist_ok=True)

                pbar = tqdm(config['cameras'], total=len(config['cameras']))
                pbar.set_description(f"Rendering {config['name']} images")
                for idx, viewpoint in tqdm(enumerate(config['cameras']), total=len(config['cameras'])):
                    wandb_cond = (idx % 10 ==0)
                    log_imgs = [] 

                    # add rendered image to log candidates
                    image, image_full = render_novelview_image(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene)
                    gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    
                    # do evaluation
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    # log images
                    if save_dir is not None and iteration in testing_iterations:
                        torchvision.utils.save_image(image, os.path.join(viz_dirs['render_rgb'], viewpoint.image_name))
                        if image_full is not None:
                            torchvision.utils.save_image(image_full, os.path.join(viz_dirs['render_rgb'], f"{viewpoint.image_name[:-4]}_with_objects.png"))

                    # add normal image to log candidates
                    gt_norm_rgb = ((viewpoint.original_normal*-1 + 1) * 0.5) * 255 
                    gt_norm_rgb = torch.clip(gt_norm_rgb, min=0, max=255)
                    gt_norm_rgb = gt_norm_rgb.to(torch.uint8)                    

                    # add cov rot axis to log candidates                                                
                    if save_dir is not None and iteration in testing_iterations:
                        render_axis_best, render_axis_best_full = render_novelview_rotaxis(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, idx_best = 'min_scale')
                        torchvision.utils.save_image(render_axis_best / 255, os.path.join(viz_dirs['render_axis_min_scale'], viewpoint.image_name))
                        if render_axis_best_full is not None:
                            torchvision.utils.save_image(render_axis_best_full / 255, os.path.join(viz_dirs['render_axis_min_scale'], f"{viewpoint.image_name[:-4]}_with_objects.png"))
 

                    if save_dir is not None and iteration in testing_iterations:
                        for rx, rz, tz in cam_aug_params:
                            aug_caption = f"Rx: {rx}| Rz: {rz} | tz: {tz}"
                            image, image_full = render_novelview_image(viewpoint, all_bboxes, gaussians, pipe, background, dyn_obj_list, scene, rx, rz, tz)
                            torchvision.utils.save_image(image, os.path.join(viz_dirs['render_rgb_aug'], viewpoint.image_name[:-4] + f"_Rx{rx}_Rz{rz}_tz{tz}.png"))
                            if image_full is not None:
                                torchvision.utils.save_image(image_full, os.path.join(viz_dirs['render_rgb_aug_all'], viewpoint.image_name[:-4] + f"_Rx{rx}_Rz{rz}_tz{tz}.png"))
                            else:
                                torchvision.utils.save_image(image, os.path.join(viz_dirs['render_rgb_aug_all'], viewpoint.image_name[:-4] + f"_Rx{rx}_Rz{rz}_tz{tz}.png"))

                    # log everything
                    if wandb_cond:
                        wandb.log({config['name'] + f"_view_{viewpoint.image_name}":log_imgs}, step=iteration)   


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                wandb.log({ config['name'] + '/loss_viewpoint - l1_loss': l1_test, 
                            config['name'] + '/loss_viewpoint - psnr': psnr_test}, 
                            step=iteration) 
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dp = KITTI360DataParams(parser)
    bp = BoxModelParams(parser)
    sp = SDRegularizationParams(parser) 

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--exp_note', type=str, default="")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[100_000])
    parser.add_argument("--start_checkpoint_dir", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    # initialize wandb -------------------
    if not args.no_wandb:
        dp_cache = dp.extract(args)
        lp_cache = lp.extract(args)
        cur_data = "_" + dp_cache.seq + "_" + str(int(dp_cache.start_frame)) + "_" + str(int(dp_cache.end_frame))
        run = wandb.init(        
            project="vegs_kitti360",       # Set the project where this run will be logged
            name = args.exp_note,       # exp name
            tags = args.exp_note.split('_') + [dp_cache.seq] + [str(dp_cache.start_frame)] + [str(dp_cache.end_frame)] + [lp_cache.data_type], 
            config = parser             # Track hyperparameters and run metadata        
        )
        assert run is wandb.run    
    else:
        run = None
    # -----------------------------------


    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), 
             op.extract(args), 
             pp.extract(args), 
             dp.extract(args),
             bp.extract(args),
             sp.extract(args), 
             args.test_iterations, 
             args.save_iterations, 
             args.checkpoint_iterations, 
             args.start_checkpoint_dir, 
             args.debug_from, 
             exp_note=args.exp_note,
             run=run,
             args=args,
             output_dir=args.output_dir)

    # All done
    print("\nTraining complete.")

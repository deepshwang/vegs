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

import torch
from scene import Scene
import os, gc
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_all, render_dyn, return_gaussians_boxes_and_box2worlds
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, KITTI360DataParams, BoxModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera, make_camera_like_input_camera
import numpy as np
from train import render_novelview_image
from utils.graphics_utils import quaternion_multiply, quaternion_to_matrix, matrix_to_quaternion, decompose_T_to_RS, decompose_box2world_to_RS
import quaternion
from PIL import Image
import cv2


def render_set(model_path, name, iteration, views, scene, pipe, background):
    render_path = os.path.join(model_path, "results", "test", "aug_videos", str(iteration))
    os.makedirs(render_path, exist_ok=True)

                        #Rx  Rz  Tz
    cam_aug_params = [  [0, 0, 0], \
                        [0, 60,  0],  \
                        [0, -60,  0], \
                        [-10, 0, 1] ]

    # Save GT cameras for visualization
    vid_images = []
    for idx, viewpoint_cam in enumerate(tqdm(views[::2][5:], desc="Rendering progress")):
        # img = cv2.imread(viewpoint_cam.image_path, cv2.IMREAD_COLOR)
        img = torch.from_numpy(np.array(Image.open(viewpoint_cam.image_path)))
        vid_images.append(img)
    vid_images = torch.stack(vid_images)
    save_video(vid_images, os.path.join(render_path, "train_cameras.mp4"), 5)

    all_bboxes = scene.getTrainBboxes()

    # Interpolate cameras
    n_jump = 1
    num_interp = 4 * n_jump

    # Use Cam 0 of stereo camera
    remove_first = 5 
    views = views[remove_first:][::2][::n_jump]
    interp_views = []
    for view_t1, view_t2 in zip(views[:-1], views[1:]):
        R1 = view_t1.R
        T1 = view_t1.T # t part of w2c

        w2c = np.eye(4)
        w2c[:3, :3] = R1.transpose(1, 0)
        w2c[:3, 3] = T1.squeeze()
        c2w = np.linalg.inv(w2c)

        R1 = c2w[:3, :3]
        T1 = c2w[:3, 3]

        q1 = matrix_to_quaternion(torch.from_numpy(R1)).numpy()
        q1 = np.quaternion(q1[0], q1[1], q1[2], q1[3])
        
        R2 = view_t2.R
        T2 = view_t2.T # t part of w2c

        w2c = np.eye(4)
        w2c[:3, :3] = R2.transpose(1, 0)
        w2c[:3, 3] = T2.squeeze()
        c2w = np.linalg.inv(w2c)

        R2 = c2w[:3, :3]
        T2 = c2w[:3, 3]

        q2 = matrix_to_quaternion(torch.from_numpy(R2)).numpy()
        q2 = np.quaternion(q2[0], q2[1], q2[2], q2[3])

        for i in range(num_interp):
            q = quaternion.slerp_evaluate(q1, q2, i/(num_interp))
            q = np.array([q.w, q.x, q.y, q.z])
            R = quaternion_to_matrix(torch.from_numpy(q)).numpy()
            T = T1 + (T2 - T1) * i/(num_interp)
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = T
            w2c = np.linalg.inv(c2w)
            R = c2w[:3, :3]
            T = w2c[:3, 3]
            i_cam = Camera(colmap_id=view_t1.colmap_id, 
                                    R=R,  T=T, 
                                    FoVx=view_t1.FoVx, FoVy=view_t1.FoVy, 
                                    K=view_t1.K, 
                                    image=view_t1.original_image,
                                    normal = view_t1.original_normal,
                                    gt_alpha_mask = None,
                                    image_name=view_t1.image_name,
                                    image_path=view_t1.image_path,
                                    normal_path=view_t1.normal_path,
                                    uid=view_t1.uid,
                                    frame=view_t1.frame
                               )
            interp_views.append(i_cam) 


    for i, (rx, rz, tz) in enumerate(cam_aug_params):
        vid_images = []
        for idx, viewpoint in enumerate(tqdm(interp_views, desc="Rendering progress")):
            # Interpolate bounding-box
            interp_idx = idx % num_interp
            interp_bboxes = all_bboxes.copy()
            cur_frame = int(viewpoint.frame)
            next_frame = cur_frame + 1
            if cur_frame in interp_bboxes.keys() and next_frame in interp_bboxes.keys():
                cur_bboxes = all_bboxes[cur_frame]
                next_bboxes = all_bboxes[next_frame]
                for k, bbox in cur_bboxes.items():
                    if k in next_bboxes.keys():
                        s_cur_, cur_r_box2world = decompose_T_to_RS(bbox.box2world)
                        s_next_, next_r_box2world = decompose_T_to_RS(next_bboxes[k].box2world)
                        s_cur = torch.eye(3)
                        s_next = torch.eye(3)
                        s_cur[0, 0] = s_cur_[0, 0]
                        s_cur[1, 1] = s_cur_[0, 1]
                        s_cur[2, 2] = s_cur_[0, 2]
                        s_next[0, 0] = s_next_[0, 0]
                        s_next[1, 1] = s_next_[0, 1]
                        s_next[2, 2] = s_next_[0, 2]
                        s_cur = s_cur.cuda().double()
                        s_next = s_next.cuda().double()

                        cur_box2world = bbox.box2world
                        next_box2world = next_bboxes[k].box2world

                        q1 = matrix_to_quaternion(cur_r_box2world[:3, :3].cpu()).numpy()
                        q1 = np.quaternion(q1[0], q1[1], q1[2], q1[3])

                        q2 = matrix_to_quaternion(next_r_box2world[:3, :3].cpu()).numpy()
                        q2 = np.quaternion(q2[0], q2[1], q2[2], q2[3])

                        q = quaternion.slerp_evaluate(q1, q2, interp_idx/(num_interp))

                        q = np.array([q.w, q.x, q.y, q.z])
                        R = quaternion_to_matrix(torch.from_numpy(q)).numpy()
                        # debug
                        T = cur_box2world[:3, 3] + (next_box2world[:3, 3] - cur_box2world[:3, 3]) * interp_idx/(num_interp)
                        S = s_cur + (s_next - s_cur) * interp_idx/(num_interp)
                        interp_bboxes[cur_frame][k].box2world[:3, :3] = torch.matmul(torch.from_numpy(R).cuda(), S)
                        interp_bboxes[cur_frame][k].box2world[:3, 3] = T.cuda()

            image, image_full = render_novelview_image(viewpoint, interp_bboxes, scene.gaussians, pipe, background, ["car"], scene, rx, rz, tz)
            if image_full is not None:
                vid_images.append(image_full.detach().cpu())
            else:
                vid_images.append(image.detach().cpu())
        vid_images = (torch.clip(torch.stack(vid_images), 0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1)
        _, h, w, _ = vid_images.shape
        save_video(vid_images, os.path.join(render_path, f"render_{rx}_{rz}_{tz}.mp4"))
        
        if i == 0 or i ==3:
            vid_images_2 = vid_images[:, :, w//4:3*w//4, :]
        elif i == 1:
            vid_images_2 = vid_images[:, :, w//2:, :]
        else:
            vid_images_2 = vid_images[:, :, :w//2, :]
        save_video(vid_images_2, os.path.join(render_path, f"render_{rx}_{rz}_{tz}_v2.mp4"))

    res_lr = 200
    half_res_lr = res_lr // 2

    res_ud = 100
    half_res_ud = res_ud // 2

    cam_aug_params_lr = []
    lar = list(range(0, res_lr//2)) + list(range(res_lr//2, -res_lr//2, -1)) + list(range(-res_lr//2, 0))
    for i in lar:
        cam_aug_params_lr.append([0, i/half_res_lr * 60, 0])

    ud = list(range(0, res_ud//2)) + list(range(res_ud//2, 0, -1))
    cam_aug_params_ud = []
    for i in ud:
        cam_aug_params_ud.append([-i/half_res_ud * 10, 0, i/half_res_ud * 1])    

    cam_aug_params_v1 = cam_aug_params_lr + cam_aug_params_ud
    cam_aug_params_v2 = cam_aug_params_ud + cam_aug_params_lr

    for v, cam_aug_params in enumerate([cam_aug_params_v1, cam_aug_params_v2]):
        vid_images = []
        for idx, viewpoint in enumerate(tqdm(interp_views, desc="Rendering progress")):
            rx, rz, tz = cam_aug_params[idx % len(cam_aug_params)]
            image, image_full = render_novelview_image(viewpoint, all_bboxes, scene.gaussians, pipe, background, ["car"], scene, rx, rz, tz)
            if image_full is not None:
                vid_images.append(image_full.detach().cpu())
            else:
                vid_images.append(image.detach().cpu())
        save_video((torch.clip(torch.stack(vid_images), 0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1), os.path.join(render_path, f"demo_v{v}.mp4"))

def get_subdirs(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def save_video(images, outputfile, fps=30):
    basepath, _ = os.path.split(outputfile)
    # give writing permission to basepath
    os.system(f"chmod 777 {basepath}")
    outputVideo = cv2.VideoWriter()
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    size = (images.shape[2], images.shape[1])
    outputVideo.open(outputfile, fourcc, fps, size, True)
    images = images.numpy() 
    for image in images:
        outputVideo.write(image[..., ::-1])  

    outputVideo.release() #close the writer

    return None

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, cfg_kitti : KITTI360DataParams, cfg_box: BoxModelParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        # dataset.eval=True
        scene = Scene(dataset, gaussians, cfg_kitti, cfg_box, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTrainCameras(), scene, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    dp = KITTI360DataParams(parser)
    bp = BoxModelParams(parser) 
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--folder_name", default='test', type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), dp.extract(args), bp.extract(args), args.skip_train, args.skip_test)
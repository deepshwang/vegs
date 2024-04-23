import copy
import math

import torch
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import normal_to_rot, cam_normal_to_world_normal, standardize_quaternion, matrix_to_quaternion, quaternion_to_matrix

@torch.no_grad()
def initialize_gaussians_with_normals(gaussians, scene, pipe, background):
    print(__name__)    
    
    # validataion
    ptr_gs_rot = gaussians._rotation.data_ptr
    ptr_gs_scale = gaussians._scaling.data_ptr

    viewpoint_stack = scene.getTrainCameras()
    n_cameras = len(viewpoint_stack)
    quaternion_new = copy.deepcopy(gaussians._rotation)

    with tqdm(range(n_cameras)) as pbar:
        pbar.set_description("initialize with normal prediction")
        for i in pbar:        
            # estimate rotations from normals ---------------------------
            viewpoint_cam = viewpoint_stack[i]
            norm_pred = viewpoint_cam.original_normal
            _, H, W = norm_pred.shape

            norm_pred_world = cam_normal_to_world_normal(norm_pred, viewpoint_cam.R)
            rot_from_norm = normal_to_rot(norm_pred_world.permute(1,2,0).reshape(-1,3))  # world2newworld      # n_pix, 3, 3  
            quat_from_norm = matrix_to_quaternion(rot_from_norm)
            quat_from_norm = standardize_quaternion(quat_from_norm)
            
            scaling_modifier = 1.0
            tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_cam.image_height),
                image_width=int(viewpoint_cam.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_cam.world_view_transform,
                projmatrix=viewpoint_cam.full_proj_transform,
                sh_degree=gaussians.active_sh_degree,
                campos=viewpoint_cam.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            visibility_mark =rasterizer.markVisible(gaussians.get_xyz)
            visible_xyz = (gaussians.get_xyz[visibility_mark]).unsqueeze(dim=-1)
            R = torch.from_numpy(viewpoint_cam.R).transpose(-1, -2).to(device=visible_xyz.device).type_as(visible_xyz) # world 2 cam
            T = torch.from_numpy(viewpoint_cam.T).to(device=visible_xyz.device).type_as(visible_xyz) # world 2 cam
            K = torch.from_numpy(viewpoint_cam.K).to(device=visible_xyz.device).type_as(visible_xyz) 
            visible_xyz_cam = ((R @ visible_xyz) + T[None, :, None])
            pix = (K @ visible_xyz_cam).squeeze()
            pix /= pix[:, -1:]

            # -1 ~ 1        
            pix[:, 0] = (pix[:, 0]*2 - W)/W
            pix[:, 1] = (pix[:, 1]*2 - H)/H
            
            quat_init = torch.nn.functional.grid_sample(quat_from_norm.permute(1,0).reshape(1, -1, H, W), pix[None, None, :, :-1], mode='nearest', align_corners=True)
            quat_init = standardize_quaternion(quat_init)
            
            mask_zero_quat = (quat_init.abs().squeeze().sum(dim=-2)< 1e-9)
            quaternion_new[visibility_mark] = (quat_init.squeeze().permute(1,0)) * ~mask_zero_quat[:, None] + quaternion_new[visibility_mark] * mask_zero_quat[:, None] # exclude zeros

    quaternion_new = standardize_quaternion(quaternion_new)
    quaternion_new /= (torch.linalg.norm(quaternion_new, axis=-1, keepdims=True) + 1e-9)
    
    gaussians._rotation.data[: ,0] = copy.deepcopy(quaternion_new[:, 0].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,1] = copy.deepcopy(quaternion_new[:, 1].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,2] = copy.deepcopy(quaternion_new[:, 2].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,3] = copy.deepcopy(quaternion_new[:, 3].type_as(gaussians._rotation))
    
    gaussians._scaling[:, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))

    assert gaussians._rotation.data_ptr == ptr_gs_rot
    assert gaussians._scaling.data_ptr == ptr_gs_scale

    return gaussians

def sort_quaternion_candidates(quaternion_full):
    n_pnt_full = quaternion_full.shape[0]
    n_pnt_batch = 512

    for i in range(math.ceil(n_pnt_full / 512)):
        quaternion_full_subset = quaternion_full[i*n_pnt_batch: min((i+1)*n_pnt_batch, n_pnt_full)]
        quaternion_sim = quaternion_full_subset@quaternion_full_subset.permute(0,2,1)
        idx_best_normal = torch.argsort(quaternion_sim.sum(dim=2), dim=1, descending=True)
        quaternion_full[i*n_pnt_batch: min((i+1)*n_pnt_batch, n_pnt_full)] = torch.gather(quaternion_full_subset, index=idx_best_normal[:, :, None].repeat(1,1,4), dim=1)
    return quaternion_full

def refresh_quaternion_new(quaternion_new, idx_accum_quat):
    max_memory = idx_accum_quat.max().item()
    new_memory = int(max_memory *0.7)
    
    # check mem full
    mask_full = (idx_accum_quat >=new_memory)

    quaternion_full = quaternion_new[mask_full]
    quaternion_full = sort_quaternion_candidates(quaternion_full)
    quaternion_new[mask_full] = quaternion_full
    idx_accum_quat[mask_full] = new_memory

    return quaternion_new, idx_accum_quat

def best_quaternion_new(quaternion_new, idx_accum_quat):
    max_memory = quaternion_new.shape[1]
    for n_accum in idx_accum_quat.unique():
        mask = (idx_accum_quat == n_accum)[:, None].repeat(1, max_memory)
        mask[:, n_accum:] = False
        quaternion_accum = quaternion_new[mask].reshape(-1, n_accum, 4)
        quaternion_accum = sort_quaternion_candidates(quaternion_accum)
        quaternion_new[mask] = quaternion_accum.reshape(-1,4)        
    
    return quaternion_new[:, 0, :]

@torch.no_grad()
def initialize_gaussians_with_window_normals(gaussians, scene, pipe, background):
    print(__name__)    
    
    # validataion
    ptr_gs_rot = gaussians._rotation.data_ptr
    ptr_gs_scale = gaussians._scaling.data_ptr
    
    viewpoint_stack = scene.getTrainCameras()
    n_cameras = len(viewpoint_stack)

    max_memory = 100
    quaternion_new = copy.deepcopy(gaussians._rotation).reshape(-1,1,4).repeat(1,max_memory,1) # consider max 100 normal at once
    n_pnt = quaternion_new.shape[0]
    idx_accum_quat = torch.zeros((n_pnt)).to(quaternion_new.device).to(torch.int64)

    with tqdm(range(n_cameras)) as pbar:
        pbar.set_description("initialize with normal prediction")
        for i in pbar:        
            # estimate rotations from normals ---------------------------
            viewpoint_cam = viewpoint_stack[i]
            norm_pred = viewpoint_cam.original_normal
            _, H, W = norm_pred.shape

            norm_pred_world = cam_normal_to_world_normal(norm_pred, viewpoint_cam.R)
            rot_from_norm = normal_to_rot(norm_pred_world.permute(1,2,0).reshape(-1,3))  # world2newworld      # n_pix, 3, 3  
             
            quat_from_norm = matrix_to_quaternion(rot_from_norm)
            quat_from_norm = standardize_quaternion(quat_from_norm)
            
            # match 2d / 3d normal -----------------------            
            scaling_modifier = 1.0
            # Set up rasterization configuration
            tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_cam.image_height),
                image_width=int(viewpoint_cam.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_cam.world_view_transform,
                projmatrix=viewpoint_cam.full_proj_transform,
                sh_degree=gaussians.active_sh_degree,
                campos=viewpoint_cam.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            visibility_mark =rasterizer.markVisible(gaussians.get_xyz) # visibility_mark.shape: torch.Size([2370245])
                    
            visible_xyz = (gaussians.get_xyz[visibility_mark]).unsqueeze(dim=-1)
            R = torch.from_numpy(viewpoint_cam.R).transpose(-1, -2).to(device=visible_xyz.device).type_as(visible_xyz) # world 2 cam
            T = torch.from_numpy(viewpoint_cam.T).to(device=visible_xyz.device).type_as(visible_xyz) # world 2 cam
            K = torch.from_numpy(viewpoint_cam.K).to(device=visible_xyz.device).type_as(visible_xyz) 
            visible_xyz_cam = ((R @ visible_xyz) + T[None, :, None])
            pix = (K @ visible_xyz_cam).squeeze()
            pix /= pix[:, -1:]

            # -1 ~ 1        
            pix[:, 0] = (pix[:, 0]*2 - W)/W
            pix[:, 1] = (pix[:, 1]*2 - H)/H
            
            quat_init = torch.nn.functional.grid_sample(quat_from_norm.permute(1,0).reshape(1, -1, H, W), pix[None, None, :, :-1], mode='nearest', align_corners=True)
            quat_init = standardize_quaternion(quat_init) # torch.Size([1, 4, 1, n_vis])
            
            mask_zero_quat = (quat_init.abs().squeeze().sum(dim=-2)< 1e-9) # torch.Size([n_vis])
            quat_init_valid = quat_init[:, :, :,~mask_zero_quat].squeeze().permute(1,0) # -> quat_init_valid.shape: torch.Size([n_vis_val,4])
            mask3d_visible_valid = visibility_mark.clone()
            mask3d_visible_valid[visibility_mark] *= ~mask_zero_quat # torch.Size([n_pnt])

            quaternion_new[mask3d_visible_valid] = torch.scatter(quaternion_new[mask3d_visible_valid], dim=1, index=idx_accum_quat[mask3d_visible_valid][:, None, None].repeat(1,1,4), src=quat_init_valid[:, None, :])
            
            idx_accum_quat[visibility_mark] +=1

            if idx_accum_quat.max().item() == max_memory:
                quaternion_new, idx_accum_quat = refresh_quaternion_new(quaternion_new, idx_accum_quat)

    quaternion_new = best_quaternion_new(quaternion_new, idx_accum_quat)
    quaternion_new = standardize_quaternion(quaternion_new)
    quaternion_new /= (torch.linalg.norm(quaternion_new, axis=-1, keepdims=True) + 1e-9)

    gaussians._rotation.data[: ,0] = copy.deepcopy(quaternion_new[:, 0].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,1] = copy.deepcopy(quaternion_new[:, 1].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,2] = copy.deepcopy(quaternion_new[:, 2].type_as(gaussians._rotation))
    gaussians._rotation.data[: ,3] = copy.deepcopy(quaternion_new[:, 3].type_as(gaussians._rotation))
    
    gaussians._scaling[:, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))

    assert gaussians._rotation.data_ptr == ptr_gs_rot
    assert gaussians._scaling.data_ptr == ptr_gs_scale

    return gaussians

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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel, GaussianBoxModel
from utils.sh_utils import eval_sh
from typing import List
from utils.graphics_utils import matrix_to_quaternion, quaternion_to_matrix, decompose_T_to_RS, quaternion_multiply

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, depth_image, rendered_cov_quat, rendered_cov_scale, alpha, radii = rasterizer(    
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # original -------------------------
    # rendered_image, radii = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)
    # ------------------------------
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image, # torch.Size([3, 376, 1408]) # torch.float32
            # emjay added ---------------
            'render_depth':  depth_image, 
            'render_cov_quat': rendered_cov_quat,  # torch.Size([4, 376, 1408]) # torch.float32
            'render_cov_scale': rendered_cov_scale, # torch.Size([3, 376, 1408]) # torch.float32
            'alpha': alpha, # torch.Size([1, 376, 1408]) # torch.float32
            # ---------------------------
            "viewspace_points": screenspace_points, # torch.Size([2233571, 3]) # torch.float32
            "visibility_filter" : radii > 0, # torch.Size([2233571]) # torch.bool
            "radii": radii # torch.Size([2233571]) # torch.int32
            }

def prepare_rasterization(viewpoint_camera, pc : GaussianModel, pipe, scaling_modifier = 1.0, override_color = None, box2world=None):
    means3D = pc.get_xyz
    if box2world is not None:
        means3D = torch.cat((means3D, torch.ones(pc.get_xyz.shape[0], 1).cuda()), dim=1)
        means3D = torch.matmul(box2world, means3D.transpose(1, 0).contiguous()).transpose(1, 0).contiguous()
        means3D = means3D[:, :3] / means3D[:, 3:]
    opacity = pc.get_opacity


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    if box2world is not None:    
        box2world_scale, box2world_rot = decompose_T_to_RS(box2world)
        
        ### Rotate R part of covariance
        ## quaternion
        # rotations = quaternion_multiply(rotations ,matrix_to_quaternion(box2world_rot)[None, ...])

        ## quaternion -> matrix -> quaternion
        rotations_mat = quaternion_to_matrix(rotations)
        rotation_mat = torch.matmul(box2world_rot[None, ...], rotations_mat)
        rotations = matrix_to_quaternion(rotation_mat)
        
        ### Adjust S part of covariance
        scales = scales * box2world_scale




    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).contiguous().view(-1, 3, (pc.max_sh_degree+1)**2).contiguous()
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).contiguous()
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return {"means3D": means3D,
            "shs": shs,
            "colors_precomp": colors_precomp,
            "opacities": opacity,
            "scales": scales,
            "rotations": rotations,
            "cov3D_precomp": cov3D_precomp}

def merge_kwargs(render_kwargs, render_kwargs_box):
    for k, v in render_kwargs.items():
        if v is not None and render_kwargs_box[k] is not None:
            render_kwargs[k] = torch.cat((v, render_kwargs_box[k]), dim=0).contiguous()
    return render_kwargs

def render_dyn(viewpoint_camera, pc_boxes : List[GaussianBoxModel], box2worlds: List, pipe, bg_color : torch.Tensor, scaling_modifier = 2.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc_boxes[0].active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    render_kwargs = None
    for box2world, pc_box in zip(box2worlds, pc_boxes):
        render_kwargs_box = prepare_rasterization(viewpoint_camera, pc_box, pipe, scaling_modifier, override_color, box2world=box2world)
        if render_kwargs is None:
            render_kwargs = render_kwargs_box
        else:
            render_kwargs = merge_kwargs(render_kwargs, render_kwargs_box)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    screenspace_points = torch.zeros_like(render_kwargs["means3D"], dtype=render_kwargs["means3D"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    render_kwargs["means2D"] = screenspace_points
    rendered_image, depth_image, rendered_cov_quat, rendered_cov_scale, alpha, radii = rasterizer(**render_kwargs)   
    # original ------------------------------
    # rendered_image, radii = rasterizer()
    # -----------------------------------

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # return {"render": rendered_image,
    #         # emjay added ---------------
    #         'render_depth':  depth_image, 
    #         'render_quat': rendered_quat, 
    #         'render_quat_scale': rendered_quat_scale,
    #         'alpha': alpha,
    #         # ---------------------------
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii}

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image, # torch.Size([3, 376, 1408]) # torch.float32
            # emjay added ---------------
            'render_depth':  depth_image, 
            'render_cov_quat': rendered_cov_quat,  # torch.Size([4, 376, 1408]) # torch.float32
            'render_cov_scale': rendered_cov_scale, # torch.Size([3, 376, 1408]) # torch.float32
            'alpha': alpha, # torch.Size([1, 376, 1408]) # torch.float32
            # ---------------------------
            "viewspace_points": screenspace_points, # torch.Size([2233571, 3]) # torch.float32
            "visibility_filter" : radii > 0, # torch.Size([2233571]) # torch.bool
            "radii": radii # torch.Size([2233571]) # torch.int32
            }


def render_all(viewpoint_camera, pc : GaussianModel, pc_boxes : List[GaussianBoxModel], box2worlds: List, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    render_kwargs = None
    render_kwargs = prepare_rasterization(viewpoint_camera, pc, pipe, scaling_modifier, override_color)

    for box2world, pc_box in zip(box2worlds, pc_boxes):
        render_kwargs_box = prepare_rasterization(viewpoint_camera, pc_box, pipe, scaling_modifier, override_color, box2world=box2world)
        render_kwargs = merge_kwargs(render_kwargs, render_kwargs_box)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    screenspace_points = torch.zeros_like(render_kwargs["means3D"], dtype=render_kwargs["means3D"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    render_kwargs["means2D"] = screenspace_points
    rendered_image, depth_image, rendered_cov_quat, rendered_cov_scale, alpha, radii = rasterizer(**render_kwargs)   
    # original ------------------------------
    # rendered_image, radii = rasterizer()
    # -----------------------------------

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # return {"render": rendered_image,
    #         # emjay added ---------------
    #         'render_depth':  depth_image, 
    #         'render_quat': rendered_quat, 
    #         'render_quat_scale': rendered_quat_scale,
    #         'alpha': alpha,
    #         # ---------------------------
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii}

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image, # torch.Size([3, 376, 1408]) # torch.float32
            # emjay added ---------------
            'render_depth':  depth_image, 
            'render_cov_quat': rendered_cov_quat,  # torch.Size([4, 376, 1408]) # torch.float32
            'render_cov_scale': rendered_cov_scale, # torch.Size([3, 376, 1408]) # torch.float32
            'alpha': alpha, # torch.Size([1, 376, 1408]) # torch.float32
            # ---------------------------
            "viewspace_points": screenspace_points, # torch.Size([2233571, 3]) # torch.float32
            "visibility_filter" : radii > 0, # torch.Size([2233571]) # torch.bool
            "radii": radii # torch.Size([2233571]) # torch.int32
            }


def return_gaussians_boxes_and_box2worlds(bboxes, scene, insts_in_frame, return_adjusted_box=True):
    gaussians_boxes = []
    box2worlds = []
    box_models = []
    for instanceId, bbox in bboxes.items():
            if instanceId in insts_in_frame:
                if return_adjusted_box:
                    box2world = bbox.adjustbox2world()
                else:
                    box2world = bbox.box2world
                box2worlds.append(box2world)
                gaussians_boxes.append(scene.gaussian_box_models[instanceId])
                box_models.append(bbox)
    return gaussians_boxes, box_models, box2worlds
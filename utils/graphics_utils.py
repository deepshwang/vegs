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
import numpy as np
from typing import NamedTuple
import torch.nn.functional as F



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
import numpy as np
from typing import NamedTuple
import torch.nn.functional as F

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array = None

class DynamicPointCloud(NamedTuple):
    points : torch.Tensor
    colors : torch.Tensor
    instances: torch.Tensor
    timestamps: torch.Tensor
    normals : torch.Tensor = None

def decompose_T_to_RS(m):
    R = m[:3, :3]
    S = torch.norm(R, dim=0, keepdim=True)
    R = R / S
    return S, R

def decompose_box2world_to_RS(m):
    T = torch.eye(4).cuda()
    T[:3, 3] = m[:3, 3]
    
    L = torch.zeros(4, 4).cuda()
    L[:3, :3] = m[:3, :3]
    L[3, 3] = 1.0

    R, s = polar_decomp(L)
    S = torch.zeros(3).cuda()
    S[0] = s[0, 0]
    S[1] = s[1, 1]
    S[2] = s[2, 2]
    return S, R[:3, :3]



def polar_decomp(m):   # express polar decomposition in terms of singular-value decomposition
    U, S, Vh = torch.linalg.svd(m)
    u = U @ Vh
    p = Vh.T.conj() @ S.diag().to (dtype = m.dtype) @ Vh
    return  u, p

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )


    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    # o = torch.stack(
    #     (
    #         1 - two_s * (j * j + k * k),
    #         two_s * (i * j + k * r),
    #         two_s * (i * k - j * r),
    #         two_s * (i * j - k * r),
    #         1 - two_s * (i * i + k * k),
    #         two_s * (j * k + i * r),
    #         two_s * (i * k + j * r),
    #         two_s * (j * k - i * r),
    #         1 - two_s * (i * i + j * j),
    #     ),
    #     -1,
    # )
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    o_reshape = o.reshape(quaternions.shape[:-1] + (3, 3))
    return o_reshape

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    '''
        Reference for using K to calculate the projection matrix
        http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    '''
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixwithPrincipalPointOffset(znear, zfar, fovX, fovY, fx, fy, cx, cy, w, h):
    '''
        Reference for refleecting principal point shift to calculate the projection matrix
        http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    '''
    
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top_c = tanHalfFovY * znear
    bottom_c = -top_c
    right_c = tanHalfFovX * znear
    left_c = -right_c

    # Project difference between camera center and half of dimension to znear plane
    dx = (cx - w/2) / fx * znear
    dy = (cy - h/2) / fy * znear

    top = top_c + dy
    bottom = bottom_c + dy 
    left = left_c + dx
    right = right_c + dx    
    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = z_sign * (right + left) / (right - left)
    P[1, 2] = z_sign * (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def normal_to_rot(vector) : # vector shape nx3
    normal = vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True)    
    normal_new = torch.zeros_like(normal)

    while (torch.linalg.vector_norm(normal_new, dim=-1)==0).sum() > 0:
        normal_new = normal + torch.rand(3, device=vector.device) + 1e9
    normal_new = normal_new/torch.linalg.vector_norm(normal_new, dim=-1, keepdim=True)
    
    ortho1 = normal_new - (normal*normal_new).sum(dim=-1, keepdim=True) *  normal
    # ortho1 = torch.linalg.cross(normal, normal_new )
    ortho1 = ortho1/torch.linalg.vector_norm(ortho1, dim=-1, keepdim=True)
    ortho2 = torch.linalg.cross(normal, ortho1 )
    ortho2 = ortho2/torch.linalg.vector_norm(ortho2, dim=-1, keepdim=True)
    return torch.stack((normal, ortho1, ortho2), dim=-1) # col stack
    # return torch.stack((normal, ortho1, ortho2), dim=-2) # row stack

def cam_normal_to_world_normal(norm_pred, R_cam2world):
    # R_cam2world: 3x3
    norm_pred_cam = norm_pred.permute(1, 2, 0).unsqueeze(dim=-1) # torch.Size([376, 1408, 3, 1])
    # norm_pred_cam = norm_pred_cam.squeeze().permute(2,0,1)   # => c, h, w        
    norm_pred_world = (torch.from_numpy(R_cam2world[None, None, :, :]).to(norm_pred.device).float() @ norm_pred_cam)   # torch.Size([376, 1408, 3, 1])
    norm_pred_world = norm_pred_world.squeeze().permute(2,0,1) # => c, h, w            
    return norm_pred_world

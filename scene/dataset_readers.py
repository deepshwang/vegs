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

import torch, torchvision
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud, DynamicPointCloud
from scene.kitti_loader import tracking_calib_from_txt, get_poses_calibration, \
    invert_transformation, get_camera_poses_tracking, get_obj_pose_tracking, get_scene_images, \
    bilinear_interpolate_numpy, bbox_rect_to_lidar, boxes_to_corners_3d, is_within_3d_box, points_to_canonical
import shutil

from glob import glob
from tqdm import tqdm
import math
import torch.nn.functional as F
from model import BoxModel
import open3d as o3d

from kitti360scripts.helpers.project import CameraPerspective as KITTICameraPerspective
from kitti360scripts.devkits.commons.loadCalibration import loadPerspectiveIntrinsic
from kitti360scripts.helpers.annotation import Annotation3D, local2global, KITTI360Bbox3D

class CompactCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    width: int
    height: int
    FovY: np.array
    FovX: np.array
    K: np.array
    image_name: str

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image: np.array
    normal: np.array
    normal_path: str
    image_path: str
    image_name: str
    width: int
    height: int
    frame: int 
    cam_idx: int
    FovY: np.array
    FovX: np.array
    K: np.array

class CompacterSceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list

class CompactSceneInfo(NamedTuple):
    train_cameras: list
    train_bboxes: dict
    inst_info: list

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    dyn_point_cloud: DynamicPointCloud 
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    train_bboxes: dict
    test_bboxes: dict
    instances_info: list

def getNerfppNorm(cam_info, pcd=None):
    def get_center_and_diag(cam_centers):
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    cam_centers = np.hstack(cam_centers)
    
    if pcd is not None:
        points = pcd.points.T
        cam_centers = np.concatenate((points, cam_centers), axis=1)
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}



def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readKITTI3DAnnotations(path, seq, cfg_box):
    basedir = os.path.join(path, "training")
    calibration_path = os.path.join(basedir, 'calib', seq+'.txt')
    oxts_path_tracking = os.path.join(basedir, 'oxts', seq+'.txt') # IMU
    tracklet_path = os.path.join(basedir, 'label_02', seq+'.txt') # 3D annotations
    bboxes = {}
    inst_info = []
    class SimpleBox():
        def __init__(self, R, T):
            self.R = R
            self.T = T

    # Load calibrated intrinsics
    calib = tracking_calib_from_txt(calibration_path)

    # Load extrinsics (pose) from IMU (oxts)
    poses_imu, _, _ = get_poses_calibration(basedir, oxts_path_tracking) # imu2world ?

    # Load 3D annotation
    visible_objects, object_meta, visible_objects_box2world = get_obj_pose_tracking(tracklet_path, poses_imu, calib)
    for timestamp in range(visible_objects_box2world.shape[0]):
        for j, box2world in enumerate(visible_objects_box2world[timestamp]):
            if np.all(box2world != np.ones_like(box2world) * -1):
                obj = SimpleBox(box2world[:3, :3], box2world[:3, 3])
                boxmodel = BoxModel(obj, cfg_box)
                if timestamp not in bboxes.keys():
                    bboxes[timestamp] = {}
                inst_id = visible_objects[timestamp][j][2]
                obj_type = visible_objects[timestamp][j][3]
                if obj_type in [0.0, 2.0]: # car, van, truck
                    bboxes[timestamp][inst_id] = boxmodel
                    inst_info.append(inst_id)
    inst_info = list(set(inst_info))
    return bboxes, inst_info 

def readKITTI3603DAnnotations(path, seq, cfg_box, start_frame=None, end_frame=None, obj_sem_ids=[26, 27, 28, 29, 30]):
    ann = Annotation3D(labelDir=os.path.join(path, 'data_3d_bboxes'), 
                       sequence=seq)
    bboxes = {}
    inst_info = []
    for globalID, timestamps_obj in ann.objects.items():
        timestamps = list(timestamps_obj.keys())
        if start_frame is not None and end_frame is not None:
            timestamps = [t for t in timestamps if t >= int(start_frame) and t < int(end_frame)] # automatically prune -1 where it's static
        for timestamp in timestamps:
            obj = timestamps_obj[timestamp] # This is a kitti360scripts.helpers.annotation.KITTI360BBox3D instance 
            boxmodel = BoxModel(obj, cfg_box)
            inst = globalID # semanticID*1000 + classInstanceID
            sem = obj.semanticId
            if sem not in obj_sem_ids: 
                continue
            inst = globalID  # inst = local2global(sem, local_inst)
            if timestamp not in bboxes.keys():
                bboxes[timestamp] = {}
            bboxes[timestamp][inst] = boxmodel
            
            inst_info.append(inst)
            
    
    return bboxes, list(set(inst_info))


def readKITTI360Cameras(path, seq, start_frame=None, end_frame=None, preload_image=False, cache_dir=None):
    kitti_cams = [KITTICameraPerspective(path, seq=seq, cam_id=0), KITTICameraPerspective(path, seq=seq, cam_id=1)]


    assert np.all(kitti_cams[0].frames == kitti_cams[1].frames), "cam_0 and cam_1 frames don't match! please check."
    frames = sorted(list(kitti_cams[0].frames))
    # Subsample frames that correspond to the stacked pointcloud range
    if start_frame is None:
        sidx = 0
    else:
        if start_frame in frames:
            sidx = frames.index(start_frame)
        else:
            frames_cache = frames + [float(start_frame)]
            frames_cache = sorted(frames_cache)
            sidx = frames_cache.index(start_frame) + 1
    
    if end_frame is None:
        eidx = len(frames)
    else:
        if end_frame in frames:
            eidx = frames.index(end_frame)
        else:
            frames_cache = frames + [float(end_frame)]
            frames_cache = sorted(frames_cache)
            eidx = frames_cache.index(end_frame) - 1

    frames = frames[sidx:eidx]
    
    uid = 0
    cam_infos = []
    for frame in tqdm(frames, total=len(frames)):
        for cam_idx in [0, 1]:
            cam = kitti_cams[cam_idx]
            w2c = np.linalg.inv(cam.cam2world[frame])
            R = np.transpose(w2c[:3, :3]) # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            K = cam.K[:3, :3]
            w = cam.width
            h = cam.height

            focal_length_x = K[0, 0]
            focal_length_y = K[1, 1]
            FovY = focal2fov(focal_length_y, h)
            FovX = focal2fov(focal_length_x, w)

            # Load image
            image_name = f'{str(int(frame)).zfill(10)}.png'
            image_path = os.path.join(path, "data_2d_raw", seq, f"image_{str(cam_idx).zfill(2)}", "data_rect", image_name)
            if preload_image:
                image = Image.open(image_path)
            else:
                image = None
                if len(cache_dir) != 0:
                    image_path_cache = image_path.replace(path, os.path.join(cache_dir, "dataset/3D/KITTI-360"))
                    if not os.path.exists(image_path_cache):
                        os.makedirs(os.path.dirname(image_path_cache), exist_ok=True)
                        shutil.copy(image_path, image_path_cache)
                        image_path = image_path_cache


            # Load normal
            normal_path = os.path.join(path, "data_2d_normal_omnidata_all", seq, f"image_{str(cam_idx).zfill(2)}", image_name.split('.')[0] + '_norm.npy')            
            if preload_image:
                normal = np.load(normal_path)
            else:
                normal=None
                if len(cache_dir) != 0:
                    normal_path_cache = normal_path.replace(path, os.path.join(cache_dir, "dataset/3D/KITTI-360"))
                    if not os.path.exists(normal_path_cache):
                        os.makedirs(os.path.dirname(normal_path_cache), exist_ok=True)
                        shutil.copy(normal_path, normal_path_cache)
                        normal_path = normal_path_cache

            
            cam_info = CameraInfo(uid=uid, R=R, T=T, image=image, FovX=FovX, FovY=FovY, 
                                  image_path=image_path, image_name=image_name, 
                                  normal_path=normal_path, normal=normal,
                                  width=w, height=h, K=K, frame=frame, cam_idx=cam_idx)
            cam_infos.append(cam_info)
            uid += 1
    cam_infos.append(cam_info)
    return cam_infos


def readKITTICameras(path, seq, preload_image=False):
    cam_infos = []
    basedir = os.path.join(path, "training")
    calibration_path = os.path.join(basedir, 'calib', seq+'.txt')
    oxts_path_tracking = os.path.join(basedir, 'oxts', seq+'.txt') # IMU
    
    # Load calibrated intrinsics
    calib = tracking_calib_from_txt(calibration_path)
    focal = calib['P2'][0, 0]

    # Load extrinsics (pose) from IMU (oxts)
    poses_imu, _, _ = get_poses_calibration(basedir, oxts_path_tracking) # imu2world ?
    
    imu2velo = calib['Tr_imu2velo']
    velo2imu = invert_transformation(imu2velo[:3, :3], imu2velo[:3, 3])

    poses_velo = np.matmul(poses_imu, velo2imu)

    for cam_i in range(2, 4): # stereo cameras
        transformation = np.eye(4)
        projection = calib[f'P{cam_i}']
        K_inv = np.linalg.inv(projection[:3, :3])
        R_t = projection[:3, 3]

        t_crect2c = np.matmul(K_inv, R_t)
        transformation[:3, 3] = t_crect2c
        calib[f"Tr_camrect2cam0{cam_i}"] = transformation
        calib[f"K{cam_i}"] = projection[:3, :3]

    cam_poses = get_camera_poses_tracking(poses_velo, calib) # cam2world
    images_path = sorted(get_scene_images(basedir, seq)) 
    for uid, image_path in enumerate(images_path):
        cam_idx = image_path.split("/")[-3].split("_")[-1]
        frame = int(os.path.basename(image_path).split(".")[0])
        cam_pose = cam_poses[uid]
        w2c = np.linalg.inv(cam_pose)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        K = calib[f"K{int(cam_idx)}"]
        image = Image.open(image_path)
        normal_path = image_path.replace("image_", "normal_").replace(".png", "_norm.npy")
        normal = np.load(normal_path)
        w, h = image.size
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, h)
        FovX = focal2fov(focal_length_x, w)
        if not preload_image:
            image = None
            normal = None
        cam_info = CameraInfo(uid=uid, R=R, T=T, image=image, FovX=FovX, FovY=FovY, image_path=image_path,
                                        normal=normal, normal_path=normal_path, 
                                        image_name=os.path.basename(image_path),
                                        width=w, height=h, K=K, frame=frame, cam_idx=cam_idx) 
        cam_infos.append(cam_info)
    return cam_infos

def fetchDynamicPlyKITTI360(path, semantic_ids=[26], visible_only=True):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    instances = vertices['instance']
    timestamps = vertices['timestamp']
    

    if visible_only:
        semantics = vertices['semantic']
        semantic_mask = np.concatenate([np.where(semantics == sid)[0] for sid in semantic_ids], axis=0)
        visibility = np.where(plydata['vertex']['visible'] == 1)[0]
        mask = np.intersect1d(semantic_mask, visibility)
    
        positions = positions[mask]
        colors = colors[mask]
        instances = instances[mask]
        timestamps = timestamps[mask]

    return DynamicPointCloud(points=positions, colors=colors, instances=instances, timestamps=timestamps)

def fetchPlyKITTI360(path, visible_only=True, exclude_lidar=False, exclude_colmap=True, colmap_data_type=''):

    all_positions = []
    all_colors = [] 
    if not exclude_lidar:
        # Fetch velodyne
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        if visible_only:
            visibility = np.where(plydata['vertex']['visible'] == 1)[0]
            positions = positions[visibility]
            colors = colors[visibility]
        all_positions.append(positions)
        all_colors.append(colors)
    
    # Fetch points triangulated with COLMAP
    colmap_path = path.replace("data_3d_semantics", f"data_3d_colmap{colmap_data_type}").replace(".ply", "")
    colmap_ply_path = os.path.join(colmap_path, "points3D.ply")
    
    if not exclude_colmap:
        assert os.path.exists(colmap_ply_path), "Colmap ply file not found!"
        colmap_pcd = fetchPly(colmap_ply_path, return_normals=False)
        all_positions.append(colmap_pcd.points)
        all_colors.append(colmap_pcd.colors)
    all_positions = np.concatenate(all_positions, axis=0).astype(np.float32)
    all_colors = np.concatenate(all_colors, axis=0)

    return BasicPointCloud(points=all_positions, colors=all_colors)

def fetchPly(path, return_normals=True):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    if return_normals:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals=None
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storeDynamicPly(path, xyz, rgb, instance, timestamp):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('instance', '>i4'), ('timestamp', '>i4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, instance[..., None], timestamp[..., None]), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    return ply_data


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    return ply_data

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readKITTI360SceneInfo(path, cfg_box, eval=True, seq="2013_05_28_drive_0009_sync", 
                          start_frame=10703.0, end_frame=11118.0, visible_only=True, llffhold=8, preload_image=False,
                          exclude_lidar=False, exclude_colmap=False, colmap_data_type='', cache_dir=None):
    ### [1-1] Read PointClouds ###
    print("Reading KITTI-360 stacked static pointclouds")
    pcd_filename = os.path.join(path, "data_3d_semantics", "train", seq, "static", f"{str(int(start_frame)).zfill(10)}_{str(int(end_frame)).zfill(10)}.ply")
    pcd = fetchPlyKITTI360(pcd_filename, visible_only=visible_only, exclude_lidar=exclude_lidar, exclude_colmap=exclude_colmap, colmap_data_type=colmap_data_type)

    ### [1-2] Read dynamic pointclouds ###
    dyn_pcd_filename = os.path.join(path, "data_3d_semantics", "train", seq, "dynamic", f"{str(int(start_frame)).zfill(10)}_{str(int(end_frame)).zfill(10)}.ply")
    dyn_pcd = fetchDynamicPlyKITTI360(dyn_pcd_filename)

    ### [2] Read Annotated Cameras ###
    print("Reading KITTI-360 Cameras")
    cam_infos = readKITTI360Cameras(path, seq, start_frame, end_frame, preload_image=preload_image, cache_dir=cache_dir)

    ### [3] Read 3D annotations ###
    bboxes, instances_info = readKITTI3603DAnnotations(path=path, seq=seq, cfg_box=cfg_box, start_frame=start_frame, end_frame=end_frame)
    # split
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # split 3D annotations accordingly
    filterByKey = lambda keys, data: {x: data[x] for x in keys if x in data.keys()}
    train_bboxes = filterByKey([int(cam.frame) for cam in train_cam_infos], bboxes)
    test_bboxes = filterByKey([int(cam.frame) for cam in test_cam_infos], bboxes)
    
    ### [4] Calculate normalizations ###
    nerf_normalization = getNerfppNorm(cam_infos, pcd)


    ### [5] Set ply storage path
    ply_dir = ".cache"
    os.makedirs(ply_dir, exist_ok=True)
    ply_name = f"points3d_{str(start_frame).zfill(10)}_{str(end_frame).zfill(10)}.ply"
    ply_path = os.path.join(ply_dir, ply_name)
    dyn_ply_name = f"dyn_points3d_{str(start_frame).zfill(10)}_{str(end_frame).zfill(10)}.ply"
    dyn_ply_path = os.path.join(ply_dir, dyn_ply_name)

    if not os.path.exists(ply_path):
        disp_name = os.path.join("data_3d_semantics", "train", seq, "gaussians", ply_name)
        print(f"Setting up {disp_name}. Will happen only the first time you open the scene / frame segement.")
        stored_pcd = storePly(ply_path, pcd.points, np.uint8(pcd.colors * 255))
    if not os.path.exists(dyn_ply_path):
        disp_name = os.path.join("data_3d_semantics", "train", seq, "gaussians", dyn_ply_name)
        print(f"Setting up {disp_name}. Will happen only the first time you open the scene / frame segement.")
        stored_pcd = storeDynamicPly(dyn_ply_path, dyn_pcd.points, np.uint8(dyn_pcd.colors * 255), dyn_pcd.instances, dyn_pcd.timestamps)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           dyn_point_cloud=dyn_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           train_bboxes=train_bboxes,
                           test_bboxes=test_bboxes,
                           instances_info=instances_info)
    

    return scene_info 



def generateRandomCameras(n_views, elevation_deg=0, camera_distance=2.0, fov=45.0, width=256, height=256):
    """
    Generate a set of random cameras that are parametrized by azimuth, elevation, field-of-view, and distance from the origin.
    """
    azimuth_deg = np.linspace(0, 360, n_views + 1)[:-1] + np.random.rand(n_views) * 360.0 / n_views
    elevation_deg = np.full_like(azimuth_deg, elevation_deg)
    camera_distance = np.full_like(azimuth_deg, camera_distance)

    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180

    camera_positions = np.stack([camera_distance * np.cos(elevation) * np.sin(azimuth),
                                 camera_distance * np.cos(elevation) * np.cos(azimuth), 
                                 camera_distance * np.sin(elevation)], axis=-1)
    
    center = np.zeros_like(camera_positions)
    up = np.array([0.0, 0.0, 1.0])[None, :]

    fov = fov * math.pi / 180

    lookat = center - camera_positions
    lookat = lookat / np.linalg.norm(lookat)

    right = np.cross(lookat, up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, lookat)
    up = up / np.linalg.norm(up)

    c2w3x4 = np.concatenate(
                       [np.stack([right, up, -lookat], axis=-1), camera_positions[:, :, None]],
                       axis=-1,
                       )
    c2w = np.concatenate(
        [c2w3x4, np.zeros_like(c2w3x4[:, :1])], axis=1
    )
    c2w[:, 3, 3] = 1.0

    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:, :3,:3], (0, 2, 1))  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:, :3, 3] 
    
    fx = width / (np.tan(fov / 2))
    fy = height / (np.tan(fov / 2))
    K = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
    
    #CompactCameraInfo
    cam_infos = []
    for i in range(w2c.shape[0]):
        virtual_image_name = f"virtual_{str(i).zfill(10)}.png"
        cam_info = CompactCameraInfo(uid=i, R=R[i], T=T[i], width=width, height=height, FovY=fov, FovX=fov, K=K, image_name=virtual_image_name)
        cam_infos.append(cam_info)


    return cam_infos



def readKITTISceneInfo(path, cfg, cfg_box, llffhold=8, eval=True):
    seq = cfg.seq
    ### [1-1] Read PointClouds ###
    print("Reading KITTI_tracking stacked static pointclouds")
    pcd_filename = os.path.join(path, "training","3d_semantics", "static", f"{seq}.ply")
    pcd = fetchPlyKITTI360(pcd_filename, visible_only=False)

    ### [1-2] Read dynamic pointclouds ###
    dyn_pcd_filename = os.path.join(path, "training","3d_semantics", "dynamic", f"{seq}.ply")
    dyn_pcd = fetchDynamicPlyKITTI360(dyn_pcd_filename, semantic_ids=[0, 2])

    ### [1-3] Read Cameras
    print("Reading KITTI_tracking Cameras")
    cam_infos = readKITTICameras(path, seq)

    ### [1-4] Read 3D Annotations
    print("Reading KITTI_tracking 3D Annotations")
    bboxes, instances_info = readKITTI3DAnnotations(path, seq, cfg_box)


    # split
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # split 3D annotations accordingly
    filterByKey = lambda keys, data: {x: data[x] for x in keys if x in data.keys()}
    train_bboxes = filterByKey([int(cam.frame) for cam in train_cam_infos], bboxes)
    test_bboxes = filterByKey([int(cam.frame) for cam in test_cam_infos], bboxes)

    ### [4] Calculate normalizations ###
    nerf_normalization = getNerfppNorm(cam_infos, pcd)


    ### [5] Set ply storage path
    ply_dir = ".cache"
    os.makedirs(ply_dir, exist_ok=True)
    ply_name = f"points3d_{seq}.ply"
    ply_path = os.path.join(ply_dir, ply_name)
    dyn_ply_name = f"dyn_points3d_{seq}.ply"
    dyn_ply_path = os.path.join(ply_dir, dyn_ply_name)

    if not os.path.exists(ply_path):
        disp_name = os.path.join("data_3d_semantics", "train", seq, "gaussians", ply_name)
        print(f"Setting up {disp_name}. Will happen only the first time you open the scene / frame segement.")
        stored_pcd = storePly(ply_path, pcd.points, np.uint8(pcd.colors * 255))
    if not os.path.exists(dyn_ply_path):
        disp_name = os.path.join("data_3d_semantics", "train", seq, "gaussians", dyn_ply_name)
        print(f"Setting up {disp_name}. Will happen only the first time you open the scene / frame segement.")
        stored_pcd = storeDynamicPly(dyn_ply_path, dyn_pcd.points, np.uint8(dyn_pcd.colors * 255), dyn_pcd.instances, dyn_pcd.timestamps)
    # try:
    #     pcd = fetchPlyKITTI360(ply_path, visible_only=False, obtain_colmap=False)
    #     dyn_pcd = fetchDynamicPlyKITTI360(dyn_ply_path, visible_only=False)
    # except:
    #     pcd = None
    #     dyn_pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           dyn_point_cloud=dyn_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           train_bboxes=train_bboxes,
                           test_bboxes=test_bboxes,
                           instances_info=instances_info)
    
    return scene_info 



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "KITTI360": readKITTI360SceneInfo,
    "KITTI": readKITTISceneInfo
}
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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, generateRandomCameras
from scene.gaussian_model import GaussianModel, GaussianBoxModel
from arguments import ModelParams, KITTI360DataParams, BoxModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from kitti360scripts.helpers import labels as kittilabels
from typing import Any
from argparse import Namespace

class Scene:

    gaussians: Any
    def __init__(self, args : ModelParams, gaussians : GaussianModel, cfg_kitti: KITTI360DataParams, cfg_box: BoxModelParams, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        # self.spherical_gaussians = spherical_gaussians

        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.train_bboxes = {}
        self.test_bboxes = {}

        if args.data_type == "colmap":
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif args.data_type == "blender":
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif args.data_type == "kitti360":
            print("Loading KITTI-360 Dataset!")
            if load_iteration:
                frames = self.model_path.split("/")[-2].split("_")[-2:]
                seq = "_".join(self.model_path.split("/")[-2].split("_")[:-2])
                cfg_kitti.seq = seq
                cfg_kitti.start_frame = int(frames[0])
                cfg_kitti.end_frame = int(frames[1])
            scene_info = sceneLoadTypeCallbacks["KITTI360"](args.source_path,
                                                            cfg_box=cfg_box, 
                                                            eval=args.eval, 
                                                            seq=cfg_kitti.seq, 
                                                            start_frame=cfg_kitti.start_frame, 
                                                            end_frame=cfg_kitti.end_frame,
                                                            preload_image=args.preload_image,
                                                            exclude_lidar=cfg_kitti.exclude_lidar,
                                                            exclude_colmap=cfg_kitti.exclude_colmap,
                                                            colmap_data_type=cfg_kitti.colmap_data_type,
                                                            cache_dir=args.cache_dir)

        elif args.data_type == "kitti":
            print("Loading KITTI Dataset!")
            scene_info = sceneLoadTypeCallbacks["KITTI"](args.source_path, cfg_kitti, cfg_box)
        else:
            assert False, "Could not recognize scene type!"
        
        self.instances_info = scene_info.instances_info

        # if not self.loaded_iter and not kitti:
        # TODO: Fix this. This will cause possible error when loading with saved iters
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
            
            # Initialize separate spaces for instance gaussians
            for instance in self.instances_info:
                json_cams = []
                camlist = generateRandomCameras(n_views=90)
                save_path = os.path.join(self.model_path, f"instance_{str(instance).zfill(6)}")
                os.makedirs(save_path, exist_ok=True)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(save_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)
                with open(os.path.join(save_path, "cfg_args"), 'w') as cfg_log_f:
                    cfg_log_f.write(str(Namespace(**vars(args))))
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            print("Loading 3D Bounding Boxes")
            self.train_bboxes[resolution_scale] = scene_info.train_bboxes
            self.test_bboxes[resolution_scale] = scene_info.test_bboxes

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, 
                                           self.cameras_extent)

        

        self.gaussian_box_models = {}

        for inst in self.instances_info:
            print(f"Constructing dynamic objects on Semantic: {kittilabels.id2label[26].name}, Instance: {inst}")
            self.gaussian_box_models[inst] = GaussianBoxModel(self.gaussians.max_sh_degree, instanceId = inst)
            if self.loaded_iter:
                try:
                    self.gaussian_box_models[inst].load_ply(os.path.join(self.model_path, f"instance_{str(inst).zfill(6)}", "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
                except:
                    self.gaussian_box_models[inst].initialize(self.cameras_extent, scene_info.dyn_point_cloud, self.train_bboxes[1.0], inst, debug_path=self.model_path, cfg_box=cfg_box)
            else:
                self.gaussian_box_models[inst].initialize(self.cameras_extent, scene_info.dyn_point_cloud, self.train_bboxes[1.0], inst, debug_path=self.model_path, cfg_box=cfg_box)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        for inst, box_gaussian in self.gaussian_box_models.items():
            point_cloud_path = os.path.join(self.model_path, f"instance_{str(inst).zfill(6)}", "point_cloud/iteration_{}".format(iteration))
            os.makedirs(point_cloud_path, exist_ok=True)
            box_gaussian.save_ply(os.path.join(point_cloud_path, f"point_cloud.ply"))
   
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getTrainBboxes(self, scale=1.0):
        return self.train_bboxes[scale]
    
    def getTestBboxes(self, scale=1.0):
        return self.test_bboxes[scale]
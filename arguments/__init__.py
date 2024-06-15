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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.preload_image = False
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.output_dir = "./output"
        self.data_type = "kitti360"
        self.cache_dir = ""
        self.save_results_as_images = False
        self.seed = 7
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 100_000
        self.position_lr_init = 0.000016
        self.position_lr_final = 0.0000016
        self.box_lr_mult = 0.5
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dssim_guidance = 0.0
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_until_iter_box = 50_000 # 50_000 CHANGE
        self.densify_grad_threshold = 0.0002
        
        self.do_normal_guidance = False
        self.normal_initialization = False 
        self.lambda_dnormal = 1e-3 #1e-3 #1e-4 #0.001
        super().__init__(parser, "Optimization Parameters")

class KITTI360DataParams(ParamGroup):
    def __init__(self, parser):
        self.start_frame = 3972
        self.end_frame= 4258
        self.seq = "2013_05_28_drive_0009_sync"
        self.exclude_lidar = False
        self.exclude_colmap = False
        self.colmap_data_type = '_processed'

        super().__init__(parser, "Data Parameters")


class BoxModelParams(ParamGroup):
    def __init__(self, parser):
        self.boxmodel_lr = 0.005
        self.boxmodel_lambda_reg = 0.001
        self.gaussian_box_model_init_opacity = 0.1
        super().__init__(parser, "Box optimization model training Parameters")

class SDRegularizationParams(ParamGroup):
    def __init__(self, parser):
        # Training options
        self.reg_with_diffusion = False 
        self.guidance_mode = "score-matching" # options: score-matching, sds
        self.start_guiding_from_iter = 97_500
        self.end_guiding_at_iter = 100_000
        self.sd_image_size = 512
        self.global_crop = False
        
        # LoRA options
        self.lora_model_dir = "lora/models"
        self.lora_checkpoint_iter = None 
        
        # Stable Diffusion options
        self.sd_model_key = "stabilityai/stable-diffusion-2-1-base"
        self.prompts = "a photography of a suburban street"
        self.negative_prompts = ""
        self.sd_guidance_scale = 7.5
        self.sd_min_step = 0
        self.sd_max_step = 50

        # score-matching options
        self.sm_lambda = 1e-13

        ## SDS options
        self.sds_grad_scale = 1.0
        
        ## yaw(deg) option (+ is looking right, - is looking left) 
        self.yaw_start = 30
        self.yaw_end = 90
        self.yaw_eval = 60
        
        # pitch(deg) option (+ is looking up, - is looking down)
        self.pitch_eval = 0
        self.pitch_start = 0
        self.pitch_end = 0

        self.trans_z_range = 0.5
        self.trans_z_eval = 0

        ## Perceptual loss
        self.perceptual_loss = False
        self.perceptual_loss_lambda = 1.0

        super().__init__(parser, "Stable Diffusion Guidance parameters")

        


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

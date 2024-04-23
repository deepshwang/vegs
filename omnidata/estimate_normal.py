import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
import torchvision
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
from glob import glob
import sys


from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

parser.add_argument('--task', type=str, default='normal' , help="normal or depth")
parser.add_argument('--mode', type=str, default='all', help="all or cropped")

parser.add_argument('--data_dir', type=str, default='/home/nas4_dataset/3D/KITTI-360', help="path to rgb image")

args = parser.parse_args()

root_dir = './omnidata/pretrained_models/'

trans_topil = transforms.ToPILImage()

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# get target task and model
if args.task == 'normal':
    image_size = 384
    
    ## Version 1 model
    # pretrained_weights_path = root_dir + 'omnidata_unet_normal_v1.pth'
    # model = UNet(in_channels=3, out_channels=3)
    # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

    # if 'state_dict' in checkpoint:
    #     state_dict = {}
    #     for k, v in checkpoint['state_dict'].items():
    #         state_dict[k.replace('model.', '')] = v
    # else:
    #     state_dict = checkpoint
    
    
    pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        get_transform('rgb', image_size=None)])
    
    trans_totensor_full = transforms.Compose([transforms.Resize((image_size, image_size), interpolation=PIL.Image.BILINEAR),
                                        get_transform('rgb', image_size=None)])

elif args.task == 'depth':
    image_size = 384
    pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=0.5, std=0.5)])

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose([transforms.Resize((376, 1408), interpolation=PIL.Image.BILINEAR)])


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_dir):
    with torch.no_grad():

        img = Image.open(img_path)
        w, h = img.size

        if args.mode == 'all':        
            img_tensor = trans_totensor_full(img)[:3].unsqueeze(0).to(device)
            output = model(img_tensor).clamp(min=0, max=1)
            output = trans_rgb(output).squeeze()
        else:
            num_crop = w // h + 1
            outputs = []
            for i in range(num_crop):
                if i < num_crop - 1:
                    img_crop = img.crop((i*h, 0, (i+1)*h, h))
                else:
                    img_crop = img.crop((w-h, 0, w, h))
                img_tensor = trans_totensor(img_crop)[:3].unsqueeze(0).to(device)

                if img_tensor.shape[1] == 1:
                    img_tensor = img_tensor.repeat_interleave(3,1)

                output = model(img_tensor).clamp(min=0, max=1)
                if i == (num_crop - 1):
                    output = output[:, :, :, -(w - i*h):]
                outputs.append(output)
            output = torch.cat(outputs, dim=3).squeeze()
        # resize to original size
        output = torchvision.transforms.Resize((h, w), interpolation=TF.InterpolationMode.NEAREST)(output)

        
        # convert output from [0, 1] to [-1, 1]
        pred_norm = (output - 0.5) * 2
        
        # now, it's x-right, y-down, z-backward

        # convert to mj's coordinate (x-left, y-up, z-backward)
        pred_norm[:2, :, :] = pred_norm[:2, :, :] * -1
        pred_norm = pred_norm / torch.norm(pred_norm, dim=0)[None, ...]

        # convert from [-1, 1] to [0, 1] for visualization
        pred_norm_rgb = ((pred_norm.detach().cpu().numpy() + 1) * 0.5) * 255
        pred_norm_rgb = np.clip(pred_norm_rgb, 0, 255).astype(np.uint8)
        im = Image.fromarray(pred_norm_rgb.transpose(1, 2, 0))
        im.save(os.path.join(output_file_dir, img_path.split("/")[-1].split(".")[0] + "_pred_norm.png"))
        np.save(os.path.join(output_file_dir, img_path.split("/")[-1].split(".")[0] + "_norm.npy"), pred_norm.detach().cpu().numpy())

seqs = [f"2013_05_28_drive_{str(i).zfill(4)}_sync" for i in [0, 2, 3, 4, 5, 6, 7, 9, 10]]
for seq in seqs:
    seq_dir = os.path.join(args.data_dir, 'data_2d_raw', seq, "**/*.png")
    img_paths = sorted(glob(seq_dir, recursive=True))
    new_idx = []
    for i in range(len(img_paths)):
        if i %2 == 0:
            new_idx.append(i//2)
        else:
            new_idx.append(len(img_paths)//2 + i//2)
    img_paths = [img_paths[i] for i in new_idx]
    for img_path in tqdm(img_paths, total=len(img_paths)):
        output_file_dir = os.path.join(args.data_dir, f'data_2d_normal_omnidata_{args.mode}', seq, img_path.split("/")[-3])
        os.makedirs(output_file_dir, exist_ok=True)
        save_outputs(img_path, output_file_dir)
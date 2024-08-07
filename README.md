<div align="center">

# [ECCV'24] VEGS: View-Extrapolation of Urban Scenes in 3D Gaussian Splatting using Learned Priors


<font size="6">
<a href="https://deepshwang.github.io/" style="font-size:100%;">Sungwon Hwang</a>*<sup>1</sup>&emsp;
<a href="https://emjay73.github.io" style="font-size:100%;">Min-Jung Kim</a>*<sup>1</sup>&emsp;
<a href="https://keh0t0.github.io" style="font-size:100%;">Taewoong Kang</a><sup>1</sup>&emsp;
<a href="https://vegs.github.io" style="font-size:100%;">Jayeon Kang</a><sup>2</sup>&emsp;
<a href="https://sites.google.com/site/jaegulchoo/" style="font-size:100%;">Jaegul Choo</a><sup>1</sup>&emsp;
</font>

<br>

<font size="4">
*: Equal contribution
</font>


<br>

<font size="6">
<sup>1</sup>KAIST, <sup>2</sup>Ghent University
</font>

<br>

<font size="4">
| <a href="https://vegs3d.github.io">Project Page</a> | <a href="http://arxiv.org/abs/2407.02945">arXiv</a> | <a href="https://github.com/deepshwang/vegs">Code</a> |
</font>

<br>

<br>

<img src="./assets/teaser.gif" alt="teaser.gif" width="1000"/> <br>
<b>Our method aligns and flattens Gaussian covariances to scene surfaces estimated from </br> monocular normal estimations.</b>

<br>

<br>

<img src="./assets/dynamic.gif" alt="dynamic.gif" width="410"/><img src="./assets/object_relocation.gif" alt="object_relocation.gif" width="410"/> <br>
<b>Our method jointly reconstructs static scene with dynamic object such as cars, which can then be relocated arbitrarily.</b>

</div>



## Abstract
Neural rendering-based urban scene reconstruction methods commonly rely on images collected from driving vehicles with cameras facing and moving forward. Although these methods can successfully synthesize from views similar to training camera trajectory, directing the novel view outside the training camera distribution does not guarantee on-par performance. In this paper, we tackle the Extrapolated View Synthesis (EVS) problem by evaluating the reconstructions on views such as looking left, right or downwards with respect to training camera distributions. To improve rendering quality for EVS, we initialize our model by constructing dense LiDAR map, and propose to leverage prior scene knowledge such as surface normal estimator and large-scale diffusion model. Qualitative and quantitative comparisons demonstrate the effectiveness of our methods on EVS. To the best of our knowledge, we are the first to address the EVS problem in urban scene reconstruction. We will release the code upon acceptance.

## Installation

### 1. Requirements

The software requirements are the following:
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions
- CUDA toolkit 11.8 for PyTorch extensions
- C++ Compiler and CUDA SDK must be compatible

Please refer to the original <a href="https://github.com/graphdeco-inria/gaussian-splatting">3D Gaussian Splatting repository</a> for more details about requirements.

### 2. Clone the repository

```shell
# HTTPS
git clone https://github.com/deepshwang/vegs.git --recursive
```

or

```shell
# SSH
git clone git@github.com:deepshwang/vegs.git --recursive
```

### 3. Install packages

Create and activate the environemnt with the required packages installed.

```shell
conda env create -f environment.yml
conda activate vegs
```

## Dataset Preparation

We provide training pipeline for <a href="https://www.cvlibs.net/datasets/kitti-360/">KITTI-360</a> Dataset. Pleaser refer to the data <a href="https://www.cvlibs.net/datasets/kitti-360/documentation.php">documentation</a> for details on the data structure. 

### [1] Download Data
You may register and log-in for <a href="https://www.cvlibs.net/datasets/kitti-360/">KITTI-360</a> page. Then, please download the following data.

```
KITTI-360
└───calibration
└───data_2d_raw
│   └───2013_05_28_drive_{seq:0>4}_sync
└───data_3d_semantics
│   └───train
│       └───static
│           └───{start_frame:0>10}_{end_frame:0>10}.ply
│       └───dynamic
│           └───{start_frame:0>10}_{end_frame:0>10}.ply
└───data_3d_bboxes
│   └───train
│       └───2013_05_28_drive_{seq:0>4}_sync.xml
│   └───train_full
│       └───2013_05_28_drive_{seq:0>4}_sync.xml
└───data_poses
│   └───2013_05_28_drive_{seq:0>4}_sync
```
**Since each sequence is too large to construct as a single scene model, we use scene segment pre-divided by frames, `start_frame` and `end_frame`.**

<br>

### [2-1] ***EITHER*** Triangulate 3D points from training images and known camera poses using COLMAP.
In addition to the LiDAR map, we use points triangulated from training images. To prepare the points, run the following command. (COLMAP must be installed to run)
```
python triangulate.py --data_dir ${KITTI360_DIR}
```

where `${KITTI360_DIR}` is the KITTI-360 data directory. By default, the script will triangulate for all scene sgements in data, and save the results in `data_3d_colmap` and `data_3d_colmap_processed` folder under the KITTI-360 data directory. 

### [2-2] ***OR*** Download Triangluated points for KITTI-360.

You may download the points from <a href="https://drive.google.com/file/d/18jL5VHpU6dhko6RgKYUSVH8XtlR8RPHn/view?usp=sharing">here</a> and save them into `${KITTI360_DIR}/data_3d_colmap_processed`


<br>

### [3-1] ***EITHER*** Prepare Monocular Surface Normal Estimations
We use <a href="https://github.com/EPFL-VILAB/omnidata">omnidata</a> for monocular surface normal estimation. Please download and place the pretrained model in `omnidata/pretrained_models/omnidata_dpt_normal_v2.ckpt`. Running the following scripts will save monocular surface normal estimations in `data_2d_normal_omnidata_all` under the KITTI-360 data directory. To prepare the data, run

```
bash bash_scripts/normal_preprocess_kitti360.sh ${GPU_NUM} ${KITTI360_DIR}
```

### [3-2] ***OR*** Download Monocular Surface Normal Estimations

You may download pre-calculated monocular surface normal estimations from <a href="https://drive.google.com/file/d/1uWLGO5hprrCMz5wggMktEEb0yeVwMA9O/view?usp=sharing">here</a>, and save them into `${KITTI360_DIR}/data_2d_normal_omnidata_all`. 

***Note that the file only contains a frame segment from `3972` to `4258` in sequence `0009` as files for all sequences are too large.***

<br>

### [4-1] ***EITHER*** Prepare training images & Fine-tune with LoRA
To prepare dataset for LoRA training, run the following command.
```
bash bash_scripts/lora_preprocess_kitti360.sh
```
This will prepare square-cropped dataset and save them into `lora/data/kitti360`.

By default, this will prepare images for scene segments listed in `lora/data/kitti360/2013_05_28_drive_train_dynamic_vehicle_human_track_num_vehicles.txt`, which includes scene fragements where vehicles are the only dynamic objects in the scene (as our method cannot handle topologically-varying dynamic objects such as walking people). You may change the text file to only process the scene segment of interest.

We use <a href="https://github.com/huggingface/diffusers">diffusers</a> to train Stable-Diffusion with LoRA. To train, run the following command.

```
bash bash_scripts/lora_train_kitti360.sh ${GPU_NUM}
```

By default, the script will train fine-tuned models for all scene segments listed in `lora/data/kitti360/2013_05_28_drive_train_dynamic_vehicle_human_track_num_vehicles.txt`. 


### [4-2] ***OR*** download pre-trained LoRA weights for KITTI-360.

You may download pre-trained LoRA weights from <a href="https://drive.google.com/file/d/1i2XP2QUIUsxHN1Gg9epMEpS4BUalN0JC/view?usp=sharing">here</a> and unzip them under `lora/models/kitti360`. Again, we only provide models for scene segments listed in `lora/data/kitti360/2013_05_28_drive_train_dynamic_vehicle_human_track_num_vehicles.txt`.

<br>

## Training

To train VEGS for a scene segment of interest, run the following command.

```
bash bash_scripts/train_kitti360.sh ${GPU_NUM} ${DATA_PATH} ${SEQUENCE} ${START_FRAME} ${END_FRAME} ${EXPERIMENT_NOTE}
```

| Parameter | Description | Default |
| :-------: | :--------: | :--------: |
| `${GPU_NUM}`  | Index of GPU to use.|  `0` |
| `${DATA_PATH}`  | Data path |  `./KITTI-360` |
| `${SEQUENCE}`  | Index of sequence to train | `0009`|
| `${START_FRAME}`  | Start frame number of the frame segment | `3972`|
| `${END_FRAME}`  | End frame number of the frame segment | `4258`|
| `${EXP_NOTE}`  | Optional note for the run. </br> The note will be included to the folder that the model will be saved. | `""` |

Trained model and images rendered on conventional and extrapolated cameras will be saved in `output`. 

## Video Rendering

We also provide a script to render and save from camera trajectories, along with novel cameras interpolated between adjacent pairs of the cameras within the trajectory for smooth video rendering.
```
bash bash_scripts/render_video.sh ${GPU_NUM} ${MODEL_PATH}
```

where `${MODEL_PATH}` is the path of the trained gaussian model. Running the script will give you smooth video renderings from both interpolated and extrapolated views. 

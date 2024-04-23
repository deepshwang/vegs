export CUDA_VISIBLE_DEVICES=${1:-0}
DATA_DIR=${2:-"/home/nas4_dataset/3D/KITTI-360"}

python omnidata/estimate_normal.py --data_dir $DATA_DIR
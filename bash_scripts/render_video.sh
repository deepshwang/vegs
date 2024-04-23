export CUDA_VISIBLE_DEVICES=${1:-0}

MODELPATH=${2:-'/home/nas4_user/sungwonhwang/ws/3dgs_kitti/output_new/2013_05_28_drive_0000_sync_0000000372_0000000610/35e47f31-a_full_test'}

python render_video.py -m ${MODELPATH} \
                        --data_type kitti360 \
                        --source_path /home/nas4_dataset/3D/KITTI-360
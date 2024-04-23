export CUDA_VISIBLE_DEVICES=${1:-0}
SEQ=${2:-'0006'}
START_FRAME=${3:-9038}
END_FRAME=${4:-9223}
NOTE=${5:-''}

python train.py -s /home/nas4_dataset/3D/KITTI-360 \
                --seq 2013_05_28_drive_${SEQ}_sync \
                --start_frame ${START_FRAME}\
                --end_frame ${END_FRAME} \
                --exp_note ${NOTE} \
                --do_normal_guidance \
                --normal_initialization \
                --reg_with_diffusion \
                --save_results_as_images \
                --output_dir output \
                --cache_dir /home/$USER/.cache \
                --eval
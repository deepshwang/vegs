export CUDA_VISIBLE_DEVICES=${1:-0}
MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
SEED=1337

python -m accelerate.commands.launch --mixed_precision="fp16"  lora/scripts/train_text_to_image_lora_kitti360.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=300 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --checkpointing_steps=100 \
  --validation_prompt="a photography of a suburban street" \
  --seed=$SEED
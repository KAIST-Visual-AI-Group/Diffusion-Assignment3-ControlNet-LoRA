export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./sample_data/dreambooth-cat"
export OUTPUT_DIR="./runs/dreambooth_cat"

accelerate launch --mixed_precision="no" train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a sks cat" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks cat in a bucket" \
  --validation_epochs=50 \
  --checkpoints_total_limit 2 \
  --seed="0"
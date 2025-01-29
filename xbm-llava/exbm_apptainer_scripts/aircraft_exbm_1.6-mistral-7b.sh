#!/bin/bash
echo "NODE_LIST:"$SLURM_JOB_NODELIST
singularity exec --nv --bind /home/shinya.yamaguchi.mw/dataset:/dataset,$(pwd):/works ~/sif/pytorch-24.03-py3-llava.sif \
torchrun --nproc_per_node=8 train_exbm.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path liuhaotian/llava-v1.6-mistral-7b \
    --version v1 \
    --dataset aircraft \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./result/aircraft_exbm_1.6-mistral-7b \
    --max_epoch 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --init_lr 3e-5 \
    --lr_decay_rate 0.9 \
    --warmup_lr 1e-6 \
    --warmup_step 50 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 10 \
    --num_classes 100 \
    --max_decode_length 50 \
    --min_decode_length 20 \
    --fixed_caps  False \
    --lambda 0.01 \
    --temperature 10.0 \
    --temperature_annealing exp \
    --distributed True \
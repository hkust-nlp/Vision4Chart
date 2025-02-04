#!/bin/bash
#SBATCH --job-name=test          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=8                 # number of GPUs per node
#SBATCH --time=60:00:00        # total run time limit (HH:MM:SS)
#SBATCH --partition=normal      # partition(queue) where you submit
#SBATCH --account=deemreason     # only require for multiple projects


module purge                     # clear environment modules inherited from submission
module load Anaconda3/2023.09-0  # load the exact modules required


export TRANSFORMERS_CACHE='/project/deemreason/junteng/hf_home'
export HF_DATASETS_CACHE='/project/deemreason/junteng/hf_home'
export HF_HOME='/project/deemreason/junteng/hf_home'



export PATH=/home/jliugi/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/home/jliugi/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/home/jliugi/cuda-11.7


# pip install deepspeed

cd LLaVA
# export  NCCL_SOCKET_IFNAME=eno1

nvidia-smi


vision_tower_path='/project/deemreason/junteng/Vision4Math/data/vision4math_clip_model/clip-vit-large-patch14-336'
mm_projector_path='/project/deemreason/junteng/Vision4Math/data/train_llava/llava-v1.5-13b-pretrain/mm_projector.bin'
output_dir='/project/deemreason/junteng/Vision4Math/data/train_llava/llava-v1.5-13b-sampled'

deepspeed --include=localhost:0,1,2,3,4,5,6,7  --master_port=29501 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /project/deemreason/junteng/Vision4Math/data/vicuna-13b/vicuna-13b-v1.5 \
    --version v1 \
    --data_path /project/deemreason/junteng/Vision4Math/data/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k_sampled.json \
    --image_folder /project/deemreason/junteng/Vision4Math/data/llava_data \
    --vision_tower ${vision_tower_path} \
    --pretrain_mm_mlp_adapter ${mm_projector_path} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb



vision_tower_path='/project/deemreason/junteng/Vision4Math/data/vision4math_clip_model/trained_plotqav2_negclip_14_336'
mm_projector_path='/project/deemreason/junteng/Vision4Math/data/train_llava/llava-v1.5-13b-plotqav2_negclip_pretrain/mm_projector.bin'
output_dir='/project/deemreason/junteng/Vision4Math/data/train_llava/llava-negclip-plotqav2-v1.5-13b'

deepspeed --include=localhost:0,1,2,3,4,5,6,7  --master_port=29501 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /project/deemreason/junteng/Vision4Math/data/vicuna-13b/vicuna-13b-v1.5 \
    --version v1 \
    --data_path /project/deemreason/junteng/Vision4Math/data/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k_sampled.json \
    --image_folder /project/deemreason/junteng/Vision4Math/data/llava_data \
    --vision_tower ${vision_tower_path} \
    --pretrain_mm_mlp_adapter ${mm_projector_path} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb


vision_tower_path='/project/deemreason/junteng/Vision4Math/data/vision4math_clip_model/trained_plotqav2_no_hard_clip_14_336'
mm_projector_path='/project/deemreason/junteng/Vision4Math/data/train_llava/llava-v1.5-13b-plotqav2_no_hard_clip_pretrain/mm_projector.bin'
output_dir='/project/deemreason/junteng/Vision4Math/data/train_llava/llava-no_hard_clip-plotqav2-v1.5-13b'

deepspeed --include=localhost:0,1,2,3,4,5,6,7  --master_port=29501 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /project/deemreason/junteng/Vision4Math/data/vicuna-13b/vicuna-13b-v1.5 \
    --version v1 \
    --data_path /project/deemreason/junteng/Vision4Math/data/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k_sampled.json \
    --image_folder /project/deemreason/junteng/Vision4Math/data/llava_data \
    --vision_tower ${vision_tower_path} \
    --pretrain_mm_mlp_adapter ${mm_projector_path} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb


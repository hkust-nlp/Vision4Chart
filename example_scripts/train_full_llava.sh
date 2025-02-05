
export PATH=$PATH:~/.local/bin



vision_tower_path=VISION_PATH  #"data/vision4math_clip_model/clip-vit-large-patch14-336"

export WANDB_ENTITY="junteng_hkust"
export WANDB_PROJECT="LLaVA_finetuning"
export WANDB_NAME="llava-v1.5-13b-chart_mixed_v3-full"



data_path="data/llava_data/chart_mixed_v3/llava_chart_mixed_v3_250k.json"
# vision_tower_path='data/vision4math_clip_model/negclip_fqa_epoch2'
model_path=${output_dir}
output_dir="data/train_llava/llava-v1.5-13b-chart_mixed_v3-full"


deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=29495 LLaVA/llava/train/train_full.py \
    --tune_entire_model True \
    --vision_tower_lr 2e-6 \
    --deepspeed LLaVA/scripts/zero3.json \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${data_path} \
    --image_folder "${PREV_PATH}" \
    --vision_tower ${vision_tower_path} \
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
    --save_steps 5000 \
    --save_total_limit 20 \
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
    --report_to wandb \
    --save_only_model

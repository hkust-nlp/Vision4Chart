# Here is example for openclip training
# in openclip environment
epochs_num=3
train_data=csv_data/plotqa_v3.csv
log_folder=plotqa_v3
pretrained_model=data/openclip-vit-14-336/openclip_model.pt
lr=5e-6
torchrun --nproc_per_node 8 --master_port=29510 -m training.main \
    --batch-size 64 \
    --precision amp \
    --workers 4 \
    --report-to wandb \
    --save-frequency 1 \
    --logs data/trained_openclip/no_hard_negative_logs/${log_folder} \
    --dataset-type csv \
    --csv-separator="," \
    --train-data ${train_data} \
    --csv-img-key img_path \
    --csv-caption-key caption \
    --warmup 0 \
    --lr=${lr} \
    --wd=0. \
    --epochs=${epochs_num} \
    --model ViT-L-14-336 \
    --pretrained ${pretrained_model} \
    --wandb-project-name open-clip-${log_folder} \
    --force-quick-gelu 


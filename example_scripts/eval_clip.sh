model_path=MODEL_PATH
model_notion=MODEL_NOTION
PREV_PATH=. # indicate the img folder path

CUDA_VISIBLE_DEVICES=0 python eval_clip/evaluate_clip.py \
    --input_file_path eval_clip/multi_false/chartbench/val_qa_captions_v2.json \
    --model_file_path ${model_path} \
    --save_result_path eval_clip/multi_false/chartbench/results/v2_${model_notion}.json \
    --wandb_name ${model_notion} \
    --abs_path_prefix ${PREV_PATH} &

CUDA_VISIBLE_DEVICES=1 python eval_clip/evaluate_clip.py \
    --input_file_path eval_clip/multi_false/chartqa/test_qa_captions_v2.json \
    --model_file_path ${model_path} \
    --save_result_path eval_clip/multi_false/chartqa/results/v2_${model_notion}.json \
    --wandb_name ${model_notion} \
    --abs_path_prefix ${PREV_PATH} &

CUDA_VISIBLE_DEVICES=2 python eval_clip/evaluate_clip.py \
    --input_file_path eval_clip/multi_false/dvqa/val_easy_qa_captions.json \
    --model_file_path ${model_path} \
    --save_result_path eval_clip/multi_false/dvqa/results/easy_${model_notion}.json \
    --wandb_name ${model_notion} \
    --abs_path_prefix ${PREV_PATH} &

CUDA_VISIBLE_DEVICES=3 python eval_clip/evaluate_clip.py \
    --input_file_path eval_clip/multi_false/dvqa/val_hard_qa_captions.json \
    --model_file_path ${model_path} \
    --save_result_path eval_clip/multi_false/dvqa/results/hard_${model_notion}.json \
    --wandb_name ${model_notion} \
    --abs_path_prefix ${PREV_PATH} &

CUDA_VISIBLE_DEVICES=4 python eval_clip/evaluate_clip.py \
    --input_file_path eval_clip/multi_false/figureqa/val_qa_captions.json \
    --model_file_path ${model_path} \
    --save_result_path eval_clip/multi_false/figureqa/results/${model_notion}.json \
    --wandb_name ${model_notion} \
    --abs_path_prefix ${PREV_PATH} &

CUDA_VISIBLE_DEVICES=5 python eval_clip/evaluate_clip.py \
    --input_file_path eval_clip/multi_false/plotqa/val_qa_captions.json \
    --model_file_path ${model_path} \
    --save_result_path eval_clip/multi_false/plotqa/results/${model_notion}.json \
    --wandb_name ${model_notion} \
    --abs_path_prefix ${PREV_PATH} &

wait

done
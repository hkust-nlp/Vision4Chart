PREV_PATH=.
model_path=MODEL_PATH
model_notion=MODEL_NOTION

CUDA_VISIBLE_DEVICES=0 python eval_llava/eval_figure/response.py \
    --data_dir 'eval_llava/eval_data/plotqa' \
    --input_file 'plotqa_sampled.json' \
    --output_dir "${PREV_PATH}/eval_llava/eval_data/plotqa/eval_results" \
    --output_file "${model_notion}_responses_sampled.json" \
    --model_path $model_path \
    --temperature 0. \
    --prompt_template 'plotqa' \
    --metric 'relaxed_correctness' \
    --abs_path_prefix ${PREV_PATH} \
    --save_frequency 200 \
    --wandb_name "${model_notion}_plotqa" \
    --summary_file "${model_notion}_plotqa_summary.json" &


CUDA_VISIBLE_DEVICES=1 python eval_llava/eval_figure/response.py \
    --data_dir 'eval_llava/eval_data/figureqa' \
    --input_file 'figureqa_sampled.json' \
    --output_dir "${PREV_PATH}/eval_llava/eval_data/figureqa/eval_results" \
    --output_file "${model_notion}_responses_sampled.json" \
    --model_path $model_path \
    --temperature 0. \
    --prompt_template 'fqa' \
    --metric 'em' \
    --abs_path_prefix ${PREV_PATH} \
    --save_frequency 200 \
    --wandb_name "${model_notion}_fqa" \
    --summary_file "${model_notion}_fqa_summary.json" &

CUDA_VISIBLE_DEVICES=2 python eval_llava/eval_figure/response.py \
    --data_dir 'eval_llava/eval_data/dvqa' \
    --input_file dvqa_hard_sampled.json \
    --output_dir "${PREV_PATH}/eval_llava/eval_data/dvqa/eval_results" \
    --output_file "${model_notion}_hard_responses_sampled.json" \
    --model_path $model_path \
    --temperature 0. \
    --prompt_template 'dvqa' \
    --metric 'em' \
    --abs_path_prefix ${PREV_PATH} \
    --save_frequency 200 \
    --wandb_name "${model_notion}_dvqa" \
    --summary_file "${model_notion}_dvqa_hard_summary.json" &

CUDA_VISIBLE_DEVICES=3 python eval_llava/eval_figure/response.py \
    --data_dir 'eval_llava/eval_data/dvqa' \
    --input_file dvqa_easy_sampled.json \
    --output_dir "${PREV_PATH}/eval_llava/eval_data/dvqa/eval_results" \
    --output_file "${model_notion}_easy_responses_sampled.json" \
    --model_path $model_path \
    --temperature 0. \
    --prompt_template 'dvqa' \
    --metric 'em' \
    --abs_path_prefix ${PREV_PATH} \
    --save_frequency 200 \
    --wandb_name "${model_notion}_dvqa" \
    --summary_file "${model_notion}_dvqa_easy_summary.json" &

CUDA_VISIBLE_DEVICES=4 python eval_llava/eval_figure/response.py \
    --data_dir 'eval_llava/eval_data/chartbench' \
    --input_file 'chartbench_binary.json' \
    --output_dir "${PREV_PATH}/eval_llava/eval_data/chartbench/eval_results" \
    --output_file "${model_notion}_bin_responses_sampled.json" \
    --model_path $model_path \
    --temperature 0. \
    --prompt_template 'chartbench_bin' \
    --metric 'em' \
    --abs_path_prefix ${PREV_PATH} \
    --save_frequency 200 \
    --wandb_name "${model_notion}_chartbench_binary" \
    --summary_file "${model_notion}_chartbench_binary_summary.json" &

CUDA_VISIBLE_DEVICES=5 python eval_llava/eval_figure/response.py \
    --data_dir 'eval_llava/eval_data/chartbench' \
    --input_file 'chartbench_nqa.json' \
    --output_dir "${PREV_PATH}/eval_llava/eval_data/chartbench/eval_results" \
    --output_file "${model_notion}_nqa_responses_sampled.json" \
    --model_path $model_path \
    --temperature 0. \
    --prompt_template 'chartbench_nqa' \
    --metric 'relaxed_cordrectness' \
    --abs_path_prefix ${PREV_PATH} \
    --save_frequency 200 \
    --wandb_name "${model_notion}_chartbench_nqa" \
    --summary_file "${model_notion}_chartbench_nqa_summary.json" &

CUDA_VISIBLE_DEVICES=6 python eval_llava/eval_figure/response.py \
    --data_dir 'eval_llava/eval_data/chartqa' \
    --input_file 'chartqa.json' \
    --output_dir "${PREV_PATH}/eval_llava/eval_data/chartqa/eval_results" \
    --output_file "${model_notion}_responses_sampled.json" \
    --model_path $model_path \
    --temperature 0. \
    --prompt_template 'chartqa' \
    --metric 'relaxed_cordrectness' \
    --abs_path_prefix ${PREV_PATH} \
    --save_frequency 200 \
    --wandb_name "${model_notion}_chartqa" \
    --summary_file "${model_notion}_chartqa_summary.json" &

CUDA_VISIBLE_DEVICES=7 python eval_llava/eval_figure/response.py \
    --data_dir 'eval_llava/eval_data/chartX' \
    --input_file 'chartX.json' \
    --output_dir "${PREV_PATH}/eval_llava/eval_data/chartX/eval_results" \
    --output_file "${model_notion}_responses_sampled.json" \
    --model_path $model_path \
    --temperature 0. \
    --prompt_template 'chartX' \
    --metric 'relaxed_cordrectness_chartX' \
    --abs_path_prefix ${PREV_PATH} \
    --save_frequency 200 \
    --wandb_name "${model_notion}_chartX" \
    --summary_file "${model_notion}_chartX_summary.json" &

wait
echo "done"


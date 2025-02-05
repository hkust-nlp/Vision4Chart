
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
cd eval_llava/Math-LLaVA
cd evaluation_mathvista
PREV_PATH="/cfs/hadoop-aipnlp/zengweihao02/hkust-project/Vision4chart"
declare -a models=(
    # MODEL_PATH
)

declare -a prefixes=(
    # MODEL_NOTION
)

max_jobs=8
current_jobs=0
Index_task=104
Log_file_path=/cfs2/hadoop-aipnlp/zengweihao02/hkust-project/Vision4chart/output_folder/${Index_task}_train_old_nohard_llava.log

for i in "${!models[@]}"; do
    model_path=${models[$i]}
    prefix=${prefixes[$i]}
    gpu=$((i % 8))  # 使用 GPU 0-7

    export CUDA_VISIBLE_DEVICES=$gpu

    python response.py --img_dir /cfs/hadoop-aipnlp/zengweihao02/hkust-project/Vision4chart/data/mathvista --output_dir /cfs2/hadoop-aipnlp/zengweihao02/hkust-project/Vision4chart/eval_llava/Math-LLaVA/evaluation_mathvista/mathvista_outputs --output_file ${prefix}responses.json --model_path ${model_path} --temperature 0.   2>&1 | tee -a ${Log_file_path} &

    ((current_jobs++))

    # 如果当前任务数达到最大值，则等待
    if [[ $current_jobs -ge $max_jobs ]]; then
        wait  # 等待所有并行任务完成
        current_jobs=0  # 重置任务计数
    fi
done

# 等待最后剩余的任务完成
wait
echo "Done!"

for prefix in "${prefixes[@]}"; do
    python extract_answer.py --output_file ${prefix}responses.json
    python calculate_score.py --output_file ${prefix}responses.json --score_file ${prefix}responses_score.json
done
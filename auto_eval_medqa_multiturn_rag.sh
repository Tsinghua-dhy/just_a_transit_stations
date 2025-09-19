#!/bin/bash

# Check if step numbers are provided as arguments
if [ $# -eq 0 ]; then
    echo "错误：请提供步骤编号作为参数（例如：./eval_mmlupro_multiturn.sh 7 10 20 30）"
    exit 1
fi

# Validate that all arguments are numbers
for arg in "$@"; do
    if ! [[ "$arg" =~ ^[0-9]+$ ]]; then
        echo "错误：参数 '$arg' 不是有效的数字"
        exit 1
    fi
done

# Store provided step numbers
STEP_NUMBERS=("$@")

# Define paths
SCRIPT_DIR="$(pwd)"  # 脚本所在目录
MODEL_DIR="/mnt/data4/lwt/model/qwen-2.5-7b-instruct-rl-ours-mmlu-v0.3.23"
INPUT_JSON="/mnt/data4/lwt/eval/dataset/medqa/medqa_test_v0.jsonl"
SUBSET_NUM=-1
TEMPERATURE=0.3
TOP_P=0.5
REPETITION_PENALTY=1.0
MAX_TOKENS=1536
GPU_id="1,2"
gpu_memory_rate=0.95
topk=3
port="5006"
max_rounds=6
# Convert subjects array to a space-separated string
SUBJECTS_STR=$(IFS=" "; echo "${SUBJECTS[*]}")

# Extract version number from MODEL_DIR (e.g., v0.3)

# Function to rename a single file to .bin and return original/new filenames
rename_to_bin() {
    local model_dir=$1
    local step=$2
    local original_file=""
    local new_file=""

    # Construct the filename without padding (e.g., step10)
    original_file="$model_dir/pytorch_model_fp32_step$step"
    new_file="$model_dir/pytorch_model_fp32.bin"

    # Debug: Print the expected file
    echo "调试：尝试查找文件：$original_file" >&2

    if [ -f "$original_file" ]; then
        mv "$original_file" "$new_file"
        echo "已重命名：$original_file -> $new_file" >&2
    else
        echo "警告：文件 $original_file 不存在，跳过..." >&2
        return 1
    fi

    # Output only the filenames for readarray
    echo "$original_file"
    echo "$new_file"
}

# Function to restore a single file
restore_file() {
    local original_file=$1
    local new_file=$2

    if [ -f "$new_file" ]; then
        mv "$new_file" "$original_file"
        echo "已恢复：$new_file -> $original_file" >&2
    else
        echo "警告：文件 $new_file 不存在，无法恢复" >&2
    fi
}

# Main loop to process each step sequentially
for step in "${STEP_NUMBERS[@]}"; do
    echo "处理步骤：$step"

    # Change to model directory
    cd "$MODEL_DIR" || { echo "错误：无法切换到 $MODEL_DIR"; exit 1; }

    # Rename file for the current step
    readarray -t file_info < <(rename_to_bin "$MODEL_DIR" "$step")
    echo "调试：file_info 数组内容：${file_info[*]}" >&2
    echo "调试：file_info 数组长度：${#file_info[@]}" >&2

    if [ ${#file_info[@]} -eq 2 ]; then
        original_file="${file_info[0]}"
        new_file="${file_info[1]}"
        echo "调试：原始文件：$original_file，新文件：$new_file" >&2

        # Return to script directory
        cd "$SCRIPT_DIR" || { echo "错误：无法返回到 $SCRIPT_DIR"; exit 1; }

        # Run evaluation
        echo "运行评估：eval_medqa_multiturn_search_o1.py for step $step"
        python "./eval_medqa_multiturn_search_o1.py" \
            --model_path "$MODEL_DIR" \
            --input_json "$INPUT_JSON" \
            --subset_num $SUBSET_NUM \
            --temperature $TEMPERATURE \
            --top_p $TOP_P \
            --repetition_penalty $REPETITION_PENALTY \
            --max_tokens $MAX_TOKENS \
            --gpu_id $GPU_id \
            --gpu_memory_rate $gpu_memory_rate \
            --topk $topk \
            --max_rounds $max_rounds\
            --port $port 

        # Return to model directory to restore file
        cd "$MODEL_DIR" || { echo "错误：无法切换到 $MODEL_DIR"; exit 1; }

        # Restore original filename
        restore_file "$original_file" "$new_file"
    else
        echo "跳过步骤 $step：文件重命名失败"
    fi

    # Return to script directory for next iteration
    cd "$SCRIPT_DIR" || { echo "错误：无法返回到 $SCRIPT_DIR"; exit 1; }
done

echo "所有步骤处理完成"

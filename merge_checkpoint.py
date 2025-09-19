import os
import argparse
from pathlib import Path
import subprocess
from multiprocessing import Pool
import multiprocessing

def run_cmd(cmd):
    """执行 shell 命令"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        print(f"命令执行成功: {cmd}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {cmd}")
        print(f"错误信息: {e.stderr}")
        raise

def merge_checkpoint(args):
    """合并指定 step 的检查点分片"""
    step, base_ckpt_dir, model_output_dir, version = args
    try:
        step_dir = os.path.join(base_ckpt_dir, f"global_step{step}")
        if not Path(step_dir).exists():
            print(f"step 文件夹不存在: {step_dir}")
            return

        output_model_bin = f"{model_output_dir}/pytorch_model_fp32_step{step}"
        if Path(output_model_bin).exists():
            print(f"step {step} 的输出文件已存在，跳过合并: {output_model_bin}")
            return

        print(f"开始处理 step {step}...")
        merge_cmd = f"python /mnt/data4/lwt/R1-Searcher/scripts/zero_to_fp32.py {step_dir} {output_model_bin} --tag \"\""
        run_cmd(merge_cmd)
        print(f"step {step} 合并完成")
    except Exception as e:
        print(f"处理 step {step} 时发生错误: {str(e)}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="多进程合并检查点分片")
    parser.add_argument("--steps", type=str, required=True, help="要处理的 step 列表，例如 '140,141'")
    parser.add_argument("--version", type=str, required=True, help="模型版本，例如 'qwen-2.5-7b-instruct-rl-ours-mmlu-v0.2.1'")
    args = parser.parse_args()

    # 处理 steps 参数
    try:
        steps = [int(step) for step in args.steps.split(',')]
        if len(steps) != len(set(steps)):
            print("警告：steps 参数中包含重复的 step 值，将自动去重")
            steps = list(set(steps))  # 去重
    except ValueError:
        print("steps 参数格式错误，应为逗号分隔的整数，例如 '140,141'")
        return

    # 设置路径
    base_ckpt_dir = f"/mnt/data4/lwt/R1-Searcher/scripts/trainning_results/results/ckpts/{args.version}/_actor"
    model_output_dir = f"/mnt/data4/lwt/model/{args.version}"

    # 确保输出目录存在
    try:
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"输出目录已创建或存在: {model_output_dir}")
    except Exception as e:
        print(f"创建输出目录失败: {model_output_dir}, 错误: {str(e)}")
        return

    # 设置最大进程数
    max_processes = min(len(steps), multiprocessing.cpu_count(), 64)
    print(f"使用 {max_processes} 个进程进行并行处理")

    # 准备任务参数
    tasks = [(step, base_ckpt_dir, model_output_dir, args.version) for step in steps]

    # 使用进程池并行执行
    with Pool(processes=max_processes) as pool:
        pool.map(merge_checkpoint, tasks)

    print("所有 step 的合并任务已完成")

if __name__ == "__main__":
    main()

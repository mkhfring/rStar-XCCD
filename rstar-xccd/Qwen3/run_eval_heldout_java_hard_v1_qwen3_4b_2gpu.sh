#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-heldout-java-hard-v1
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd

echo "Current directory: $(pwd)"
echo "Running SLURM job script: $0"
echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)"

module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"
export VLLM_USE_V1=0

source ../venv-qwen3/bin/activate

# sft_checkpoints/qwen3-4b-hard-v1 (jobs 3645950/3645952, 2026-09-22):
# fine-tuned on 59 safe leaves from BOTH the java and rust hard-mining
# tracks (eval_data/sft_hard_v1_{java,rust}.jsonl -- see
# Qwen3/run_sft_hard_v1_qwen3_4b_4gpu_fard.sh). Evaluated here ONLY on
# eval_data/rl_heldout_python_java_CLCCD.jsonl (360 q), the split this
# training data's source rl_train_python_java_CLCCD.jsonl was carved
# from -- never on rl_train itself or the full CLCCD test file (that
# would be leakage; SFT_DATA_PROCESS.txt section 2's rule). Compare
# against run_eval_heldout_java_baseline_qwen3_4b_2gpu.sh's result on the
# SAME 360-question file, not FINAL_FOR_CLCCD.md's full-file number.
MODEL="sft_checkpoints/qwen3-4b-hard-v1"
QAF="eval_data/rl_heldout_python_java_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="heldout-hard-v1"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume

#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-clccd-java-hard-codenet-java-only-v2
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

# v2 checkpoint (SFT_DATA_PROCESS.txt section 9), hard-codenet-java-only-v2 on java.
# (v1 note follows, same eval protocol) sft_checkpoints/qwen3-4b-hard-codenet-rust-only-v1:
# fine-tuned on 110 safe leaves (17 unique questions) mined from the "hard"
# subset of eval_data/codenet_train_python_rust.jsonl, pooling the
# standard-budget (3870572/3870573) and wide-budget (3881191) re-mining
# runs -- see SFT_DATA_PROCESS.txt section 8(f). All 17 questions are
# genuine (not binary-blob records, section 8(g)). Rust-only training data,
# so java eval is skipped (same as codenet-rust-only-v1). Track-B CodeNet
# source pool, disjoint from the CLCCD test files, so FULL CLCCD eval is
# valid. Same eval config as the published Qwen3-4B baseline (rust F1
# 0.8739 -- FINAL_FOR_CLCCD.md) and codenet-all-v1 (rust 0.8498).
#
# TIME BUDGET: 72h (3-00:00:00, gpubase_bynode_b4 max) -- raised from the
# 48h used by earlier fine-tuned evals: the fp32-saved fine-tuned
# checkpoints run ~25 q/h on rust (codenet-rust-only-v1, job 3757227:
# 1108/1200 at 44h) and ~35 q/h on java (hard-codenet-java-only-v1, job
# 3921168), i.e. ~48h rust / ~52h java for the full test files.
# RESUME: --no-resume dropped (no prior output exists for this checkpoint,
# so the first run is fresh anyway); if it still times out, just resubmit
# this script unchanged and main.py resumes from the latest output file.
MODEL="sft_checkpoints/qwen3-4b-hard-codenet-java-only-v2"
QAF="eval_data/test_python_java_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="assert-consistency-score"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH"

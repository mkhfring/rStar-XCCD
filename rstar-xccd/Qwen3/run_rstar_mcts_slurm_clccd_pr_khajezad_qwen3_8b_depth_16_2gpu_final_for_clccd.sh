#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-clccd-pr-8b-mcts-d16-final-for-clccd-khajezad
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# See the sibling 4B script for why this cd is absolute rather than derived
# from ${BASH_SOURCE[0]} or $SLURM_SUBMIT_DIR.
cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd

echo "Current directory: $(pwd)"
echo "Running SLURM job script: $0"
echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)"

# The venv's PyTorch links against nccl/2.18.3 (CUDA 12.2 build) via RPATH,
# which hits "named symbol not found" against this node's newer driver during
# multi-GPU NCCL init. Preload a newer NCCL build to override it (RPATH would
# otherwise take precedence over LD_LIBRARY_PATH, so LD_PRELOAD is required).
# Must load modules BEFORE activating the venv: `module load` rebuilds PATH
# and would otherwise demote the venv's python behind the system one.
module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"

# vLLM V1 (the default in 0.8.5.post1) dropped support for the `best_of`
# sampling param this codebase relies on for candidate re-ranking. Force
# the legacy V0 engine, which still supports it and still supports Qwen3.
export VLLM_USE_V1=0

# Activate environment. venv-qwen3, NOT ../venv: the latter pins
# transformers 4.45.2, which predates Qwen3 and fails at config load with
# "model type `qwen3` but Transformers does not recognize this
# architecture" (seen on jobs 2608422/2608423).
source ../venv-qwen3/bin/activate

# Qwen3-8B, MCTS max_depth=16, on the CLCCD-derived Python/Rust
# clone-detection test set (see CLCCD_DATASET_CONSTRUCTION.md).
#
# Search scoring is this branch's assert-consistency-score implementation
# (see FINAL_FOR_CLCCD.md); is_sampling: False in the config, so the search
# never sees the ground-truth label.
#
# --branch tags the output filename with "final-for-clccd" so these runs are
# distinguishable from the 4B runs made on other branches, and can never be
# accidentally resumed from one of them.
# --no-resume forces a fresh output file rather than appending to any
# existing matching run.
#
# 24h rather than the 12h the 4B scripts use: Qwen3-4B depth 16 on
# python-java is tracking ~8h (job 2572325), and 8B roughly doubles the
# per-step cost. With --no-resume a timeout would discard the run, so the
# wall clock is over-requested on purpose. No completed 8B MCTS run exists
# to calibrate against -- every prior attempt was cancelled before it ran.
MODEL="models/Qwen3-8B"
QAF="eval_data/test_python_rust_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_8b_depth_16_2gpu_no_sampling.yaml"
BRANCH="final-for-clccd"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume

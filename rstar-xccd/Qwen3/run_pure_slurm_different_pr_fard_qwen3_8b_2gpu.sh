#!/bin/bash

#SBATCH --time=8:00:00

#SBATCH --ntasks=1

#SBATCH --account=def-fard_gpu

#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:h100:2

#SBATCH --mem=1024G

#SBATCH --job-name=q3-diff-pr-8b-pure

#SBATCH --output=%x_%j.log.out

#SBATCH --error=%x_%j.log.err

# Go to the project root directly. NOT ${BASH_SOURCE[0]} or
# $SLURM_SUBMIT_DIR/..: SLURM copies the script into a spool dir on
# the compute node before running it (so BASH_SOURCE would resolve to
# the spool path), and SLURM_SUBMIT_DIR is wherever `sbatch` was
# invoked from, not this script's location, so a relative ".." is
# fragile to how the job is submitted.

cd /project/6104180/khajezad/rStar-XCCD/rstar-xccd

echo "Current directory: $(pwd)"

echo "Running SLURM job script: $0"

# The venv's PyTorch links against nccl/2.18.3 (CUDA 12.2 build) via RPATH,
# which hits "named symbol not found" against this node's newer driver during
# multi-GPU NCCL init. Preload a newer NCCL build to override it (RPATH would
# otherwise take precedence over LD_LIBRARY_PATH, so LD_PRELOAD is required).
# Must load modules BEFORE activating the venv: `module load` rebuilds PATH
# and would otherwise demote the venv's python behind the system one.
# Qwen3's venv also needs gcc/arrow/python3.11 on top of the CUDA/NCCL
# modules the Qwen2.5 venv gets by with (same set as the working Qwen3 MCTS
# slurm scripts in this directory).
module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"

# vLLM V1 (the default in 0.8.5.post1) has issues with Qwen3 that the
# legacy V0 engine doesn't; force V0 here too, matching the Qwen3 MCTS
# slurm scripts, even though this single-shot pure-inference path doesn't
# use best_of re-ranking.
export VLLM_USE_V1=0

# Activate environment (Qwen3 uses its own venv, separate from Qwen2.5's)

source ../venv-qwen3/bin/activate

# Pure inference: one generate() call per question via vLLM (no rStar
# step/tool-use loop), tensor-parallel across 2 GPUs. Dataset, model, and
# vLLM settings all come from the config file.

CFG="pure_inference/config/qwen3_8b_diff_pr.yaml"

python pure_inference/offline_inference_fine_tuned_model_extended_experiments.py \
  --custom_cfg "$CFG"

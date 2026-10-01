#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# The venv's PyTorch links against nccl/2.18.3 (CUDA 12.2 build) via RPATH,
# which hits "named symbol not found" against this node's newer driver during
# NCCL init. Preload a newer NCCL build to override it (RPATH would
# otherwise take precedence over LD_LIBRARY_PATH, so LD_PRELOAD is required).
# Must load modules BEFORE activating the venv: `module load` rebuilds PATH
# and would otherwise demote the venv's python behind the system one.
module load StdEnv/2023 gcc arrow/15.0.1 opencv/4.14.0 python/3.11 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"

# vLLM V1 (the default in 0.8.5.post1) dropped support for the `best_of`
# sampling param this codebase relies on for candidate re-ranking. Force
# the legacy V0 engine, which still supports it and still supports Qwen3.
export VLLM_USE_V1=0

source ../venv-qwen3/bin/activate

MODEL="models/Qwen3-8B"
QAF="eval_data/test_same_python_java_sample10.jsonl"
CFG="config/my_test_mcts_sample10.yaml"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL"

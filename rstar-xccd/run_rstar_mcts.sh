#!/bin/bash

MODEL="models/Qwen2.5-Coder-3B-Instruct"
QAF="eval_data/test_clone.jsonl"
CFG="config/my_test_mcts.yaml"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL"

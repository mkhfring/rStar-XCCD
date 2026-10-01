#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-sanity20-trackA-rust-v2-strict-tokfix
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

# Sanity check for the train_SFT.py special-token fix (2026-09-25): the
# trackA_rust_v2_strict checkpoint retrained WITHOUT added special tokens, on
# 20 rl_heldout_strict questions (10 clone / 10 non-clone). The broken
# checkpoint wrote code in 0/110 trees and left 52/110 unjudged; the base
# model writes code in every tree. Pass = most trees contain code steps and
# the no-judgment rate is back near the base model's.
MODEL="sft_checkpoints/qwen3-4b-trackA-rust-v2-strict-tokfix/checkpoint-68"
QAF="eval_data/sanity20_heldout_strict_python_rust_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="sanity20-trackA-rust-v2-strict-tokfix"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH"

python3 - <<'EOF'
import glob, json
from evaluate_clone_results import node_sort_key
from rescore_runs import tested
out = sorted(f for f in glob.glob("eval_data/sanity20_*.mcts.*sanity20-trackA-rust-v2-strict-tokfix*.jsonl"))[-1]
n = code = real = 0
for line in open(out):
    t = json.loads(line).get("rstar")
    if not t:
        continue
    n += 1
    numeric = {k: v for k, v in t.items() if node_sort_key(k) is not None}
    code += any("<code>" in (v.get("text") or "") + (v.get("action_input") or "") for v in numeric.values())
    real += any(tested(numeric, k) for k in numeric)
print(f"SANITY: trees with a <code> step: {code}/{n} | trees with a real test: {real}/{n}  ({out})")
EOF

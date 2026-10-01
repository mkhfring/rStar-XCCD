#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --job-name=q3-4b-e16a-sft-dev-rust
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd

echo "Current directory: $(pwd)"
echo "Running SLURM job script: $0"
echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)"

module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7 java/17.0.6
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"
export VLLM_USE_V1=0

source ../venv-qwen3/bin/activate

# E16a: SFT model (Claude python-only traces) on the E1 dev set -> choose ITS aggregation rule.
# CodeNet dev set (eval_data/e16a/build_codenet_disjoint.py), python-only arm, 1 GPU.
# Arm "pyonly": config/qwen3_4b_ablation_pyonly_1gpu.yaml (header explains the arm).
# Resume: resubmit this script unchanged if it times out.
# java/17.0.6 provides javac/java for Code 2 (missing in the first submission
# of the java dual-exec arm, 4018936, which was cancelled and restarted).
MODEL="sft_checkpoints/qwen3-4b-e16a-claude-pyonly"
QAF="eval_data/e16a/dev_python_rust_codenet.jsonl"
CFG="config/qwen3_4b_ablation_pyonly_1gpu.yaml"
BRANCH="e16a-sft-pyonly"

START=$(date +%s)
python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH"
echo "WALLCLOCK_SECONDS: $(( $(date +%s) - START ))"

OUT=$(ls -t eval_data/e16a/dev_python_rust_codenet_depth_16.jsonl.mcts.qwen3-4b-e16a-claude-pyonly.e16a-sft-pyonly.*.jsonl | head -1)
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]; L = "rust".capitalize()
n = tried = compiled = blocked = 0
for line in open(out):
    t = json.loads(line).get("rstar")
    if not t:
        continue
    n += 1
    txt = "\n".join((v.get("text") or "") for v in t.values() if isinstance(v, dict))
    compiled += f"Code 2 ({L}) compiled" in txt
    blocked += "Code 2 cannot be executed in this setting" in txt
    tried += (f"Code 2 ({L})" in txt) or ("Code 2 staged" in txt) or ("external crate" in txt) or ("Code 2 cannot be executed" in txt)
print(f"E16A SFT DEV rust: trees {n} | tried to run Code 2 {tried} | compiled {compiled} | blocked (python-only) {blocked}")
PY
python rescore_runs.py "$OUT" --lang rust   # use the 'full' column only: dev indices are NOT CLCCD indices

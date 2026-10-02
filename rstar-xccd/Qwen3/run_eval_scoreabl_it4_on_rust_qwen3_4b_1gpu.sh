#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --job-name=q3-4b-e1-dev-pyonly-rust
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# Twin guard (both GPU accounts; see twin_guard.inc)
source /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd/Qwen3/twin_guard.inc

cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd

echo "Current directory: $(pwd)"
echo "Running SLURM job script: $0"
echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)"

module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7 java/17.0.6
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"
export VLLM_USE_V1=0

source ../venv-qwen3/bin/activate

# SEARCH-SCORE ABLATION (plan section 00b, 2026-10-01): dev set, iterations 4, scores on.
# CodeNet dev set (eval_data/e16a/build_codenet_disjoint.py), python-only arm, 1 GPU.
# Arm "pyonly": config/qwen3_4b_scoreabl_it4_on_1gpu.yaml (header explains the arm).
# Resume: resubmit this script unchanged if it times out.
# java/17.0.6 provides javac/java for Code 2 (missing in the first submission
# of the java dual-exec arm, 4018936, which was cancelled and restarted).
MODEL="models/Qwen3-4B"
QAF="eval_data/e16a/dev_python_rust_codenet.jsonl"
CFG="config/qwen3_4b_scoreabl_it4_on_1gpu.yaml"
BRANCH="scoreabl-it4-on"

START=$(date +%s)
python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH"
echo "WALLCLOCK_SECONDS: $(( $(date +%s) - START ))"

OUT=$(ls -t eval_data/e16a/dev_python_rust_codenet_depth_16.jsonl.mcts.Qwen3-4B.scoreabl-it4-on.*.jsonl | head -1)
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
print(f"SCORE ABLATION it4 on rust: trees {n} | tried to run Code 2 {tried} | compiled {compiled} | blocked (python-only) {blocked}")
PY
python rescore_runs.py "$OUT" --lang rust   # use the 'full' column only: dev indices are NOT CLCCD indices

#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-fard_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --job-name=q3-4b-ablation-dualexec-java
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

# LOCKED hard-negative evaluation set (hardeval, 2026-10-01): extension confirmation runs. Do not tune on these results.
# base Qwen3-4B, frozen E0 pipeline, 1 GPU, full CLCCD java test set, seed = run 1.
# Arm "dualexec": config/qwen3_4b_hard_autocode2_javav3_1gpu.yaml (header explains the arm).
# Resume: resubmit this script unchanged if it times out.
# java/17.0.6 provides javac/java for Code 2 (missing in the first submission
# of the java dual-exec arm, 4018936, which was cancelled and restarted).
MODEL="models/Qwen3-4B"
QAF="eval_data/e16a/hardeval_python_java_codenet.jsonl"
CFG="config/qwen3_4b_hard_autocode2_javav3_1gpu.yaml"
BRANCH="hard-locked-ext"

START=$(date +%s)
python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH"
echo "WALLCLOCK_SECONDS: $(( $(date +%s) - START ))"

OUT=$(ls -t eval_data/e16a/hardeval_python_java_codenet_depth_16.jsonl.mcts.Qwen3-4B.hard-locked-ext.*.jsonl | head -1)
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]; L = "java".capitalize()
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
print(f"HARD autocode2-javav3 (java prompt v3) java: trees {n} | tried to run Code 2 {tried} | compiled {compiled} | blocked (python-only) {blocked}")
PY
python rescore_runs.py "$OUT" --lang java   # full column only

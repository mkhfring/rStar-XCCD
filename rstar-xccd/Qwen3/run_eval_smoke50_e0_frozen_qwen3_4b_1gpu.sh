#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --job-name=q3-4b-eval-smoke50-e0-frozen-1gpu
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

# E0 smoke test (EXPERIMENT_PLAN_2026-09-26.txt): base Qwen3-4B, FROZEN config
# (1 GPU, rust_crate_check strict) on 50 rust pairs = the 16 pairs whose
# Code 2 the legacy crate check wrongly refused + 34 random others (seed 0).
# Pass = on the 16, Rust Code 2 compiles and runs whenever the model tries to
# execute it, and no "external crate(s)" refusal appears for them.
# --no-resume is not needed: the branch tag is new.
MODEL="models/Qwen3-4B"
QAF="eval_data/smoke50_e0_python_rust_CLCCD.jsonl"
CFG="config/qwen3_4b_frozen_e0_1gpu.yaml"
BRANCH="e0-smoke50-frozen-1gpu"

START=$(date +%s)
python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH"
echo "WALLCLOCK_SECONDS: $(( $(date +%s) - START )) for 50 pairs on 1 GPU"

python3 - <<'EOF'
import glob, json
out = sorted(glob.glob("eval_data/smoke50_e0_*.mcts.Qwen3-4B.e0-smoke50-frozen-1gpu*.jsonl"))[-1]
fixed = {32, 46, 279, 282, 286, 292, 305, 346, 361, 431, 443, 632, 694, 699, 831, 947}
tried = compiled = refused = 0
code_trees = n = 0
for line in open(out):
    r = json.loads(line)
    t = r.get("rstar") or {}
    n += 1
    texts = [v.get("text") or "" for v in t.values() if isinstance(v, dict)]
    code_trees += any("<code>" in x for x in texts)
    if r["index"] in fixed:
        blob = "\n".join(texts)
        tried += "candidate_code2" in blob or "Code 2 (Rust)" in blob or "external crate" in blob
        compiled += "Code 2 (Rust) compiled" in blob
        refused += "external crate" in blob
print(f"SMOKE: {n} trees, {code_trees} with a <code> step ({out})")
print(f"SMOKE: previously-refused pairs (16): Code 2 staged {tried}, compiled OK {compiled}, "
      f"refused as external crate {refused}  -> PASS if refused == 0 and compiled == tried")
EOF

#!/bin/bash
#SBATCH --account=def-khajezad
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=rust-output-check
#SBATCH --output=/lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD-train/rstar-xccd/eval_data/train/rust_build_rate_2026-10-06/slurm-outputs-%j.out
cd /lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD-train/rstar-xccd
unset LD_PRELOAD
export PATH=$HOME/.cargo/bin:$PATH
module load java/17 2>/dev/null
../venv-qwen3/bin/python tools/rust_crates/check_outputs.py --builds eval_data/train/rust_build_rate_2026-10-06/builds.jsonl --out eval_data/train/rust_build_rate_2026-10-06 --workers 30

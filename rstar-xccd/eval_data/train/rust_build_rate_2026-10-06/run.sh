#!/bin/bash
#SBATCH --account=def-khajezad
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=rust-build-rate
#SBATCH --output=/lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD-train/rstar-xccd/eval_data/train/rust_build_rate_2026-10-06/slurm-%j.out
cd /lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD-train/rstar-xccd
unset LD_PRELOAD
export PATH=$HOME/.cargo/bin:$PATH
../venv-qwen3/bin/python tools/rust_crates/measure_build_rate.py --out eval_data/train/rust_build_rate_2026-10-06 --workers 30

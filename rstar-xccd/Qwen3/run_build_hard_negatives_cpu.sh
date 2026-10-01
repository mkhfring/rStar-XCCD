#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=build-hard-negatives
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# Hard-negative pilot set (PAPER_COMPLETION_PLAN_2026-09-30, minimum plan item 2):
# same-problem Accepted-Python vs Wrong-Answer-Java/Rust pairs, verified on the
# official CodeNet sample inputs. CPU only.
cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd
module load StdEnv/2023 python/3.11 java/17.0.6
source ../venv-data/bin/activate
export PATH=$HOME/.cargo/bin:$PATH
export JAVA_TOOL_OPTIONS="-Xmx1g"

for L in java rust; do
  python eval_data/e16a/build_hard_negatives.py --lang $L --n 100 --n_cross 50 --workers 14
done

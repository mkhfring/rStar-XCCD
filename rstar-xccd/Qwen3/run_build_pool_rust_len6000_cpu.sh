#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=build-pool-rust-len6000
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# TRAINING_PLAN_2026-09-30 step 2: problem-level pool of verified clones and
# same-problem Wrong-Answer non-clones (build_training_pool.py docstring). CPU only.

cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd
module load StdEnv/2023 python/3.11 java/17.0.6
source ../venv-data/bin/activate
export PATH=$HOME/.cargo/bin:$PATH
export JAVA_TOOL_OPTIONS="-Xmx1g"

for L in rust; do
  python eval_data/e16a/build_training_pool.py --lang $L --workers 30 --max_len 6000 --suffix _len6000
done

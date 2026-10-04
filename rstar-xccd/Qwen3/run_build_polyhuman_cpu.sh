#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=build-polyhuman
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# External PolyHuman hard-negative set (Python-Java), see build_polyhuman_set.py docstring. CPU only.
# Rules fixed before any model run (user go-ahead 2026-10-03): 300 verified hard negatives + 300 clones + 150 cross.
cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd
module load StdEnv/2023 python/3.11 java/17.0.6
source ../venv-data/bin/activate
export JAVA_TOOL_OPTIONS="-Xmx1g"

python eval_data/e16a/polyhuman_extract_tests.py
python eval_data/e16a/build_polyhuman_set.py --n 300 --n_cross 150 --workers 30
chmod a-w eval_data/e16a/polyhuman_python_java.jsonl eval_data/e16a/polyhuman_python_java_meta.jsonl

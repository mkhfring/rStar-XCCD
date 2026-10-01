# Twin guard (source at the top of a launcher, before any work).
# The same job is queued twice: under def-khajezad_gpu with job name NAME and
# under def-fard_gpu with job name NAME-fard. Whichever starts first wins: it
# cancels the pending twin; a job whose twin is already running exits.
# Submit both with:  sbatch --account=def-khajezad_gpu --job-name=NAME ... &&
#                    sbatch --account=def-fard_gpu     --job-name=NAME-fard ...
if [ "$SLURM_JOB_ACCOUNT" = "def-fard_gpu" ]; then
  TWIN_ACCOUNT="def-khajezad_gpu"; TWIN_NAME="${SLURM_JOB_NAME%-fard}"
else
  TWIN_ACCOUNT="def-fard_gpu"; TWIN_NAME="${SLURM_JOB_NAME}-fard"
fi
TWIN_RUNNING=$(squeue -h -u "$USER" -A "$TWIN_ACCOUNT" -n "$TWIN_NAME" -t RUNNING -o %i | sort -n | head -1)
# Tie-break: if both copies start at the same moment each sees the other RUNNING
# (happened 2026-09-30: both exited, no work done). The LOWER job id proceeds.
if [ -n "$TWIN_RUNNING" ] && [ "$TWIN_RUNNING" -lt "$SLURM_JOB_ID" ]; then
  echo "Twin job $TWIN_RUNNING ($TWIN_ACCOUNT, lower id) is already running; exiting without work."
  exit 0
fi
if [ -n "$TWIN_RUNNING" ]; then
  echo "Twin job $TWIN_RUNNING ($TWIN_ACCOUNT) also started; this job has the lower id and proceeds (twin exits)."
fi
for t in $(squeue -h -u "$USER" -A "$TWIN_ACCOUNT" -n "$TWIN_NAME" -t PENDING -o %i); do
  echo "Started first: cancelling pending twin $t ($TWIN_ACCOUNT)"
  scancel "$t"
done

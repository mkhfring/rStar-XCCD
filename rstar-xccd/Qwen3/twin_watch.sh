#!/bin/bash
# Cancels the still-pending twin of each rl-mining job once its counterpart under the
# other account (def-khajezad_gpu vs def-fard_gpu) starts running. Exits when every pair is resolved.
A="def-khajezad_gpu"; B="def-fard_gpu"
NAMES="q3-4b-rl-mining-hard-java q3-4b-rl-mining-hard-rust q3-4b-rl-mining-codenet-java q3-4b-rl-mining-codenet-rust"
ids() { squeue -h -u "$USER" -A "$1" -n "$2" -t "$3" -o %i; }
declare -A done_
while true; do
  left=0
  for n in $NAMES; do
    [ -n "${done_[$n]}" ] && continue
    a_run=$(ids $A "$n" RUNNING);   a_pen=$(ids $A "$n" PENDING)
    b_run=$(ids $B "$n-fard" RUNNING); b_pen=$(ids $B "$n-fard" PENDING)
    if [ -n "$a_run" ] && [ -n "$b_pen" ]; then scancel $b_pen; echo "$(date +%T) $n: $A job $a_run running -> cancelled $B twin $b_pen"; done_[$n]=1
    elif [ -n "$b_run" ] && [ -n "$a_pen" ]; then scancel $a_pen; echo "$(date +%T) $n: $B job $b_run running -> cancelled $A twin $a_pen"; done_[$n]=1
    elif [ -n "$a_run$b_run" ] || [ -z "$a_pen$b_pen" ]; then done_[$n]=1
    else left=1; fi
  done
  [ $left -eq 0 ] && { echo "$(date +%T) all pairs resolved"; exit 0; }
  sleep 60
done

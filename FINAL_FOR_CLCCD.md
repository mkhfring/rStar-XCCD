# `final-for-clccd` — the frozen CLCCD configuration

This branch pins the exact combination the CLCCD results are reported from,
so the numbers can be regenerated without reconstructing which branch
contributed which half.

| layer | choice | source |
|---|---|---|
| MCTS search scoring | **assert-consistency-score** | `rstar_deepthink/` restored byte-identical to `079fe32` |
| Leaf aggregation | **clone-on-disagreement** (default) | `evaluate_clone_results.py` |
| Execution-signature override | **on** (`no_code+blocked+assertion_failed`) | `evaluate_clone_results.py` |

`python3 evaluate_clone_results.py <run>.jsonl` with no flags produces the
reported numbers. `_result` on this branch means this configuration.

## Results

Depth 16, `n_generate_sample=2`, `is_sampling: False`, two rollouts.

| config | n | previously published | **final** | P | R | TP | FP | TN | FN | no-judg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B python↔java | 1800 | 0.9002 | **0.9454** | 1.0000 | 0.8964 | 805 | 0 | 900 | 93 | 2 |
| Qwen3-4B python↔rust | 1200 | 0.7773 | **0.8739** | 0.9676 | 0.7967 | 239 | 8 | 892 | 61 | 0 |
| Qwen2.5-Coder-3B python↔java | 1800 | 0.8897 | **0.9226** | 0.9860 | 0.8669 | 775 | 11 | 889 | 119 | 6 |
| Qwen2.5-Coder-3B python↔rust | 1200 | 0.6711 | 0.6576 | 0.5994 | 0.7282 | 217 | 145 | 752 | 81 | 5 |

"Previously published" is majority vote with no override — the scorer every
`_result` file dated before 2026-09-07 used. Those reports are preserved
next to each run as `_result.majority.no-exec-signature`, so both numbers
remain regenerable from the same rollout file.

Qwen3-4B python↔java reaches 0.9454 **at precision 1.0000** — 0 false
positives in 1800 instances. The whole gain is recall (0.8185 → 0.8964).

### The python↔rust / Qwen2.5-Coder-3B exception

That row **loses** 0.0135 under this configuration and is the one cell where
the branch default is the wrong choice. Both layers stacked here are
precision-asymmetry bets, and this model does not satisfy the precondition:
its clone-side precision is 0.63 and the execution-signature trigger
population is only 0.589 pure clone, against 0.94–1.00 for every other
config. `predict_label()`'s docstring already says not to enable
clone-on-disagreement there, and the same reasoning applies to the override.

Score that one configuration explicitly:

```bash
python3 evaluate_clone_results.py <run>.jsonl --aggregation majority --no-exec-signature
```

which reproduces 0.6711. Whether to report the uniform configuration across
all four cells, or the per-config best, is a presentation decision — but it
has to be stated either way, because 0.6711 and 0.6576 come from the same
rollout file.

## Reproducing

```bash
git checkout final-for-clccd

# final numbers (no flags needed)
python3 rstar-xccd/evaluate_clone_results.py <run>.jsonl

# the pre-2026-09-07 scorer, for comparison
python3 rstar-xccd/evaluate_clone_results.py <run>.jsonl \
    --aggregation majority --no-exec-signature

# validation tables for the override
python3 rstar-xccd/validate_exec_signature.py
```

## What differs from `exec-outcome-scoring`

1. `rstar_deepthink/` is the assert-consistency scorer, not the
   exec-outcome one. `mcts.py` scores assertion-bearing branches by
   self-consistency with their eventual conclusion; it does **not** charge
   `negative_reward` for `no_code`/`blocked`, and does not persist
   `state["exec_outcome"]`.
2. The default aggregation is `clone-on-disagreement` rather than
   `majority`.

Everything else — the override, `validate_exec_signature.py`, the audit
docs — carries over unchanged.

### The observation fallback is load-bearing here

Because this branch's `tree.py` predates `state["exec_outcome"]`, the
override depends entirely on `exec_outcome_of()`'s fallback, which recovers
the three outcomes from `state["observation"]` by exact match against the
sentinels `tree.py` emits. Verified on
`Qwen3-4B.assert-consistency-score.20260904210649`: 0 trees carry the
persisted field, 176 trees fire the override through the fallback. If that
fallback is ever removed, the override silently becomes a no-op on this
branch rather than failing loudly.

### Run scripts are not versioned

`.gitignore:234` ignores `*.sh`, so no SLURM launcher is tracked and the
branch cannot control them. In particular the four
`*_exec_outcome_scoring.sh` scripts still sit in `Qwen3/` and `Qwen2.5/`
with `BRANCH="exec-outcome-scoring"`. Running one **while this branch is
checked out** would execute assert-consistency scoring and tag the output
`.exec-outcome-scoring.`, silently corrupting the provenance of the
comparison. Use the `*_assert_consistency_score.sh` scripts here.

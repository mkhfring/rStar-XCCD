# Evaluator Changes: `--exec-signature`

> **On the `final-for-clccd` branch** the default aggregation is
> `clone-on-disagreement`, not `majority`, and the search scoring is
> assert-consistency rather than exec-outcome. See `FINAL_FOR_CLCCD.md`.
> Everything below describes the override itself, which is identical on
> both branches.

## Why this exists

`rstar-xccd/evaluate_clone_results.py` turns a finished MCTS run into
precision/recall/F1. Until now it only ever read the **leaves** of each tree:
collect every `final_answer` that normalizes to clone/non-clone, combine them
by majority vote (or, since the previous change, by
`clone-on-disagreement`), and report. Everything else in the tree — including
the record of what the code steps actually did — was ignored.

This change adds an override that reads the **code-execution outcomes**
recorded on interior nodes and answers `clone` when the tree contains a step
that never actually ran. It came out of an audit of the rollout trees
(see "Where this came from" below) which found that this signal is a
near-perfect clone detector on Qwen3, and that it was already present in
every run ever produced — including runs that predate the branch which
introduced it.

**As of 2026-09-07 this is the default.** Scoring a run applies the
override unless you pass `--no-exec-signature`. The rationale and the one
configuration where you should pass it are in "When not to use it".

## What changed, concretely

| | previous evaluator | current evaluator |
|---|---|---|
| Inputs read per tree | leaf `final_answer` only | leaf `final_answer` + interior `exec_outcome` / `observation` |
| `predict_label()` signature | `(rstar_tree, aggregation=MAJORITY)` | `(rstar_tree, aggregation=MAJORITY, exec_signature=frozenset())` |
| `evaluate()` signature | `(input_path, aggregation=MAJORITY)` | `(input_path, aggregation=MAJORITY, exec_signature=frozenset())` |
| New helpers | — | `exec_outcome_of(node)`, `tree_exec_outcomes(tree)` |
| New CLI flags | — | `--exec-signature [SET]` (on by default), `--no-exec-signature` |
| `predict_label` / `evaluate` default | n/a | `exec_signature=DEFAULT_EXEC_SIGNATURE` (on) |
| Instances with no parsable leaf verdict | always counted as "no judgment" | classified if a trigger fired, else "no judgment" |
| Report header | `Leaf aggregation:` | adds `Exec-signature override:` + how many instances it fired on |
| `stats` dict | — | adds `exec_signature`, `exec_signature_fired` |

136 insertions, 4 deletions. The 4 deleted lines are the two function
signatures and their call sites; no existing logic was rewritten.

### Backward compatibility

The previous behaviour is reachable with `--no-exec-signature`, which takes
exactly the path the evaluator took before — the override is checked inside
`if exec_signature and ...`, which is then false. Verified by diffing a
regenerated report against the published `_result` file for
`test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen2.5-Coder-3B-Instruct.exec-outcome-scoring.20260906233732.jsonl`:
identical apart from the new `Exec-signature override: off` header line.

**This is a breaking change to what `_result` means.** Every `_result` file
in `eval_data/` dated before 2026-09-07 is leaf-vote-only; every one written
after is not. The header line `Exec-signature override:` distinguishes them,
and re-scoring a run whose existing report used a different setting prints a
warning to stderr rather than replacing it silently. Non-default settings
still write to their own suffixed path (`_result.no-exec-signature`,
`_result.<aggregation>`, `_result.exec-signature-<triggers>`).

Three callers are pinned to `exec_signature=frozenset()` because they exist
to reproduce numbers published under the old scorer, and would otherwise
change silently:

| caller | why pinned |
|---|---|
| `analyse_search_signals.py:206` | its "F1 now" column is documented to equal the run's own `_result` |
| `figures/make_discussion_figures.py:290,303` | the discussion figures were published against leaf-vote scoring |
| `validate_exec_signature.py:73` | load-bearing — the override must not be folded into the baseline it is measured against |

`main.py:176` and
`pure_inference/offline_inference_fine_tuned_model_extended_experiments.py:537`
are deliberately *not* pinned: they score fresh runs, which is exactly where
the new default should apply.

### Reading outcomes from runs that predate `exec_outcome`

`mcts.py` only started persisting `state["exec_outcome"]` on the
`exec-outcome-scoring` branch. Older runs still carry
`state["observation"]`, and the three outcomes this override cares about are
recoverable from it **exactly**, not heuristically:

| outcome | recovered from | source |
|---|---|---|
| `no_code` | `observation == NO_CODE_MESSAGE` | fixed sentinel, `tree.py:186` |
| `blocked` | `observation.startswith(JAVA_BLOCKED_PREFIX)` | fixed sentinel, `tree.py:188` |
| `assertion_failed` | `observation.startswith("AssertionError")` | interpreter formats the type name, `tree.py:202` |

Anything else is reported as `"ran"`; nothing here needs to tell `ok` from
`error` apart. This is what makes the validation below possible: the rule can
be tested on `baseline` and `assert-consistency-score` trees, not only on the
branch that introduced the field.

Note this is *not* the discredited `"error" not in observation.lower()` test
that `MCTS_SEARCH_SCORING.md` documents removing. That test tried to infer
*whether an exception happened* from prose, and could not distinguish a
sentinel from a successful run. These three are exact string matches against
constants the code itself emits.

## The rule

> If any node in the tree has an execution outcome in
> {`no_code`, `blocked`, `assertion_failed`}, answer `clone`.
> Otherwise fall through to the chosen aggregation.

It is orthogonal to `--aggregation` and composes with it.

**Why it is not arbitrary.** Only Code 1 is executable in this harness —
`tree.py` writes just the Python side to `candidate_code.py` and blocks the
Java/Rust side. So "the model never produced runnable Python", "it reached
for the blocked side", and "its own invented assertion failed" all mean the
same thing: *this branch could not settle the comparison by running
anything*. Empirically that happens almost only on genuinely equivalent
pairs.

Ground truth of the population each trigger fires on:

| run | trigger | fires | clone | non-clone | purity | accuracy there |
|---|---|---:|---:|---:|---:|---:|
| java/Q3 baseline | `no_code` | 131 | 131 | 0 | **1.000** | 0.740 |
| java/Q3 baseline | `assertion_failed` | 114 | 114 | 0 | **1.000** | 0.702 |
| java/Q3 assert | `no_code` | 123 | 123 | 0 | **1.000** | 0.805 |
| java/Q3 assert | `assertion_failed` | 115 | 115 | 0 | **1.000** | 0.730 |
| java/Q3 exec | `no_code` | 85 | 85 | 0 | **1.000** | 0.741 |
| java/Q3 exec | `assertion_failed` | 93 | 93 | 0 | **1.000** | 0.645 |
| rust/Q3 assert | `no_code` | 125 | 121 | 4 | 0.968 | 0.776 |
| rust/Q3 assert | `assertion_failed` | 75 | 73 | 2 | 0.973 | 0.693 |
| java/Q2.5 assert | `assertion_failed` | 224 | 224 | 0 | **1.000** | 0.879 |
| rust/Q2.5 baseline | `no_code` | 158 | 93 | 65 | 0.589 | 0.595 |

The last row is the exception, and it is the whole reason this is opt-in —
see "When not to use it".

Note the `accuracy there` column: on the population the trigger identifies,
the evaluator was previously getting 0.59–0.88 against ~0.93 elsewhere.
The trigger marks precisely the instances the runs were losing, which is
why the gain lands almost entirely in recall.

## Effect

`n` is instances scored; `+sig` is `--exec-signature`; `+sig+cod` adds
`--aggregation clone-on-disagreement`.

| run | n | majority | +sig | +sig+cod | P (maj → +sig) | R (maj → +sig) |
|---|---:|---:|---:|---:|---|---|
| rust/Q3 baseline | 1200 | 0.7692 | 0.8629 | 0.8711 | 0.9744 → 0.9516 | 0.6355 → 0.7893 |
| rust/Q3 assert | 1200 | 0.7773 | 0.8713 | 0.8739 | 0.9897 → 0.9713 | 0.6400 → 0.7900 |
| rust/Q3 exec | 1200 | 0.7603 | 0.8614 | 0.8640 | 0.9840 → 0.9668 | 0.6195 → 0.7767 |
| java/Q3 baseline | 1800 | 0.8957 | 0.9348 | 0.9398 | 1.0000 → 1.0000 | 0.8112 → 0.8775 |
| java/Q3 assert | 1800 | 0.9002 | 0.9335 | 0.9454 | 1.0000 → 1.0000 | 0.8185 → 0.8753 |
| java/Q3 exec ⁺ | 1361 | 0.8938 | 0.9386 | **0.9485** | 1.0000 → 1.0000 | 0.8080 → 0.8843 |
| rust/Q2.5 baseline | 1200 | 0.6109 | 0.6388 | 0.6516 | 0.6130 → 0.5920 | 0.6088 → 0.6936 |
| rust/Q2.5 assert | 1200 | 0.6711 | **0.6595** | 0.6576 | 0.6591 → 0.6073 | 0.6835 → 0.7215 |
| rust/Q2.5 exec | 1200 | 0.6427 | 0.6558 | 0.6599 | 0.6254 → 0.5893 | 0.6611 → 0.7391 |
| java/Q2.5 baseline | 1800 | 0.8770 | 0.9000 | 0.9053 | 0.9901 → 0.9816 | 0.7872 → 0.8309 |
| java/Q2.5 assert | 1800 | 0.8897 | 0.9149 | 0.9226 | 0.9958 → 0.9858 | 0.8041 → 0.8535 |
| java/Q2.5 exec ⁺ | 1525 | 0.8744 | 0.9063 | 0.9150 | 0.9914 → 0.9889 | 0.7821 → 0.8365 |

⁺ Still running when this table was generated (2026-09-07, jobs 2572325 /
2572328, ~6h15m into a 12h limit). Regenerate once they finish.

On Qwen3/java precision stays at exactly **1.0000** while recall gains
6–8 points. Response rate also improves slightly, because a trigger can
classify a tree that never produced a parsable verdict (java/Q2.5 baseline:
12 unjudged → 1).

## Validation

The trigger set was **not** chosen by reading whole-run F1. That is
selection on the test set — the objection already recorded against
`clone-on-disagreement` in `predict_label()`'s docstring, and it applies
here with equal force. The set was chosen by a 50× split-half protocol: fit
the trigger subset on a random half, score the F1 gain on the held-out half.

| run | n | base F1 | held-out gain | sd | modal fitted set |
|---|---:|---:|---:|---:|---|
| rust/Q3 baseline | 1200 | 0.7692 | **+0.0955** | 0.018 | `assertion_failed+no_code` [50/50] |
| rust/Q3 assert | 1200 | 0.7773 | **+0.0928** | 0.015 | `assertion_failed+no_code` [50/50] |
| rust/Q3 exec | 1200 | 0.7603 | **+0.0992** | 0.012 | `assertion_failed+no_code` [50/50] |
| java/Q3 baseline | 1800 | 0.8957 | +0.0372 | 0.005 | `assertion_failed+blocked+no_code` [29/50] |
| java/Q3 assert | 1800 | 0.9002 | +0.0329 | 0.004 | `assertion_failed+no_code` [50/50] |
| java/Q3 exec | 1341 | 0.8959 | +0.0438 | 0.007 | `assertion_failed+no_code` [50/50] |
| rust/Q2.5 baseline | 1200 | 0.6109 | +0.0225 | 0.014 | `assertion_failed+no_code` [36/50] |
| rust/Q2.5 assert | 1200 | 0.6711 | +0.0010 | 0.007 | `assertion_failed` [37/50] |
| rust/Q2.5 exec | 1200 | 0.6427 | +0.0080 | 0.009 | `no_code` [38/50] |
| java/Q2.5 baseline | 1800 | 0.8770 | +0.0223 | 0.005 | `assertion_failed+blocked+no_code` [42/50] |
| java/Q2.5 assert | 1800 | 0.8897 | +0.0245 | 0.005 | `assertion_failed+blocked+no_code` [43/50] |
| java/Q2.5 exec | 1505 | 0.8740 | +0.0311 | 0.005 | `assertion_failed+blocked+no_code` [49/50] |

Three things this shows:

1. **The gain survives held-out selection.** It is not an artefact of having
   looked at the answers.
2. **The trigger set is stable** — the same subset wins in 50/50 splits on
   both Qwen3 runs.
3. **The gain is the same size on every scoring branch.** rust/Q3:
   +0.0955 (baseline) / +0.0928 (assert) / +0.0992 (exec). The signal is
   independent of how the search was scored, which is the point — it says
   nothing about the search and everything about the classifier on top of
   it.

Still a caveat worth stating in any write-up: split-half re-uses the same
1200/1800 CLCCD instances for fitting and scoring. It rules out *trigger-set*
overfitting, not dataset-level idiosyncrasy. A genuinely held-out dataset
would be stronger.

## When not to use it

**Low clone-side precision.** On `rust/Q2.5 assert` the default set gives
**−0.0116** — the only negative cell in the table. Since the override is now
on by default, that configuration must be scored with
`--no-exec-signature` explicitly. Same precision-asymmetry
condition that governs `clone-on-disagreement`: conceding cases to `clone`
is close to free when the clone side is precision-heavy (Qwen3: P ≈ 0.97–1.00)
and is a coin flip when it is not (Qwen2.5/rust: P ≈ 0.63, trigger purity
0.589). Do not enable it there.

The rule of thumb, stated as a condition rather than a model name: check the
trigger population's purity first (table above, or section 1 of the
validation script). Above ~0.94 the override pays; near 0.59 it does not.

## Where this came from

An audit of the rollout trees, prompted by `exec-outcome-scoring` failing to
beat `assert-consistency-score`. That audit found the same
`no_code`/`blocked`/`assertion_failed` signal was already being fed to the
**search** as `negative_reward` (`mcts.py:193`), and that as a search signal
it is worth approximately nothing:

- no answer leaf is ever backed up — `need_value_func=False` short-circuits
  the backup block in `select_next_step()` (`solver.py:271` →
  `mcts.py:332`), so every leaf in every run has `value = q_value =
  visit_count = 0`, verified exhaustively;
- consequently, in all 264 trees where two leaves disagree, `q_value` is
  tied between the competing labels — the search score cannot break a
  single tie;
- 79% of trees have only one answer leaf, so there is nothing to aggregate;
- the value redirected the search in 2–11% of trees, and where
  `exec-outcome-scoring` redirected it, accuracy *fell* versus baseline on
  the same instances (rust/Q3 0.767 vs 0.861; rust/Q2.5 0.533 vs 0.726);
- oracle accuracy — correct if *any* leaf carries the right label — is only
  0.911–0.926 against 0.901–0.908 for majority, so no aggregation or
  reranking scheme can gain more than ~1.5 points.

This is the same conclusion `SEARCH_SIGNAL_DESIGN_SPACE.md` reaches from a
different direction ("search-signal work is capped by branch diversity, not
by signal quality"), and it is why the change landed in the evaluator rather
than in `mcts.py`: the signal is a good **classification feature** over a
finished tree and a poor **search reward**, and spending it as the latter
was what made `exec-outcome-scoring` underperform.

## Usage

```bash
# default -- exec-signature override ON, validated trigger set
python3 evaluate_clone_results.py <run>.jsonl

# pre-2026-09-07 behaviour, reproduces every _result published before then
python3 evaluate_clone_results.py <run>.jsonl --no-exec-signature

# explicit subset
python3 evaluate_clone_results.py <run>.jsonl --exec-signature no_code+assertion_failed

# stacked with clone-on-disagreement (best Qwen3/java configuration)
python3 evaluate_clone_results.py <run>.jsonl --aggregation clone-on-disagreement
```

## Reproduction

Every table above is regenerated by:

```bash
python3 rstar-xccd/validate_exec_signature.py
```

Section 1 prints the base rates, section 2 the split-half protocol, section 3
the shipped default applied across all runs. `RUNS` at the top of that file
lists the runs explicitly rather than globbing `eval_data/`, so a stray
re-scored copy cannot silently join the table.

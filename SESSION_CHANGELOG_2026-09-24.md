# Session changelog, 2026-09-24 (Claude Code, Opus 5.5)

This is the trace for everything changed, created, measured and submitted in
the 2026-09-24 evening session on branch `rl-self-improvement`. **Nothing was
committed.** Use this file to review, commit, or roll back; use the pointers
at the end to continue the work. Methodology text for the paper is drafted
separately in `METHODOLOGY_ADDITIONS_2026-09-24.tex`.

Related documents updated in the same session (appended sections only):

| document | new section |
|---|---|
| `SFT_DATA_PROCESS.txt` | section 9, "Process-level curation, teacher traces and preference data" |
| `MCTS_SEARCH_SCORING.md` | last section, "Scoring review, 2026-09-24" |
| `rstar-xccd/SESSION_STATUS_2026-09-24.txt` | parts G, H, I, J (job tracker) |

---

## 1. Findings, in order of importance

Every number below is reproducible with the command given in section 5.

1. **The published baselines and all checkpoint results use different
   aggregation rules.** `FINAL_FOR_CLCCD.md` (rust 0.8739, java 0.9454) was
   scored before 2026-09-12, when `predict_label()` applied the exec-signature
   override to every tree. Every `_result` written since then (all SFT
   checkpoints) applies it only to trees with no leaf vote. Same baseline
   trees under the current rule: rust **0.7952**, java **0.9163**. Under one
   rule, CodeNet SFT improved on the base model (codenet-all-v1 rust 0.8498,
   java 0.9425), so the recorded "flat or worse than base" conclusion
   compared two different metrics. `methodology.tex` describes the current
   rule but reports the old-rule numbers.
2. **A leaf-aggregation signal: "tested non-clone".** The prompt tells the
   model to skip testing when it judges a pair non-clone. A non-clone leaf
   whose path ran a real test is therefore usually a case where the model
   leaned clone and then trusted a failed guess. On base-model trees these
   leaves are ground-truth clone 91-96% of the time (`rl_train`) and 29/29 on
   held-out; leaves using the prompt's shortcut are right 93-94% of the time.
   Counting tested non-clone leaves as clone votes, with the rule chosen on
   `rl_train` and reported on held-out data:

   | base Qwen3-4B | current rule | new rule |
   |---|---|---|
   | rust `rl_heldout`, run 1 | 0.804 | 0.891 |
   | rust `rl_heldout`, run 2 (independent) | 0.792 | 0.885 |
   | rust `rl_heldout_strict`, runs 1 / 2 | 0.800 / 0.717 | 0.879 / 0.876 |
   | java `rl_heldout`, run 1 / run 2 (partial) | 0.908 / 0.907 | 0.963 / 0.921 |

   It is model-dependent: fine-tuned checkpoints also test real non-clones,
   and the rule lowers their F1 (codenet-all-v1 rust 0.850 -> 0.796). Only
   report it with per-run calibration (`rescore_runs.py --calibrate`).
   Confirmation runs with fresh trees are submitted (section 6).
3. **CodeNet data cannot fix rust recall.** On the clean CodeNet rust pool
   Qwen3-4B's recall is already 0.983 (17 false negatives in 1000 clones),
   against 0.797 on CLCCD rust. The entire CodeNet hard set, rust and java, is
   two AOJ problems (ALDS1_2_B selection sort, ALDS1_4_D allocation).
4. **The CLCCD-distribution weakness is in Track A.** `rl_train` rust has 83
   failures, 81 of them false negatives. In the baseline trees only 34/240
   `rl_train` clone questions have a verified (tier-A) correct leaf.
5. **Track A split leak on the Code-2 side.** `build_rl_split.py` separates
   Python problems only; CLCCD reuses each Code-2 program in one clone pair
   plus several non-clone pairs. Result: 240/240 `rl_heldout` rust records
   (69/360 java) have their exact Code 2 in `rl_train`. New strict split
   (both sides disjoint): rust 830 train / 110 held out / 260 dropped; 0 exact
   overlaps, 0 near-duplicates at Jaccard >= 0.95.
6. **SAFE-filter holes.** `exec_outcome_of()` maps an empty observation to
   "never tried code", so 69/168 v1 hard-set leaves that stopped at
   `import subprocess` counted as pure reasoning. Also missed: label-only
   output, answers claiming tests that never ran, clones concluded from
   inputs on which the programs diverge. Manual review also dropped leaves
   with wrong problem descriptions, tests of self-written re-implementations,
   false facts, and contradictory Code-2 descriptions (details:
   `eval_data/claude_review/leaf_review_*.jsonl`).
7. **Harness bug, not fixed.** `python_tool.rust_external_crates()` treats
   `use place` inside a comment, a local `mod math`, or `::std::io` as
   external crates and refuses to run Code 2. 5/20 CodeNet hard rust
   questions compile with plain `rustc`, including 1492, one of the three
   "unrecovered" ones. Not fixed because pending evals would stop being
   comparable. Fix it before the next eval series and re-run baselines.
8. **Search scores cannot steer eval runs.** Eval configs use
   `iterations: 2 == n_generate_sample: 2`; the unvisited-first guard spends
   both rollouts on the root's children. Step scoring matters only in mining.
   The branch runs the older assert-consistency scoring, which pays +1 for an
   empty step and is anti-predictive on clone questions (AUC 0.35-0.48).
9. **Label-ambiguous and untestable pairs.** CodeNet 1549 is a real edge-case
   divergence (Python prints 0 for n=1), so it is not taught. Track A 417 and
   1003 are Python 2, and 477 uses `fractions.gcd`, which Python 3.9 removed.

## 2. Code changes to tracked files

Exact diffs in `session_patches/2026-09-24/` (made against copies saved
before editing, so they contain only this session's changes).

| file | patch | behaviour change for existing jobs |
|---|---|---|
| `rstar-xccd/score_rl_rollouts.py` | `score_rl_rollouts.py.patch` | **Yes, by design**: strict mode is the default. `--legacy` reproduces the old counts exactly (verified: 110 rust / 58 java). No running job calls it. |
| `rstar-xccd/rstar_deepthink/agents/mcts.py` | `mcts.py.patch` | **None.** v1 scoring moved into `_score_code_step_v1()`, byte-identical (verified with diff); v2 runs only if a config sets `score_version: "v2"`. |
| `rstar-xccd/rstar_deepthink/config.py` | `config.py.patch` | **None.** Adds `score_version` (default `"v1"`) and `code2_bonus` (0.5). |

`PREEXISTING_uncommitted_not_from_this_session.patch` holds uncommitted edits
found at session start (`methodology.tex`, `nodes/mcts_node.py`, 21 lines of
`mcts.py`), so authorship stays clear.

Not modified on purpose: `evaluate_clone_results.py` (default aggregation
unchanged), `rstar_deepthink/tools/python_tool.py` (harness bug, finding 7),
`main.py`, `build_rl_split.py`, all original datasets.

## 3. New files

### Scripts (`rstar-xccd/`)

| file | purpose |
|---|---|
| `trace_checks.py` | Process flags per root-to-leaf trace (empty_code_output, claims_unrun_tests, divergent_evidence, copied_description, vacuous_clone) and tiers A/B/X. Shared by the next four scripts. |
| `process_reward.py` | Scalar trace reward: +1 verified correct, +0.5 correct but unverified, -0.5 correct but flagged, -1 wrong. |
| `harness_exec.py` | Runs code through the exact harness path (candidate_code.py staging, Code-2 compile gate, PythonInterpreter). Offline compile fallback links the AtCoder crate set. |
| `diff_test_pairs.py` | Runs Code 1 and Code 2 on the same inputs; verdicts agree_all / disagree / no_valid_inputs / code2_unrunnable. |
| `gen_inputs_proconio.py` | Random valid-format inputs from a Rust `input!{}` declaration, in 4 line layouts. |
| `build_teacher_traces.py` | Assembles teacher traces and repairs. Analyses are hand-written; every `<output>` is a real harness observation; a trace is kept only if all cases match. |
| `build_rl_split_strict.py` | Two-sided problem split (Python and Code-2 problems disjoint). |
| `check_leakage.py` | Exact and near-duplicate snippet overlap between training files and eval files. |
| `build_preference_data.py` | Step-level PPM pairs (`train/train_RM.py` format) and trajectory-level DPO pairs. |
| `rescore_runs.py` | Re-scores finished runs under old / current / tested rules, by split, with `--calibrate` and `--report_out`. |
| `replay_step_scoring.py` | Replays v1 and v2 step scoring over stored trees; AUC against subtree correctness. |
| `rstar_deepthink/agents/step_scoring.py` | score_version v2 rules (pure functions). |

### Config

`config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_rl_mining_scorev2.yaml`:
the mining config plus `score_version: "v2"`. For mining only.

### Data (`rstar-xccd/eval_data/`)

| file | contents |
|---|---|
| `rl_train_strict_python_rust_CLCCD.jsonl` / `rl_heldout_strict_python_rust_CLCCD.jsonl` | strict split, 830 / 110 (60 clone, 50 non-clone), seed 122 |
| `sft_trackA_rust_v2_strict.jsonl` | **preferred**: 130 examples (65/65), 47 teacher, `rl_train_strict` only. Eval only on `rl_heldout_strict`. |
| `sft_trackA_rust_v2.jsonl` | 154 examples (77/77), 57 teacher, `rl_train` only. Eval only on `rl_heldout` (leak caveat, finding 5). |
| `sft_hard_codenet_{rust_only,java_only,java_rust}_v2.jsonl` | 22 / 23 / 45 examples, strict + reviewed + 4 repairs + 2 teacher. Full-CLCCD-evaluable. |
| `claude_review/diff_test_*.jsonl` | differential-test results per question |
| `claude_review/manual_inputs_rust_trackA_hard.json`, `gen_inputs_*.json` | official AtCoder sample inputs (hand-entered) and generated inputs |
| `claude_review/untestable_rust_trackA_hard.json` | 417, 1003 (Python 2), 477 (fractions.gcd) |
| `claude_review/leaf_review_{codenet_rust,codenet_java,rust_trackA}_hard.jsonl` | verdict (keep / drop / repair) + reason for every surviving leaf |
| `claude_review/teacher_analyses_*.json` | hand-written analyses + official sample inputs per question |
| `claude_review/teacher_traces_rust_trackA.jsonl` | 57 teacher traces (37 run_both, 20 python_only) |
| `claude_review/teacher_and_repairs_codenet_rust_hard.jsonl` | 2 teacher traces (1113, 1492), 4 repairs (225 x2, 673, 1383) |
| `claude_review/ppm_pairs_*.json`, `dpo_pairs_*.jsonl` | preference data (small; see section 7) |
| `claude_review/atcoder_crates/` | cargo-built AtCoder-2020 crate set (proconio 0.3.6 etc.), offline testing only |

### Launchers (`rstar-xccd/Qwen3/`)

| launcher | status |
|---|---|
| `run_eval_clccd_{rust_hard_codenet_rust_only,java_hard_codenet_java_rust,rust_hard_codenet_java_rust}_v1_qwen3_4b_2gpu{,_fard}.sh` | submitted (part G) |
| `run_sft_trackA_rust_v2_strict_qwen3_4b_4gpu.sh` + `run_eval_heldout_rust_trackA_rust_v2_strict_qwen3_4b_2gpu{,_fard}.sh` | submitted (section 6) |
| `run_eval_clccd_{rust,java}_base_newrule_qwen3_4b_2gpu{,_fard}.sh` | submitted (section 6) |
| `run_sft_{trackA_rust_v2,hard_codenet_{rust_only,java_only,java_rust}_v2}_qwen3_4b_4gpu.sh` | written, not submitted |
| `run_eval_heldout_rust_trackA_rust_v2_qwen3_4b_2gpu{,_fard}.sh`, `run_eval_clccd_*_hard_codenet_*_v2_qwen3_4b_2gpu{,_fard}.sh` | written, not submitted |
| `run_rstar_mcts_slurm_rl_mining_trackA_strict_rust_qwen3_4b{,_fard}.sh` | written, not submitted (sampled mining for PPM data) |

Launcher conventions: eval launchers written this session have no
`--no-resume` (resubmit to resume after a timeout); full-CLCCD fine-tuned
evals get 72h, since fp32 checkpoints run at about 25 rust / 35 java
questions per hour.

## 4. Incidents

- **SFT launchers without an environment (fixed).** The five v2 SFT
  launchers were generated without the `module load` + `source venv-qwen3`
  block. Job 3943747 failed in 10 s (`No module named 'omegaconf'`), and
  slurm cancelled its dependent evals 3943748/3943749. The block was
  restored (identical to the v1 launcher, verified with diff) and the run
  resubmitted as 3945991 -> 3945992/3945993. The eval, mining and new-rule
  launchers were not affected.
- Status file part F said the 0.7692 rust `_result` was written about 5 min
  after launch. It was written 09-23 22:30 over only 542 instances (a
  mid-run snapshot). Corrected in part G.

## 5. How to reproduce each number

Run from `rstar-xccd/` with `source ../venv-qwen3/bin/activate` and
`export PYTHONPATH=$PWD`.

```bash
B_RUST=eval_data/test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.assert-consistency-score.20260904210649.jsonl
B_JAVA=eval_data/test_python_java_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.assert-consistency-score.20260904205237.jsonl

# findings 1-2: every rule, by split, with calibration
python rescore_runs.py --lang rust --calibrate $B_RUST \
  eval_data/test_python_rust_CLCCD_depth_16.jsonl.mcts.qwen3-4b-codenet-all-v1.assert-consistency-score.def-khajezad_gpu.20260922231141.jsonl
python rescore_runs.py --lang java --calibrate $B_JAVA
# independent replication runs
python rescore_runs.py --lang rust --calibrate eval_data/test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.assert-consistency-score.20260905233224.jsonl

# finding 8: step-scoring replay
python replay_step_scoring.py $B_RUST --only_indices_from eval_data/rl_train_python_rust_CLCCD.jsonl

# finding 5: strict split + leakage
python build_rl_split_strict.py --lang rust
python check_leakage.py --train eval_data/rl_train_python_rust_CLCCD.jsonl --eval eval_data/rl_heldout_python_rust_CLCCD.jsonl
python check_leakage.py --train eval_data/rl_train_strict_python_rust_CLCCD.jsonl --eval eval_data/rl_heldout_strict_python_rust_CLCCD.jsonl --jaccard 0.95

# finding 6: legacy vs strict curation
R="eval_data/failed_codenet_python_rust_depth_16.jsonl.mcts.Qwen3-4B.rl-mining-hard-codenet*.jsonl"
python score_rl_rollouts.py $R --legacy
python score_rl_rollouts.py $R --diff_test eval_data/claude_review/diff_test_rust_hard.jsonl \
  --diff_test eval_data/claude_review/diff_test_rust_hard_crates.jsonl

# teacher traces (re-runs every observation through the harness)
python build_teacher_traces.py --questions eval_data/failed_qwen3_4b_rl_train_rust.jsonl \
  --diff_test eval_data/claude_review/diff_test_rust_trackA_hard.jsonl \
  --analyses eval_data/claude_review/teacher_analyses_rust_trackA.json --out /tmp/check.jsonl
```

Finding 3 (CodeNet recall 0.983) comes from `predict_label()` over
`eval_data/codenet_train_python_rust_depth_16.jsonl.mcts.Qwen3-4B.rl-mining-codenet-v1.def-fard_gpu.20260921203724.jsonl`
restricted to the indices in `codenet_train_python_rust_clean.jsonl`.

## 6. Jobs

| job | what | compare against |
|---|---|---|
| 3757227 | full CLCCD rust, codenet-rust-only-v1 (near its 48h limit) | current rule: base 0.7952 |
| 3921168 | full CLCCD java, hard-codenet-java-only-v1 (will likely time out; resume) | current rule: base 0.9163 |
| 3940133-3940139 | full CLCCD evals of hard-codenet rust-only-v1 and java-rust-v1 (fard copies running) | current rule |
| 3945991 -> 3945992 / 3945993 | SFT trackA-rust-v2-strict -> eval on `rl_heldout_strict` | base on `rl_heldout_strict`: 0.8000 (current), 0.8785 (calibrated) |
| 3945722 / 3945737 | **new-rule confirmation**, base, full CLCCD rust | predicted `rl_heldout` about 0.80 -> 0.89 |
| 3945738 / 3945744 | **new-rule confirmation**, base, full CLCCD java | predicted `rl_heldout` about 0.91 -> 0.96 |

The new-rule runs write `...Qwen3-4B.newrule-tested-agg.<account>.<ts>.jsonl`,
its `_result` (current rule) and `_result_NEWRULE_tested_agg` (all rules by
split). Always compare checkpoints with `rescore_runs.py` rather than
against `FINAL_FOR_CLCCD.md`.

## 7. Open items, next steps

1. When the new-rule runs finish, check that `_result_NEWRULE_tested_agg`
   shows the held-out gain (expect about +/-0.01 run-to-run variation; the
   strict set has only 110 questions).
2. Decide how to report baselines: restate them under the current rule, or
   recompute all checkpoints under the old rule. Update `FINAL_FOR_CLCCD.md`
   and `methodology.tex` accordingly.
3. When 3945993 or 3945992 finishes, compare the Track A strict SFT result
   with the base model on `rl_heldout_strict`, under the current rule. Do not
   calibrate this model on `rl_train`, since it was trained there.
4. Fix `rust_external_crates()` (strip comments, subtract local `mod` names,
   ignore `::std`) before the next eval series, then re-run baselines with it.
5. Sampled mining over `rl_train_strict` (launcher ready) to get step-level
   PPM pairs at useful scale. There are only 11-45 pairs per set now.
6. Java Track A teacher traces were not written (the focus was rust).
7. The codenet `_v2` SFT runs are ready, but CodeNet is the wrong
   distribution for rust recall (finding 3); lower priority.
8. Teacher traces are Claude-written. Report them as distillation, separate
   from self-improvement, and check Anthropic's usage terms for training on
   model outputs.

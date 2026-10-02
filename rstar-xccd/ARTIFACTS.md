# Artifact manifest: paper numbers -> files -> commands

Paper: `rStar-XCCD-Paper/main.tex`. All paths below are relative to `rStar-XCCD/rstar-xccd/`. Created 2026-10-01; keep it current.

- **Environment.** Run every scoring command from this directory with `source ../venv-qwen3/bin/activate`.
- **CPU only.** All scoring is CPU-only.
- **Run outputs.** Search trees (`*.mcts.*.jsonl`) and literature-prompt outputs (`eval_data/paper_prompt_replication/*.jsonl`) are large and git-ignored. They live on Narval `/project` and are not in git. The pilot and locked evaluation sets, the scripts and the score summaries are committed.

## Data

| Set | File(s) | Built by |
|---|---|---|
| CLCCD test (Java 1,800, Rust 1,200) | `eval_data/test_python_{java,rust}_CLCCD.jsonl` | CLCCD (paper §3) |
| Dev set for rule and prompt choice (300 pairs per language) | `eval_data/e16a/dev_python_{L}_codenet.jsonl` (+`_meta`) | `eval_data/e16a/build_codenet_disjoint.py` |
| CLCCD test problem ids (for exclusion and clustering) | `eval_data/e16a/test_pids_{L}.json` | `eval_data/e16a/recover_test_pids.py` |
| Hard-negative **dev** pilot (250 pairs per language) | `eval_data/e16a/hard_python_{L}_codenet.jsonl` (+`_meta`) | `build_hard_negatives.py`, then `audit_hard_negatives.py` |
| Hard-negative **LOCKED** sets (Java 375, Rust 90) | `eval_data/e16a/hardeval_python_{L}_codenet.jsonl` (+`_meta`); checksums in `hardeval_{L}_MANIFEST.txt` | `build_training_pool.py` (Rust: `--max_len 6000`), then `split_training_pool.py` |
| Cost subsets (100 pairs per language, fixed random sample) | `eval_data/e16a/cost100_python_{L}_CLCCD.jsonl` | inline, seed `cost-20261001-{L}` (see the commit message of cb43220) |

## Tables and claims

| Paper item | Result file(s) | Command |
|---|---|---|
| SCB on CLCCD (RQ1/RQ2 rows; F1 0.9691 / 0.9165) | `eval_data/test_python_{L}_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.ablation-pyonly.*.jsonl` | `python rescore_runs.py <file> --lang L` (tested row) |
| SCB seeds 1, 2 | `...ablation-pyonly-seed{1,2}.*.jsonl` | same command. **Java seed 1 was resumed after the SystemExit fix (job 4441689)** |
| `tab:main-lit-vs-scb`, `tab:litprompt-dev`, `tab:results-java`, `tab:results-rust` | `eval_data/paper_prompt_replication/{dev,test}_python_{L}_*.jsonl` | `python eval_data/e16a/summarize_baselines.py` → `eval_data/e16a/baselines_summary_2026-09-30.txt` |
| Self-consistency rows | `...sp2.nothink.s1.n16.*.jsonl` | `python eval_data/e16a/score_self_consistency.py --lang L --dev <dev n16> --test <test n16>` |
| Pair-level bootstrap CIs (RQ1/RQ2 text) | as above | `summarize_baselines.py` (10,000 resamples) |
| Problem-level bootstrap CIs (robustness check) | as above | `python eval_data/e16a/bootstrap_by_problem.py --boot 5000` → `eval_data/e16a/bootstrap_by_problem_2026-10-01.txt` |
| `tab:cost` | `Qwen3/cost-*_44107*.log.out` + `cost100_*` outputs | `python eval_data/e16a/cost_summary.py` → `eval_data/e16a/cost_summary_2026-10-01.txt` |
| `tab:hard-negatives` (RQ3) and extension CIs (RQ4) | `eval_data/e16a/hardeval_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.hard-locked-{pyonly,ext}.*.jsonl`; `paper_prompt_replication/hardeval_*` | `python eval_data/e16a/hardeval_metrics.py --bootstrap 5000` → `eval_data/e16a/hardeval_metrics_2026-10-01.txt` |
| RQ3 tree diagnosis (1% mismatch, 2% sample input, 29% hollow, 38% crashed, 38% vs 16% read-and-reject) | `...hardeval...hard-locked-pyonly.*.jsonl` | `python eval_data/e16a/diagnose_hard_negatives.py --lang L --set hardeval --arm locked-pyonly` |
| RQ4 Code 2 call rates on dev (Java 10%, Rust 77%) | `eval_data/e16a/hard_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.hard-{dualexec,dualexec-fix}.*.jsonl` | `diagnose_hard_negatives.py --lang java --arm dualexec`; `--lang rust --arm dualexec-fix` |
| `tab:code2-execution-rate` | the "tried to run Code 2" line at the end of each ablation job log (`Qwen3/q3-4b-ablation-*_4018*.log.out`, `Qwen3/q3-4b-e5-dualexecfix-rust_4344209.log.out`) | printed by the launchers |
| `tab:ablation-dual` | `...ablation-pyonly.*`, `...ablation-dualexec.*` (Java), `...ablation-dualexec-fix.*` (Rust) | `python eval_data/e16a/rules_table.py <files>` |
| `tab:aggregation-effect` | dev: `eval_data/e16a/dev_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.e1-dev-pyonly.*.jsonl`; test: `...ablation-pyonly.*` | `rules_table.py <files>` |
| `tab:ablation-ladder` | search rows: as `tab:aggregation-effect`. Single-pass rows: `pure_inference/offline_results/test_different_python_{L}.jsonl_qwen3-4b_*` (our prompt) and the literature files above | `rules_table.py`; `summarize_baselines.py` |
| `tab:tested-mechanism` | `...ablation-pyonly.*` | `python eval_data/e16a/audit_tested_rule.py` |
| Pilot results (dev hard set, all arms) | `eval_data/e16a/hard_python_{L}_codenet_depth_16.jsonl.mcts.*` | `python eval_data/e16a/score_hard_negatives.py --lang L` → `hard_negatives_scores_2026-09-30.txt` |
| Extension on CLCCD (RQ4, pending) | `...clccd-ext.*.jsonl` | `rules_table.py <file>` (clone-on-disagreement row) |

## Configurations (frozen)

- **SCB:** `config/qwen3_4b_ablation_pyonly_1gpu.yaml`. Seed configs: `..._seed{1,2}_1gpu.yaml`.
- **Dual execution (RQ4):** `config/qwen3_4b_ablation_dualexec_{java,rust}_1gpu.yaml`.
- **Extension:**
  - Java: `config/qwen3_4b_hard_autocode2_javav3_1gpu.yaml`.
  - Rust: `config/qwen3_4b_hard_autocode2_rustv4_1gpu.yaml`.
  - Both run the harness with `auto_code2`, the INCONCLUSIVE label, and the code-block fix.
- **Harness faults fixed during the study:**
  - Code-block regex (`agents/tree.py`), 2026-09-30.
  - SystemExit (`tools/python_tool.py`), 2026-10-01.
  - See the commit log of branch `claude-session-2026-09-24`.

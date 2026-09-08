# `code2-exec` smoke test — keeping four concurrent runs from colliding

The Code-2-execution smoke test submits **four jobs at once**: two models
(Qwen2.5-Coder-7B-Instruct, Qwen3-8B) × two allocations
(`def-khajezad_gpu`, `def-fard_gpu`). All four read the same 16-question
file, the same configs and the same checkout, and all four write into the
same `eval_data/`. Doubling up on the two accounts is deliberate — whichever
allocation drains its queue first gives the answer — so the runs are
genuinely concurrent, not a fallback chain.

Three things would otherwise collide. Each is handled by a different layer,
and only the first one is about the commit tag.

| collision | what breaks | prevented by |
|---|---|---|
| Output file / resume | Two runs append to one `.jsonl`; a resume glob adopts a run from *different code* | **`--branch` commit tag** folded into `run_tag` |
| Node-local scratch | One step deletes `Main.class` while another job is running it | Per-job **symlink-farm rundir** on `$SLURM_TMPDIR` |
| Interpreter | Relative `../venv` misses, job silently falls back to `~/.local` torch | **`$REPO`-anchored `VENV`** + hard `FATAL` guard |

## 1. The commit tag (`--branch`)

`--branch` is not a git operation — it is a free-form tag naming *the code
version a run came from*, which by convention is the branch (here
`code2-execution`, narrowed to the smoke test's purpose). `main.py` folds it
into the run identity:

```python
qaf_tag   = f"{qaf_stem}_depth_{config.max_depth}{qaf_ext}"
branch_tag = f".{args.branch}" if args.branch else ""
run_tag   = f"{qaf_tag}.{config.mode}.{llm_version}{branch_tag}"
saved_jsonl_file = f"{run_tag}.{datetime.now():%Y%m%d%H%M%S}.jsonl"
```

So the four jobs land on four distinct paths:

```
eval_data/smoke_test_code2_exec_depth_16.jsonl.mcts.Qwen2.5-Coder-7B-Instruct.code2-exec-smoke-test.<ts>.jsonl
eval_data/smoke_test_code2_exec_depth_16.jsonl.mcts.Qwen2.5-Coder-7B-Instruct.code2-exec-smoke-test-fard.<ts>.jsonl
eval_data/smoke_test_code2_exec_depth_16.jsonl.mcts.Qwen3-8B.code2-exec-smoke-test.<ts>.jsonl
eval_data/smoke_test_code2_exec_depth_16.jsonl.mcts.Qwen3-8B.code2-exec-smoke-test-fard.<ts>.jsonl
```

Question file, depth and model already separate *some* runs, which is why
the two models don't need the tag to stay apart. The tag does two things
those fields cannot:

- **Separates same-config runs.** The two accounts run byte-identical
  configs against the same model. Only the `-fard` suffix distinguishes
  them, so they never share a writer and can be compared against each other
  afterwards rather than being silently interleaved into one file.
- **Quarantines the code version.** `run_tag` is also the resume glob —
  `find_latest_output_file(f"{run_tag}.*.jsonl")`. Without the tag a resume
  would happily adopt the newest matching file from *before* the Code-2
  change and skip every question it recorded, producing a "passing" smoke
  test assembled from output the new harness never generated. The tag makes
  the resume search code-version-scoped: pre-`code2-execution` runs carry a
  different tag (or none) and are invisible to it.

The smoke test additionally passes `--no-resume`, so it always starts a
fresh file. That is belt-and-braces: `--no-resume` protects *this* run, the
tag protects every later run that might glob past it.

**Convention when adding a variant:** the tag names the code change under
test, plus the account if a second copy runs concurrently
(`code2-exec-smoke-test`, `code2-exec-smoke-test-fard`). Change the code
under test → change the tag. Reusing a tag across a behavioural change is
the one way to reintroduce the resume hazard.

## 2. Per-job rundir (scratch isolation)

Each step writes scratch into the *working directory* — `candidate_code.py`,
`<Class>.java` and the `.class` files `javac` emits, `candidate_code2.rs`
and its compiled binary, plus whatever the model names its own test files —
and `cleanup_generated_code()` (`rstar_deepthink/tools/python_tool.py:238`)
deletes them at the end of each step. Two jobs sharing a working directory
race: one deletes a file the other is about to execute.

So the scripts don't run from the repo. They build a symlink farm over it on
node-local scratch and `cd` there:

```bash
RUNDIR="${SLURM_TMPDIR:-/tmp}/rundir_${SLURM_JOB_ID:-$$}"
mkdir -p "$RUNDIR"
for entry in "$REPO"/* "$REPO"/.[!.]*; do
  [ -e "$entry" ] && ln -sfn "$entry" "$RUNDIR/$(basename "$entry")"
done
cd "$RUNDIR"
```

Reads and the result `.jsonl` still reach the real repo through the
symlinks, so output lands in `eval_data/` exactly as before; only the
scratch stays node-local, keyed by `$SLURM_JOB_ID`. This also keeps the repo
root clean.

## 3. Interpreter anchoring (the regression this cost us)

The rundir has a consequence that already broke a submission. Jobs
2642659–2642662 (2026-09-07 22:20) each died in seconds:

```
slurm_script: line 47: ../venv/bin/activate: No such file or directory
...
OSError: libhwloc.so.5: cannot open shared object file: No such file or directory
```

`../venv` is relative to CWD, and CWD is now `$SLURM_TMPDIR/rundir_<jobid>`,
so it resolved to `/localscratch/khajezad.<jobid>.0/venv` and missed. Worse,
`source` failing is not fatal in bash: the job continued **without a venv**,
`python main.py` picked up `~/.local` torch, and died on a missing
`libhwloc.so.5`. The traceback pointed at torch, not at the real cause.

Both fixes are now in every smoke script:

```bash
VENV="$REPO/../venv/bin/activate"        # venv-qwen3 for the Qwen3 scripts
if [ ! -f "$VENV" ]; then
  echo "FATAL: venv not found at $VENV" >&2
  exit 1
fi
source "$VENV"
```

Anchoring to `$REPO` survives the `cd`, and the guard turns a missing venv
into an immediate one-line failure instead of a misleading torch traceback
twenty lines down. **Any path in these scripts must be absolute or
`$REPO`-anchored** — `--qaf`, `--custom_cfg` and `--model_dir` stay relative
only because they resolve through the symlink farm, which mirrors the repo
root exactly.

Modules load *before* the venv activates, and `LD_PRELOAD` pins a newer NCCL
than the venv's PyTorch links against. `java/17.0.6` is on the module line
specifically for this test: `javac`/`java` must be on `PATH` or the model's
Code 2 (Java) subprocess tests fail with `FileNotFoundError` inside
`stage_code2()` instead of actually compiling. Rust needs no module —
`rustc`/`cargo` come from `~/.cargo/bin`.

## The four jobs

Depth 16, `n_generate_sample: 2`, `is_sampling: False`, `max_model_len:
16384`, 1×A100 / 4 CPU / 64G / 1h each.

| script | account | model | `--branch` tag |
|---|---|---|---|
| `Qwen2.5/…qwen2.5_7b_depth_16_1gpu.sh` | `def-khajezad_gpu` | Qwen2.5-Coder-7B-Instruct | `code2-exec-smoke-test` |
| `Qwen2.5/…qwen2.5_7b_depth_16_1gpu_fard.sh` | `def-fard_gpu` | Qwen2.5-Coder-7B-Instruct | `code2-exec-smoke-test-fard` |
| `Qwen3/…qwen3_8b_depth_16_1gpu.sh` | `def-khajezad_gpu` | Qwen3-8B | `code2-exec-smoke-test` |
| `Qwen3/…qwen3_8b_depth_16_1gpu_fard.sh` | `def-fard_gpu` | Qwen3-8B | `code2-exec-smoke-test-fard` |

All four are `run_rstar_mcts_slurm_smoke_test_code2_exec_*` under
`rstar-xccd/`. What they exercise is `stage_code2()` /
`mentions_code2_execution()` in `rstar_deepthink/tools/python_tool.py` and
the paired `mcts_prompt_code2_exec.json` — the only thing that differs from
the `no_sampling` configs. Sixteen questions
(`eval_data/smoke_test_code2_exec.jsonl`) is deliberately small: the point
is to confirm the model actually attempts and succeeds at compiling and
running Code 2 end-to-end before committing to a full 1800/1200-question,
multi-hour rerun.

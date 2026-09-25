# Search Scoring on the `search-score-update` / `assert-consistency-score` Branches

## Why this exists

The stock MCTS agent (`rstar_deepthink/agents/mcts.py`) only ever scores a
node when it becomes **terminal**, via `eval_final_answer()`. With
`is_sampling: True`, that score comes from comparing the terminal node's
`final_answer` against `self.ground_truth` — i.e. the search is told the
correct clone/non-clone label while it's still running, and PUCT selection
actively steers toward branches that agree with it. With `is_sampling: False`
(what every current CLCCD MCTS run uses, to get an honest, oracle-free
measurement of the model), that comparison is skipped entirely — which
means, in the stock agent, **no non-terminal node ever gets scored at all**.
Every intermediate branch looks equally good to PUCT, whether the model's
code crashed, ran cleanly, or never attempted anything.

These two branches add scoring for intermediate (non-terminal)
`python_interpreter` steps, without ever consulting `self.ground_truth`, so
the search still gets a real quality signal under `is_sampling: False`.

## Baseline: crash vs. clean execution (`search-score-update`)

`create_child()` classifies every `python_interpreter` step's observation
string (see `code_execution()` in `tree.py`) as either:

- **Crashed** — the tool raised an exception. Increments
  `consecutive_errors` (can terminate the branch early via
  `errors_threshold`) and, if the branch survives, is scored
  `negative_reward` immediately.
- **Clean** — no exception. Scored `positive_reward` immediately.

This is a blind binary check (`"error" not in observation.lower()`) and
makes no attempt to reason about *what* the code actually verified.

## New: self-consistency scoring for assertions (`assert-consistency-score`)

The blind crash/clean check has one important gap: the few-shot prompts
teach the model to verify clone/non-clone claims by writing
`assert <output1> == <output2>`-style checks and running them. A **failed**
assertion (`AssertionError`) is not a code defect — it's often the
*correct, informative* result of a real verification attempt (the two
snippets disagree on some input, i.e. they're not clones). Scoring it the
same as a genuine crash punishes the model for doing exactly the kind of
verification the prompts ask for.

Instead of scoring these steps immediately, `create_child()` now **defers**
them and tags the node with the verdict its execution implies:

| Execution outcome | Contains an `assert`? | Immediate score | Deferred tag (`state["assert_verdict"]`) |
|---|---|---|---|
| Crashed, not `AssertionError` | — | `negative_reward` (unchanged) | — |
| Crashed, `AssertionError` | (implicitly yes) | *none* | `"non-clone"` (assertion failed → snippets differ) |
| Clean | no | `positive_reward` (unchanged) | — |
| Clean | yes | *none* | `"clone"` (assertion held → snippets agree) |

"Contains an `assert`" is checked against the actual code that ran —
reconstructed the same way `tree.py`'s `_code_execution()` does, via
`collect_action_inputs()` + `extract_program()` — using the regex
`\bassert\b`, not just the current step's raw text, so it correctly finds
an assertion written in an earlier step and re-executed as part of the
accumulated program.

An `AssertionError` no longer increments `consecutive_errors` either: since
it's not a defect, it shouldn't count toward `errors_threshold`-triggered
early termination.

### Retroactive resolution

A tagged node's eventual "was this a good branch" answer isn't known until
some descendant leaf produces a real final answer — which might be several
steps later, and a single tagged ancestor can have multiple descendant
leaves across different rollouts. `eval_final_answer()` now calls
`_score_pending_assert_verdicts(node)` for every leaf that resolves to a
genuine (non-placeholder) `final_answer`:

1. Normalize the leaf's `final_answer` via `evaluate_clone_results.normalize_label()`
   (the same normalization the offline evaluator uses, so "agreement" here
   means the same thing it means in the final `_result` report).
2. Walk up the ancestor chain. For every ancestor carrying an
   `assert_verdict` tag, compare it to the leaf's normalized label:
   - **Match** → `positive_reward` (the branch stayed faithful to its own
     evidence).
   - **Mismatch** → `negative_reward` (e.g. the assertion found the
     snippets differ, but the branch still concluded "clone" via some other
     rationalization — a real failure mode observed in earlier runs).
3. Apply the reward via the tagged node's own `update_recursive()`, so it
   (and everything back to the root) gets the usual MCTS running-average
   treatment — repeated visits from different leaves just average in,
   exactly like every other reward in this tree.

This comparison never touches `self.ground_truth` — it only checks whether
the branch's *own* conclusion is consistent with the *evidence its own
code produced*. That keeps it meaningful under `is_sampling: False` (no
label leakage) while still rewarding/penalizing something real: internal
coherence between verification and conclusion.

## Explicitly out of scope (unchanged from `search-score-update`)

- The "No valid Python code found in the response." and "Java execution is
  not supported in this sandbox..." guardrail messages are still classified
  under the plain crash/clean binary check, unchanged. No special-casing
  was added for them in this branch.
- This scoring path runs unconditionally (regardless of `is_sampling`), matching
  the existing convention that code-execution-based scoring in
  `create_child()` is independent of `is_sampling`/ground truth.

## Files touched

- `rstar_deepthink/agents/mcts.py` — all of the above.

## Verification performed

- `python -m py_compile rstar_deepthink/agents/mcts.py`
- Full module import under the project venv (`from rstar_deepthink.agents.mcts import MCTS`)
- Regex sanity checks: `ASSERT_RE` matches `assert x == y`, does not match
  `print(1)` or identifiers like `reassertion_count`.

No MCTS run has been executed against this branch yet — the above is
static verification only.

---

# Rework on the `exec-outcome-scoring` branch

## Why this exists

A node-level audit of the four completed `assert-consistency-score` CLCCD
runs (Qwen3-4B and Qwen2.5-Coder-3B, python↔java and python↔rust; ~55k
nodes) found the scoring above was rewarding the wrong things, and that its
central inference is unsound in this sandbox.

**The assertions cannot test what the score assumes they test.** `tree.py`
writes only Code 1 (the Python side) to `candidate_code.py`, and
`is_java_execution_attempt()` blocks running Code 2. Code 2 is therefore
never executable, so the model's assertions overwhelmingly compare Code 1's
output against an expected literal *the model invented* — 73% of tagged
nodes on Qwen3/java, 91% on Qwen3/rust. An `AssertionError` means "the model
guessed the output wrong", not "the snippets differ".

Measured consequences:

| finding | evidence |
|---|---|
| the `non-clone` verdict was never right | 0/38 correct (Qwen3/java), 0/27 (Qwen3/rust), 0/45 (Qwen2.5/java) |
| verdicts assigned when no code ran at all | 65/169 (38%, Qwen3/java), 72/130 (55%, Qwen3/rust) |
| largest reward class was content-free | the model printing its own verdict ("not code clones") was 52%–79% of all code executions, each paid `positive_reward` |
| failure to act was paid, not penalized | "No valid Python code found" earned `positive_reward` 170–183×/run, on instances 170:0 and 177:6 ground-truth *clone* |
| the crash test misfired both ways | `"error" not in observation.lower()`: Qwen2.5/java had 415 observations containing "error" vs 267 real exceptions |

Every misleading reward concentrated on ground-truth-clone instances, which
is the population these runs lose (recall 0.64–0.82 against precision ~1.0).

## The five changes

1. **Dropped the `AssertionError → "non-clone"` inference.** A failed
   assertion is now scored neutrally: not a defect, but no evidence about
   equivalence either. It still does not count toward `errors_threshold`.
2. **Verdict tagging requires that code actually ran** (`EXEC_OK`). This
   falls out of the new outcome classification rather than being a separate
   guard.
3. **`EXEC_NO_CODE` and `EXEC_BLOCKED` now earn `negative_reward`**
   instead of `positive_reward`. Neither increments `consecutive_errors`:
   penalising them changes the rewards without changing the search's shape.
4. **Verdict echoes earn nothing.** `is_verdict_echo()` in `mcts.py` detects
   a short observation that normalizes to a clone/non-clone label — the
   model printing its conclusion rather than computing anything.
5. **Real error flag replaces the substring test.**
   `PythonInterpreter.run()` returns `(nothing_was_raised, text)`, and
   `code_execution()` returns `(observation, outcome)` where outcome is one
   of `EXEC_OK` / `EXEC_ERROR` / `EXEC_ASSERTION_FAILED` / `EXEC_NO_CODE` /
   `EXEC_BLOCKED`. The outcome is persisted to `state["exec_outcome"]` so
   later rollout audits can read what the scorer saw instead of re-deriving
   it from text.

`_score_pending_assert_verdicts()` is unchanged; with change 1 the only tag
it can now see is `"clone"`.

## Expected payoff, honestly

Small. These runs produce 1.2–1.5 real answers per tree and 95%–99% of
trees are label-homogeneous, so even an oracle picking the best existing
leaf gains only +0.017 to +0.032 F1. The case for these changes is that the
previous scoring was invalid and actively penalised correct branches on
clone instances — not that large gains are waiting.

## Sampling

`is_sampling` already defaults to `False` in `rstar_deepthink/config.py`,
and every config used by the runs below sets it explicitly. The search
therefore never compares against `self.ground_truth`.

## Files touched

- `rstar_deepthink/tools/python_tool.py` — `(ok, text)` return.
- `rstar_deepthink/agents/tree.py` — `EXEC_*` outcomes, `code_execution()`.
- `rstar_deepthink/agents/mcts.py` — changes 1–4.
- `rstar_deepthink/agents/beam_search.py` — uses the real error flag too.

## Verification performed

- `py_compile` on all four files; full import under both venvs
  (`venv-qwen3` for Qwen3, `venv` for Qwen2.5).
- End-to-end `code_execution()` classification for all five outcomes,
  including the two regressions change 5 fixes: a program printing "error"
  as data now classifies `ok`, and a failing assert classifies
  `assertion_failed` rather than a crash.
- `is_verdict_echo()` accepts "not code clones"/"clone" and rejects "2",
  "All Python test cases passed.", and long output.

No MCTS run had been executed against this branch at the time of writing.

---

# Scoring review, 2026-09-24 (branch `rl-self-improvement`)

## What is actually running

This branch's `mcts.py` still carries the **assert-consistency-score** rules
(section 2 above), not the exec-outcome-scoring rework (section 3 is on its own
branch and was never merged here): `"error" not in observation.lower()`, a
failed assert tags the step `non-clone`, a passing self-written assert tags it
`clone`, and a step that runs cleanly with no output (e.g. only
`import subprocess`) earns `positive_reward`.

## Two findings that matter more than the step rules

1. **Search scores cannot steer an eval run.** Eval configs use
   `iterations: 2 == n_generate_sample: 2`; the unvisited-first guard in
   `select_child()` spends both rollouts on the two root children. Step
   scoring only matters in mining configs (iterations 3-12).
2. **The published baselines and every SFT checkpoint number use different
   aggregation rules.** FINAL_FOR_CLCCD.md (rust 0.8739, java 0.9454) was scored
   before 2026-09-12 with the exec-signature override applied unconditionally;
   all later `_result` files apply it only to trees without a leaf vote. Same
   baseline trees under the current rule: **rust 0.7952, java 0.9163**.
   `rescore_runs.py` re-scores any run under all rules:

   | full CLCCD F1 | old rule | current rule |
   |---|---|---|
   | base rust | 0.8739 | 0.7952 |
   | codenet-all-v1 rust | 0.6354 | 0.8498 |
   | base java | 0.9443 | 0.9163 |
   | codenet-all-v1 java | 0.8594 | 0.9425 |

   Under ONE rule, CodeNet SFT improved on the base model (+0.055 rust,
   +0.026 java) -- the "flat to worse" conclusion compared across rules.

## Aggregation: "tested non-clone" signal

The prompt tells the model to skip testing and `print("not code clones")` when
it judges a pair non-clone; it writes tests only when it leans clone. On
base-model rl_train rust trees, a non-clone leaf whose path ran any real test
is ground-truth CLONE 91-96% of the time (held-out: 29/29); print-shortcut
non-clone leaves are right 93-94%. Counting tested non-clone leaves as clone
votes (`rescore_runs.py`, rule `tested`) chosen on rl_train, reported on
held-out: base rust rl_heldout 0.8039 -> **0.8909**, rl_heldout_strict
0.8000 -> **0.8785**, java rl_heldout 0.9080 -> **0.9634**.
It is MODEL-DEPENDENT: fine-tuned checkpoints test non-clones too, and the
rule lowers their F1 (codenet-all-v1 rust 0.8498 -> 0.7960). Always use
`rescore_runs.py --calibrate` (rule chosen per run on its rl_train subset,
reported on rl_heldout). Calibrated held-out: base still ahead
(rust rl_heldout 0.891 vs codenet-all-v1 0.844; java 0.963 vs 0.952).
The default of `evaluate_clone_results.predict_label` is NOT changed.

## score_version v2 (opt-in; default v1 unchanged, byte-identical block)

`rstar_deepthink/agents/step_scoring.py`, wired via `config.score_version`
(+ `code2_bonus`); example config
`config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_rl_mining_scorev2.yaml`.
Rules: outcome from exception pattern, not the "error" substring; empty /
label-only output and verdict echo -> 0 (v1 paid +1); a failed or passing
assert is evidence ONLY if the program actually ran Code 2 (otherwise the
model's guessed literal decides it); running Code 2 successfully earns
+code2_bonus; a clone leaf claiming tests with no output anywhere gets
negative_reward. None of it reads the ground truth.

Offline replay (`replay_step_scoring.py`, AUC of a step's score for "its
subtree is mostly right"; GT used only to evaluate):

| trees | v1 all | v2 all | v1 clone-q | v2 clone-q |
|---|---|---|---|---|
| base rust rl_train (eval) | 0.546 | 0.484 | 0.352 | 0.446 |
| Track A rust hard mining | 0.451 | 0.582 | 0.432 | 0.614 |
| CodeNet rust mining | 0.481 | 0.544 | 0.375 | 0.566 |
| base java rl_train (eval) | 0.519 | 0.500 | 0.481 | 0.580 |

v1 is anti-predictive on clone questions (it penalises the no-code /
failed-assert branches that are mostly clones -- the exec-signature effect).
v2 fixes the direction on clone questions and on mining trees, but every
step-level score stays near chance: step rewards are a weak lever here.
Making crash/no-code neutral as well ("v2-nopenalty") was not consistently
better and is not implemented. Use v2 for mining; do not expect eval gains.

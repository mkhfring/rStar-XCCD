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

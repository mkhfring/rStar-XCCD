# Label-Free Search Signals: What Was Tried and Why It Was Rejected

Working notes on scoring signals evaluated for the MCTS search but **not**
adopted. Kept because the negative results are more informative than the
positive ones and are intended as material for the paper's discussion.

Everything here was measured offline against the four completed
`assert-consistency-score` CLCCD runs — Qwen3-4B and Qwen2.5-Coder-3B,
python↔java (1800 instances) and python↔rust (1200), depth 16,
`n_generate_sample=2`, `is_sampling: False` — roughly 55k tree nodes in
total. Numbers are reproducible with `rstar-xccd/analyse_search_signals.py`.

## The constraint these signals operate under

With `is_sampling: False` the search must never consult
`self.ground_truth`, so every reward has to be derived from the branch's own
trajectory. That rules out correctness and leaves three families:

1. **Execution-grounded** — did the code run, what did it produce.
   (This is what the `exec-outcome-scoring` branch uses.)
2. **Form-grounded** — is the output well-structured. (Signal 1 below.)
3. **Self-consistency** — does the branch's conclusion follow from its own
   earlier statements. (Signal 2 below, and the original
   `assert-consistency-score` design.)

## Signal 1 — step-tag conformance

**Idea.** The prompt specifies a canonical step structure
(`<analysis>` → `<end_of_analysis>` → `<code>` → `<end_of_code>` →
`<output>` → `<end_of_output>` → `<answer>` → `<end_of_answer>`). Score a
branch on how well its emitted tag sequence follows that template, on the
theory that degenerate branches produce degenerate structure.

**Measurement.** Tag sequence reconstructed along each root→leaf path and
tested for canonical ordering.

| run | conforms | deviates | accuracy if conforms | accuracy if deviates |
|---|---:|---:|---:|---:|
| Qwen3-4B java | 2168 | 45 (2.0%) | 0.886 | 0.956 |
| Qwen3-4B rust | 1439 | 13 (0.9%) | 0.884 | 0.923 |
| Qwen2.5 java | 2661 | 9 (0.3%) | 0.902 | 1.000 |
| Qwen2.5 rust | 1740 | 6 (0.3%) | 0.813 | 1.000 |

**Rejected because there is no variance to score.** Both models follow the
template in 98%–99.7% of branches, so the reward would be a near-constant
and contribute nothing to PUCT's ability to discriminate. Where deviation
does occur it is *positively* associated with correctness in all four runs,
so the natural sign of the reward is backwards.

**The instructive part.** The intuition behind this signal is sound — there
*is* a large malformed-output population — but it is misattributed.
"No valid Python code found in the response" fires 170–183 times per run,
yet those steps carry correct tags; what fails is that the content between
`<code>` and `<end_of_code>` is not parseable Python. Structure at the tag
level and structure at the content level come apart, and only the latter
carries signal. This is handled instead by scoring the interpreter's
`EXEC_NO_CODE` outcome (see `MCTS_SEARCH_SCORING.md`).

Caveat: conformance was tested as canonical *ordering*. A stricter variant
requiring all four sections to be present might separate more, but with
deviation under 2% of leaves the ceiling on any variant is negligible.

## Signal 2 — pre-code verdict vs. final conclusion

**Idea.** Read the branch's leaning *before* it writes code, compare it to
the leaf's conclusion, and reward agreement (+1) / penalize disagreement
(−1). Label-free, and unlike the assert-consistency signal it applies to
nearly every branch rather than the 3%–8% that execute assertions.

**No judge is needed to extract the pre-code verdict.** The prompt
instructs the model to write `print("not code clones")` in `<code>` when it
has already decided non-clone, so the verdict is recoverable
deterministically: a verdict-echo observation means non-clone, an `assert`
in the code means clone. This covers 56%–85% of leaves at zero inference
cost.

### Why agreement is the wrong variable

At the agreement level the signal looks excellent — Qwen3-4B/java: 0.923
accuracy when pre-code verdict matches the leaf, 0.000 when it differs.
That separation is an artifact. Decomposing into the four transitions:

| run | `clone→clone` | `clone→non-clone` | `non-clone→non-clone` | `non-clone→clone` | no verdict |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B java | 137/137 = 1.000 | **0/66 = 0.000** | 943/1033 = 0.913 | — | 883/977 = 0.904 |
| Qwen3-4B rust | 87/91 = 0.956 | **1/50 = 0.020** | 1020/1087 = 0.938 | — | 176/224 = 0.786 |
| Qwen2.5 java | 289/289 = 1.000 | **0/39 = 0.000** | 1157/1335 = 0.867 | 51/51 = 1.000 | 912/956 = 0.954 |
| Qwen2.5 rust | 40/65 = 0.615 | 2/6 = 0.333 | 1028/1156 = 0.889 | 20/35 = 0.571 | 330/484 = 0.682 |

Two disjoint phenomena were being averaged together:

- **`clone → non-clone` (the "give-up flip") is the sharpest signal found
  anywhere in this data: 1 correct out of 155** across the three
  substantive runs. The branch decides the pair is worth testing, the test
  fails against a model-invented expected literal, and it abandons "clone".
- **`non-clone → non-clone` (the "echo match") is pure base rate.** It is
  60%–75% of all leaves, and its 0.87–0.94 accuracy just restates that the
  models are near-perfect on non-clones, which are 75% of the test set.

A +1/−1 agreement rule sends most of its reward mass to the echo match,
where +1 would be paid to 90, 67, 178, and 128 *wrong* leaves per run
respectively — precisely the false-negative population the runs are losing.

### What the data would actually support

Scoring the transition rather than the agreement: −1 for `clone→non-clone`,
+1 for `clone→clone` and `non-clone→clone`, and **0** for the echo match.

**Shelved anyway**, for three reasons:

1. **Largely redundant.** The give-up flip is the same pathology the
   `exec-outcome-scoring` branch already addresses by removing the
   `AssertionError → "non-clone"` inference. This signal catches a wider
   population (155 leaves vs. 110 assert-tagged nodes) but not a different
   one.
2. **Direction is not stable across model×dataset.** On Qwen2.5/rust,
   `clone→clone` drops to 0.615 and `non-clone→clone` to 0.571 — the rule
   would have to be enabled per-model rather than globally.
3. **It leans toward "clone" conclusions**, trading precision for recall.
   Qwen3 has headroom (precision 0.99–1.00); Qwen2.5/rust does not
   (precision 0.66), and that is exactly where the rule is weakest.

### The LLM-judge variant

The proposal included prompting the model to judge whether the pre-code
verdict and the conclusion match. That is unnecessary for the comparison
itself — both sides are clone/non-clone labels, so it is a string compare
dressed as inference.

A judge would only earn its cost on the **"no pre-code verdict"** bucket
(977 / 224 / 956 / 484 leaves; 44%, 15%, 36%, 28%), where the code block
reveals no leaning and the `<analysis>` prose would have to be read. That
bucket's accuracy is already 0.79–0.95, in line with the overall rate, so
it is not evidently where wins are hiding — and wiring an inference call
into `_score_pending_assert_verdicts()` requires threading the LLM engine
through the agent and batching across trees, a non-trivial `solver.py`
refactor.

## Two cross-cutting findings worth the discussion section

### 1. The base-rate trap in self-consistency scoring

Every self-consistency signal evaluated here looked strong at first and
collapsed on stratification. The `assert-consistency-score` design was the
first instance: its `clone` tag was 100% "accurate" (131/131, 66/66) and
its `non-clone` tag 0% (0/38, 0/27, 0/45) — but assertions are only written
on pairs the model takes seriously, and that population is ~99%
ground-truth clone. Neither tag was discriminative; one simply matched the
prior and the other opposed it. Signal 2's agreement rate reproduced the
same illusion.

The general point: on a class-imbalanced task where the model is already
near-perfect on the majority class, *any* self-consistency measure will
report high accuracy that is entirely base rate. Such signals must be
evaluated stratified by ground truth — even though the search itself, by
construction, cannot see it. Offline stratified analysis of rollouts is
therefore not optional; it is the only way to tell a real signal from a
prior.

### 2. Search-signal work is capped by branch diversity, not by signal quality

Across all four runs the trees produce **1.2–1.5 real answers each**, and
95%–99% of trees are label-homogeneous. 84%–87% of nodes never reach a
final answer at all.

| run | real answers / tree | homogeneous trees | F1 now → F1 with an oracle picking the best existing leaf |
|---|---:|---:|---|
| Qwen3-4B java | 1.23 | 97.4% | 0.9002 → 0.9174 |
| Qwen3-4B rust | 1.21 | 98.6% | 0.7773 → 0.7984 |
| Qwen2.5 java | 1.49 | 95.4% | 0.8897 → 0.9200 |
| Qwen2.5 rust | 1.46 | 95.4% | 0.6711 → 0.7032 |

No reward function can exceed that oracle. Raising `n_generate_sample` from
2 to 4 did not help either (F1 +0.009 java, +0.004 rust; McNemar p = 0.85
and 1.00), so the trees are not homogeneous for lack of sampling budget —
the models are simply confident and converge to one answer.

This reframes the contribution of any search-scoring work on this task:
its value is in *not corrupting* the search (the previous scoring actively
penalized correct branches on clone instances), not in unlocking large
gains. Reporting the oracle-over-existing-leaves ceiling alongside any
search-based result would make this legible, and is a cheap, honest
diagnostic other work in this area generally omits.

## Figures

Three figures in `rstar-xccd/figures/` (PDF for the paper, PNG for preview),
built by `figures/make_discussion_figures.py`:

| Figure | Carries | Why a figure beats the table |
|---|---|---|
| `fig1_reward_allocation` | share of code-execution steps by observation class, per run | the content-free class is 52%–80% of every run; a stacked bar makes that a single glance rather than four columns to compare |
| `fig2_transition_decomposition` | agreement split into its four transitions, accuracy + n | the whole argument is that one number averages a 0.00 bucket with a base-rate bucket — side-by-side bars show the spread that the average hides |
| `fig3_ceiling` | pure inference vs MCTS vs oracle-over-leaves, per model × dataset | three points on one axis show simultaneously that the MCTS→oracle gap is tiny and that pure inference beats MCTS on Qwen2.5-3B/java |

Palette is the validated categorical default (light mode); every mark is
directly labelled, which is also the required relief for the slots under
3:1 contrast on a light surface.

## Reproduction

```
cd rstar-xccd
python analyse_search_signals.py            # all tables above
python analyse_search_signals.py --help     # to point it at other runs
```

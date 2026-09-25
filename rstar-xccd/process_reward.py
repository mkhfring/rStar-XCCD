"""Scalar reward for one root-to-leaf trace = outcome + process penalties
(SFT_DATA_PROCESS.txt section 9).

Clone detection is binary, so an outcome-only reward pays a coin-flip guess
half the time; the process terms below are what stop RL / preference data
from reinforcing right-for-the-wrong-reason traces. The same trace_checks
flags drive the strict SFT filter, so SFT curation, preference pairs
(build_preference_data.py) and any online RL reward agree.

    reward = +1.0  correct label, tier A (verified by real execution output)
             +0.5  correct label, tier B (honest but unverified)
             -0.5  correct label, tier X (claims tests that never ran, clone
                   concluded from inputs on which the programs diverge, ...)
             -1.0  wrong label
              0.0  no parsable label
An optional leaf-review verdict (leaf_review_*.jsonl) overrides the tier:
"keep" -> tier unchanged, "drop" -> -0.5, "repair" -> +0.5 (the original
leaf's reasoning was right but its evidence was missing).

Usage (library): leaf_reward(tree, tag, truth, diff_record, verdict=None)
"""
from evaluate_clone_results import normalize_label
from trace_checks import exec_outcomes, leaf_flags, leaf_tier

TIER_REWARD = {"A": 1.0, "B": 0.5, "X": -0.5}


def leaf_reward(tree, tag, truth, diff_record=None, verdict=None):
    node = tree.get(tag) or {}
    label = normalize_label((node.get("final_answer") or "").strip()) if node.get("final_answer") else None
    if label is None:
        return 0.0
    if label != truth:
        return -1.0
    if verdict == "drop":
        return -0.5
    if verdict == "repair":
        return 0.5
    tier = leaf_tier(leaf_flags(tree, tag, label, diff_record), exec_outcomes(tree, tag))
    return TIER_REWARD[tier]

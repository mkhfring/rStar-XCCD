"""Validate the --exec-signature override on held-out data.

The override answers "clone" whenever a run's tree contains a code step that
never actually ran (no_code / blocked / assertion_failed). The trigger set was
not chosen by reading whole-run F1 -- that would be selection on the test set,
the same objection recorded against clone-on-disagreement in
evaluate_clone_results.predict_label(). This script instead runs the protocol
the choice is defended by, and regenerates every table quoted there:

  1. base rates       -- what the trigger population's ground truth actually is
  2. split-half       -- fit the trigger subset on one half, score on the other
  3. cross-run        -- fit on one run, score on a *different* run

Run:  python3 validate_exec_signature.py
"""
import itertools
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_clone_results import (  # noqa: E402
    DEFAULT_EXEC_SIGNATURE,
    EXEC_SIGNATURE_TRIGGERS,
    MAJORITY,
    predict_label,
    tree_exec_outcomes,
)

EVAL_DIR = Path(__file__).parent / "eval_data"

# One entry per (pair, model, branch). Kept explicit rather than globbed so a
# stray re-scored copy in eval_data/ can't silently join the table.
RUNS = {
    "rust/Q3   baseline": "test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.20260903014301_sampling_false.jsonl",
    "rust/Q3   assert":   "test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.assert-consistency-score.20260904210649.jsonl",
    "rust/Q3   exec":     "test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.exec-outcome-scoring.20260906232842.jsonl",
    "java/Q3   baseline": "test_python_java_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.20260903014301-sampling-false.jsonl",
    "java/Q3   assert":   "test_python_java_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.assert-consistency-score.20260904205237.jsonl",
    "java/Q3   exec":     "test_python_java_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.exec-outcome-scoring.20260906232842.jsonl",
    "rust/Q2.5 baseline": "test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen2.5-Coder-3B-Instruct.20260903190904.jsonl",
    "rust/Q2.5 assert":   "test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen2.5-Coder-3B-Instruct.assert-consistency-score.20260905233058.jsonl",
    "rust/Q2.5 exec":     "test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen2.5-Coder-3B-Instruct.exec-outcome-scoring.20260906233732.jsonl",
    "java/Q2.5 baseline": "test_python_java_CLCCD_depth_16.jsonl.mcts.Qwen2.5-Coder-3B-Instruct.20260903174715.jsonl",
    "java/Q2.5 assert":   "test_python_java_CLCCD_depth_16.jsonl.mcts.Qwen2.5-Coder-3B-Instruct.assert-consistency-score.20260905233058.jsonl",
    "java/Q2.5 exec":     "test_python_java_CLCCD_depth_16.jsonl.mcts.Qwen2.5-Coder-3B-Instruct.exec-outcome-scoring.20260906233135.jsonl",
}

SUBSETS = [frozenset(combo)
           for size in range(len(EXEC_SIGNATURE_TRIGGERS) + 1)
           for combo in itertools.combinations(EXEC_SIGNATURE_TRIGGERS, size)]


def load(path):
    """(ground_truth, base_prediction, tree_outcomes) per instance.

    The base prediction is computed once with the override off, so scoring a
    candidate trigger set later is a set-intersection rather than a re-parse.
    exec_signature=frozenset() is therefore load-bearing, not decoration:
    predict_label() defaults the override ON since 2026-09-07, and letting
    that default through here would fold the override into the baseline and
    report every held-out gain as zero.
    """
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "question" not in record:
                continue
            tree = record.get("rstar", {})
            rows.append(((record.get("answer") or "").strip().lower(),
                         predict_label(tree, aggregation=MAJORITY,
                                       exec_signature=frozenset()),
                         tree_exec_outcomes(tree)))
    return rows


def f1(rows, trigger):
    tp = fp = fn = 0
    for ground_truth, base, outcomes in rows:
        predicted = "clone" if (outcomes & trigger) else base
        if predicted == "clone":
            tp += ground_truth == "clone"
            fp += ground_truth == "non-clone"
        elif predicted == "non-clone":
            fn += ground_truth == "clone"
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    loaded = {}
    for label, name in RUNS.items():
        path = EVAL_DIR / name
        if path.exists():
            loaded[label] = load(path)
        else:
            print(f"skipping {label}: {name} not found", file=sys.stderr)

    print("=" * 100)
    print("1. BASE RATES -- ground truth of the population each trigger fires on")
    print("=" * 100)
    print(f"{'run':<20} {'trigger':<18} {'fires':>6} {'clone':>7} {'non-clone':>10} {'purity':>8} {'acc there':>10}")
    for label, rows in loaded.items():
        for trigger in EXEC_SIGNATURE_TRIGGERS:
            hit = [r for r in rows if trigger in r[2]]
            if not hit:
                continue
            clone = sum(r[0] == "clone" for r in hit)
            correct = sum(r[0] == r[1] for r in hit)
            print(f"{label:<20} {trigger:<18} {len(hit):>6} {clone:>7} {len(hit) - clone:>10} "
                  f"{clone / len(hit):>8.3f} {correct / len(hit):>10.3f}")

    print()
    print("=" * 100)
    print("2. SPLIT-HALF -- trigger set fitted on one half, F1 gain measured on the other")
    print("=" * 100)
    print(f"{'run':<20} {'n':>5} {'base F1':>8} {'held-out gain':>14} {'sd':>7}  modal fitted set")
    for label, rows in loaded.items():
        rng = random.Random(0)
        gains, picks = [], []
        for _ in range(50):
            order = list(range(len(rows)))
            rng.shuffle(order)
            half = len(order) // 2
            fit = [rows[i] for i in order[:half]]
            held = [rows[i] for i in order[half:]]
            pick = max(SUBSETS, key=lambda s: f1(fit, s))
            picks.append(pick)
            gains.append(f1(held, pick) - f1(held, frozenset()))
        modal = max(set(picks), key=picks.count)
        print(f"{label:<20} {len(rows):>5} {f1(rows, frozenset()):>8.4f} "
              f"{statistics.mean(gains):>+14.4f} {statistics.stdev(gains):>7.4f}  "
              f"{'+'.join(sorted(modal)) or '(none)'} [{picks.count(modal)}/50]")

    print()
    print("=" * 100)
    print("3. CROSS-RUN -- the shipped default applied to runs it was never fitted on")
    print("=" * 100)
    print(f"default set: {'+'.join(sorted(DEFAULT_EXEC_SIGNATURE))}")
    print(f"{'run':<20} {'base F1':>8} {'with default':>13} {'delta':>8}")
    for label, rows in loaded.items():
        before = f1(rows, frozenset())
        after = f1(rows, DEFAULT_EXEC_SIGNATURE)
        print(f"{label:<20} {before:>8.4f} {after:>13.4f} {after - before:>+8.4f}")


if __name__ == "__main__":
    main()

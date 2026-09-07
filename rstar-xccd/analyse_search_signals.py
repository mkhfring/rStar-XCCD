"""Regenerate the measurements in ../SEARCH_SIGNAL_DESIGN_SPACE.md.

Evaluates two label-free search signals that were considered for the MCTS
scorer and rejected, against completed MCTS run files:

  Signal 1 -- step-tag conformance: does a branch's emitted tag sequence
  follow the prompt's <analysis>/<code>/<output>/<answer> template, and
  does conformance predict correctness?

  Signal 2 -- pre-code verdict vs. final conclusion: does the leaning the
  branch held before writing code agree with what it concluded, and does
  agreement predict correctness?

Both are reported stratified by ground truth. That stratification is the
point: on this task the majority class (non-clone, 75% of both test sets)
is one the models already get near-perfectly, so any self-consistency
measure looks accurate in aggregate purely by base rate. The search itself
never sees ground truth (is_sampling: False) -- this is an offline audit of
what a proposed reward *would* have paid out.

Also prints the branch-diversity ceiling: no reward function can beat an
oracle that picks the best answer already present in the tree.
"""
import argparse
import collections
import json
import re
import sys

from evaluate_clone_results import normalize_label, node_sort_key, predict_label

# The canonical step structure from rstar_deepthink/few_shots/mcts_prompt.json.
CANONICAL_TAGS = [
    "analysis", "end_of_analysis",
    "code", "end_of_code",
    "output", "end_of_output",
    "answer", "end_of_answer",
]
STRUCTURAL_TAG_RE = re.compile(r"<(" + "|".join(CANONICAL_TAGS) + r")>")
ASSERT_RE = re.compile(r"\bassert\b")

# An observation this short that reads as a clone/non-clone label is the
# model echoing its own verdict, which the prompt explicitly asks for when
# the model has already decided non-clone ("only write: print(\"not code
# clones\")"). That makes the pre-code verdict recoverable without an LLM
# judge.
VERDICT_ECHO_MAX_LEN = 64

DEFAULT_RUNS = {
    "Qwen3-4B java": "eval_data/test_python_java_CLCCD_depth_16.jsonl.mcts."
                     "Qwen3-4B.assert-consistency-score.20260904205237.jsonl",
    "Qwen3-4B rust": "eval_data/test_python_rust_CLCCD_depth_16.jsonl.mcts."
                     "Qwen3-4B.assert-consistency-score.20260904210649.jsonl",
    "Qwen2.5 java": "eval_data/test_python_java_CLCCD_depth_16.jsonl.mcts."
                    "Qwen2.5-Coder-3B-Instruct.assert-consistency-score.20260905233058.jsonl",
    "Qwen2.5 rust": "eval_data/test_python_rust_CLCCD_depth_16.jsonl.mcts."
                    "Qwen2.5-Coder-3B-Instruct.assert-consistency-score.20260905233058.jsonl",
}


def iter_instances(path):
    """Yield (ground_truth, tree) per question, skipping the timing footer."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "question" not in record:
                continue
            yield (record.get("answer") or "").strip().lower(), record.get("rstar", {})


def iter_answer_leaves(tree):
    """Yield (tag, node, label) for every node carrying a real clone verdict.

    Ordered by node tag, matching predict_label()'s traversal, so that
    majority-vote ties here break the same way the offline evaluator breaks
    them (earliest-visited node wins).
    """
    numbered = [(node_sort_key(tag), tag, node) for tag, node in tree.items()
                if isinstance(node, dict) and node_sort_key(tag) is not None]
    for _key, tag, node in sorted(numbered):
        label = normalize_label((node.get("final_answer") or "").strip())
        if label is not None:
            yield tag, node, label


def ancestors_of(tree, tag):
    """Root-to-node path, as node dicts."""
    parts = tag.split(".")
    prefixes = [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
    return [tree[p] for p in prefixes if p in tree]


def conforms_to_template(tags):
    """True if `tags` advances monotonically through CANONICAL_TAGS."""
    highest = -1
    for tag in tags:
        position = CANONICAL_TAGS.index(tag)
        if position <= highest:
            return False
        highest = position
    return True


def pre_code_verdict(tree, tag):
    """The branch's leaning before its conclusion, or None if it never showed one.

    Read from the code the branch actually ran: a verdict echo means it had
    already decided non-clone, an assert means it thought the pair was worth
    testing (i.e. plausibly a clone).
    """
    verdict = None
    for node in ancestors_of(tree, tag):
        observation = (node.get("observation") or "").strip()
        action_input = node.get("action_input") or ""
        if observation and len(observation) < VERDICT_ECHO_MAX_LEN and normalize_label(observation):
            verdict = normalize_label(observation)
        elif ASSERT_RE.search(action_input):
            verdict = "clone"
    return verdict


def _rate(numerator, denominator):
    return f"{numerator / denominator:.3f}" if denominator else "  -  "


def report_tag_conformance(runs):
    print("### Signal 1 -- step-tag conformance\n")
    header = (f"{'run':15s} {'tag sequence':14s} {'n':>6s} {'acc':>7s} "
              f"{'acc|gt=clone':>13s} {'acc|gt=non':>11s}")
    print(header)
    print("-" * len(header))
    for name, path in runs.items():
        buckets = collections.defaultdict(collections.Counter)
        for ground_truth, tree in iter_instances(path):
            for tag, _node, label in iter_answer_leaves(tree):
                tags = []
                for ancestor in ancestors_of(tree, tag):
                    tags += STRUCTURAL_TAG_RE.findall(ancestor.get("text") or "")
                collapsed = [t for i, t in enumerate(tags) if i == 0 or t != tags[i - 1]]
                key = "conforms" if conforms_to_template(collapsed) else "deviates"
                buckets[key][(ground_truth, label == ground_truth)] += 1
        for key in ("conforms", "deviates"):
            counts = buckets[key]
            total = sum(counts.values())
            if not total:
                continue
            correct = counts[("clone", True)] + counts[("non-clone", True)]
            n_clone = counts[("clone", True)] + counts[("clone", False)]
            n_non = counts[("non-clone", True)] + counts[("non-clone", False)]
            print(f"{name:15s} {key:14s} {total:6d} {correct / total:7.3f} "
                  f"{_rate(counts[('clone', True)], n_clone):>13s} "
                  f"{_rate(counts[('non-clone', True)], n_non):>11s}")
        print()


def report_verdict_transitions(runs):
    print("\n### Signal 2 -- pre-code verdict -> final conclusion, by transition\n")
    header = f"{'run':15s} {'transition':26s} {'n':>6s} {'correct':>8s} {'wrong':>6s} {'accuracy':>9s}"
    print(header)
    print("-" * len(header))
    ordering = ["clone -> clone", "clone -> non-clone",
                "non-clone -> non-clone", "non-clone -> clone",
                "no pre-code verdict"]
    notes = {
        "clone -> non-clone": "  <-- the give-up flip",
        "non-clone -> non-clone": "  <-- the echo match (base rate)",
    }
    for name, path in runs.items():
        counts = collections.Counter()
        for ground_truth, tree in iter_instances(path):
            for tag, _node, label in iter_answer_leaves(tree):
                verdict = pre_code_verdict(tree, tag)
                key = "no pre-code verdict" if verdict is None else f"{verdict} -> {label}"
                counts[(key, label == ground_truth)] += 1
        for key in ordering:
            correct, wrong = counts[(key, True)], counts[(key, False)]
            if correct + wrong == 0:
                continue
            print(f"{name:15s} {key:26s} {correct + wrong:6d} {correct:8d} {wrong:6d} "
                  f"{correct / (correct + wrong):9.3f}{notes.get(key, '')}")
        print()


def report_diversity_ceiling(runs):
    print("\n### Branch-diversity ceiling (no reward function can beat this)\n")
    header = (f"{'run':15s} {'instances':>10s} {'answers/tree':>13s} "
              f"{'homogeneous':>12s} {'F1 now':>8s} {'F1 oracle':>10s}")
    print(header)
    print("-" * len(header))
    for name, path in runs.items():
        instances = leaves = homogeneous = 0
        current, oracle = [], []
        for ground_truth, tree in iter_instances(path):
            labels = [label for _tag, _node, label in iter_answer_leaves(tree)]
            if not labels:
                continue
            instances += 1
            leaves += len(labels)
            if len(set(labels)) == 1:
                homogeneous += 1
            # Reuse the offline evaluator's rule verbatim, so "F1 now" is
            # exactly the number in the run's own _result file.
            majority = predict_label(tree)
            current.append((ground_truth, majority))
            oracle.append((ground_truth, ground_truth if ground_truth in labels else majority))
        print(f"{name:15s} {instances:10d} {leaves / instances:13.2f} "
              f"{homogeneous / instances:11.1%} {_f1(current):8.4f} {_f1(oracle):10.4f}")


def _f1(pairs):
    tp = fp = fn = 0
    for ground_truth, predicted in pairs:
        if predicted == "clone" and ground_truth == "clone":
            tp += 1
        elif predicted == "clone":
            fp += 1
        elif ground_truth == "clone":
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", metavar="LABEL=PATH",
                        help="MCTS output files to analyse. Defaults to the four "
                             "completed assert-consistency-score CLCCD runs.")
    args = parser.parse_args()

    if args.runs:
        runs = {}
        for entry in args.runs:
            label, _, path = entry.partition("=")
            runs[label if path else path or label] = path or label
    else:
        runs = DEFAULT_RUNS

    missing = [p for p in runs.values() if not _exists(p)]
    if missing:
        print("Missing run file(s):", *missing, sep="\n  ", file=sys.stderr)
        return 1

    report_tag_conformance(runs)
    report_verdict_transitions(runs)
    report_diversity_ceiling(runs)
    return 0


def _exists(path):
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(main())

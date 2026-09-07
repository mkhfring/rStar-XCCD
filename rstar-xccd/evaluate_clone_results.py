"""Evaluate clone-detection MCTS results: precision, recall, F1, and response rate.

Each line of the input file is one question with ground truth label `answer`
(`clone` / `non-clone`) and an MCTS search tree under `rstar`. The tree has no
single designated "final answer" field at the top level, so for every leaf
node whose `final_answer` is non-empty and not one of the search's failure
placeholders ("Too many consecutive steps have code errors.", "Fail to sove
the problem within limited steps.", "Fail to generate parsable text for next
step."), we collect the (normalized, lower-cased/stripped) text. Note: the
per-node "value"/"q_value" fields in this file already encode correctness
against the ground truth (this run was produced with is_sampling=True), so
they cannot be used to pick a "best" candidate without leaking the label.
Instead, the model's predicted label for a question is decided by majority
vote among its leaf nodes whose normalized final_answer maps to "clone" or
"non-clone" (ties broken by the lexicographically-earliest node tag, which is
also the earliest-visited node). Models sometimes answer with a paraphrase of
the requested label instead of the literal word (e.g. "semantic clone" or
"semantic_clone" instead of "clone") -- normalize_label() maps these variants
to the intended label so they still count as a vote. A question is counted as
"no judgment" (and contributes to the response-rate count) only if no leaf
node in its tree produced a final_answer that normalizes to "clone" or
"non-clone". Lines without a "question" field (e.g. the run's trailing
timing/footer record) are skipped entirely rather than counted as instances.
"""
import argparse
import json
import sys
from pathlib import Path

FAILURE_PLACEHOLDERS = {
    "too many consecutive steps have code errors.",
    "fail to sove the problem within limited steps.",
    "fail to generate parsable text for next step.",
}
VALID_LABELS = {"clone", "non-clone"}


def node_sort_key(tag):
    """Sort key for a dot-numeric node tag (e.g. "0.1.2"), or None if the
    tag isn't one -- some pipelines (e.g. beam search) stash extra
    non-node keys like "solutions" in the same rstar dict."""
    parts = tag.split(".")
    if not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def normalize_label(final_answer):
    """Map a leaf node's final_answer text to 'clone'/'non-clone', or None if
    it doesn't express either label (e.g. a failure placeholder)."""
    text = final_answer.strip().lower().replace("_", " ")
    if text in VALID_LABELS:
        return text
    if text in FAILURE_PLACEHOLDERS:
        return None
    if "non-clone" in text or "non clone" in text or "not clone" in text or "not a clone" in text or "not code clone" in text:
        return "non-clone"
    if "clone" in text:
        return "clone"
    return None


MAJORITY = "majority"
CLONE_ON_DISAGREEMENT = "clone-on-disagreement"
AGGREGATIONS = (MAJORITY, CLONE_ON_DISAGREEMENT)


def predict_label(rstar_tree, aggregation=MAJORITY):
    """Return the label ('clone'/'non-clone') for a question's tree, or None
    if no leaf node produced an answer that normalizes to clone/non-clone.

    `aggregation` selects how the leaves' votes are combined:

    "majority" (default, unchanged): plain majority vote, ties broken by the
    earliest node tag.

    "clone-on-disagreement": if any leaf says "clone", answer "clone".
    Opt-in, and only defensible for a model whose precision far exceeds its
    recall on this task -- the asymmetry is the whole justification. When
    the non-clone side is already saturated, conceding disagreements to
    "clone" costs almost no precision and recovers false negatives; when it
    is not, the rule is a coin flip that spends precision for nothing.

    Measured over the four assert-consistency-score CLCCD runs, on the trees
    where leaves actually disagree, the clone side was correct 47/47
    (Qwen3-4B java), 15/17 (Qwen3-4B rust) and 83/83 (Qwen2.5-3B java) --
    but only 28/55 on Qwen2.5-3B rust, whose precision is 0.66. Do not
    enable it there.

    Caveat for anything published off this: the rule was chosen by
    inspecting test-set F1 on those runs. Justify it from a precision/recall
    asymmetry measured on held-out data rather than from the scores it
    produces here.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")
    numeric_nodes = [(tag, node) for tag, node in rstar_tree.items() if node_sort_key(tag) is not None]
    votes = []
    for tag, node in sorted(numeric_nodes, key=lambda kv: node_sort_key(kv[0])):
        final_answer = (node.get("final_answer") or "").strip()
        if not final_answer:
            continue
        label = normalize_label(final_answer)
        if label is not None:
            votes.append(label)
    if not votes:
        return None
    if aggregation == CLONE_ON_DISAGREEMENT and "clone" in votes:
        return "clone"
    counts = {label: votes.count(label) for label in VALID_LABELS}
    if counts["clone"] == counts["non-clone"]:
        return votes[0]  # tie -> earliest node's vote
    return max(counts, key=counts.get)


def evaluate(input_path, aggregation=MAJORITY):
    total = 0
    tp = fp = tn = fn = 0
    no_judgment = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "question" not in record:
                continue
            total += 1

            ground_truth = (record.get("answer") or "").strip().lower()
            predicted = predict_label(record.get("rstar", {}), aggregation=aggregation)

            if predicted is None:
                no_judgment += 1
                continue

            if predicted == "clone" and ground_truth == "clone":
                tp += 1
            elif predicted == "clone" and ground_truth == "non-clone":
                fp += 1
            elif predicted == "non-clone" and ground_truth == "non-clone":
                tn += 1
            elif predicted == "non-clone" and ground_truth == "clone":
                fn += 1

    judged = total - no_judgment
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    response_rate = no_judgment / total if total else 0.0

    return {
        "total_instances": total,
        "judged_instances": judged,
        "no_judgment_instances": no_judgment,
        "response_rate": response_rate,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "aggregation": aggregation,
    }


def format_report(stats, input_path):
    lines = [
        f"Evaluation results for: {input_path}",
        "",
        "Positive class: 'clone' | Negative class: 'non-clone'",
        f"Leaf aggregation: {stats.get('aggregation', MAJORITY)}",
        f"Total instances: {stats['total_instances']}",
        f"Instances with a clone/non-clone final judgment: {stats['judged_instances']}",
        f"Instances with no clone/non-clone final judgment: {stats['no_judgment_instances']}",
        "",
        "Response rate (instances with no clone/non-clone final judgment / total instances):",
        f"  {stats['no_judgment_instances']} / {stats['total_instances']} = {stats['response_rate']:.4f}",
        "",
        "Confusion matrix (computed only over judged instances):",
        f"  True Positive  (predicted clone,     actual clone):     {stats['true_positive']}",
        f"  False Positive (predicted clone,     actual non-clone): {stats['false_positive']}",
        f"  True Negative  (predicted non-clone, actual non-clone): {stats['true_negative']}",
        f"  False Negative (predicted non-clone, actual clone):     {stats['false_negative']}",
        "",
        f"Precision: {stats['precision']:.4f}",
        f"Recall:    {stats['recall']:.4f}",
        f"F1 score:  {stats['f1']:.4f}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Score a clone-detection MCTS/inference run: precision, "
                    "recall, F1, and response rate.")
    parser.add_argument("input_path", nargs="?", type=Path,
                        default=Path("eval_data/test_same_python_java.jsonl.mcts."
                                     "Qwen2.5-Coder-3B-Instruct.20260617030600.jsonl"),
                        help="run output .jsonl to score")
    parser.add_argument("output_path", nargs="?", type=Path, default=None,
                        help="where to write the report (default: alongside the input)")
    parser.add_argument("--aggregation", choices=AGGREGATIONS, default=MAJORITY,
                        help="how to combine a tree's leaf votes. 'majority' is the "
                             "default and is what every existing _result file used. "
                             "'clone-on-disagreement' answers clone whenever any leaf "
                             "does -- only valid for a precision-heavy model; see "
                             "predict_label().")
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        # Non-default aggregations write to their own file, so re-scoring a
        # run never overwrites the report it was originally published with.
        suffix = "_result" if args.aggregation == MAJORITY else f"_result.{args.aggregation}"
        output_path = args.input_path.with_name(args.input_path.name + suffix)

    stats = evaluate(args.input_path, aggregation=args.aggregation)
    report = format_report(stats, args.input_path)

    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()

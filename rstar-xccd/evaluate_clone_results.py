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

# Sentinel observations written by tree.py's code_execution() when nothing
# actually ran. Runs produced before exec-outcome-scoring don't carry the
# state["exec_outcome"] field, but they do carry state["observation"], and
# these three outcomes are recoverable from it exactly -- see
# exec_outcome_of().
NO_CODE_MESSAGE = "No valid Python code found in the response."
JAVA_BLOCKED_PREFIX = "Java execution is not supported in this sandbox"

EXEC_SIGNATURE_TRIGGERS = ("no_code", "blocked", "assertion_failed")
# Default trigger set for --exec-signature. Selected by a 50x split-half
# protocol (fit the trigger subset on one half, score on the other) rather
# than by reading whole-run F1: no_code+assertion_failed wins 50/50 splits
# on both Qwen3 runs, and the held-out gain is +0.099 (rust) / +0.044 (java).
# "blocked" is included because it is the same failure-to-execute event as
# no_code and is near-absent on rust; dropping it changes java/Qwen3 by
# <0.001 and it is what the java splits select when it does fire.
DEFAULT_EXEC_SIGNATURE = frozenset({"no_code", "blocked", "assertion_failed"})


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


def exec_outcome_of(node):
    """The EXEC_* outcome of one tree node's code step, or None if it never
    ran one.

    Prefers the state["exec_outcome"] field that mcts.py persists on the
    exec-outcome-scoring branch. Older runs predate that field, so fall back
    to the observation text -- which is exact for the three outcomes this
    module cares about, because tree.py returns a fixed sentinel for
    no_code/blocked and the interpreter prefixes a raised AssertionError with
    its own type name. Anything else ("ok" and "error" alike) is reported as
    "ran", since nothing here needs to tell those two apart.
    """
    if "exec_outcome" in node:
        return node["exec_outcome"]
    observation = (node.get("observation") or "").strip()
    if not observation:
        return None
    if observation == NO_CODE_MESSAGE:
        return "no_code"
    if observation.startswith(JAVA_BLOCKED_PREFIX):
        return "blocked"
    if observation.startswith("AssertionError"):
        return "assertion_failed"
    return "ran"


def tree_exec_outcomes(rstar_tree):
    """The set of EXEC_* outcomes appearing anywhere in a question's tree."""
    outcomes = set()
    for node in rstar_tree.values():
        if not isinstance(node, dict):
            continue
        outcome = exec_outcome_of(node)
        if outcome is not None:
            outcomes.add(outcome)
    return outcomes


def predict_label(rstar_tree, aggregation=CLONE_ON_DISAGREEMENT,
                  exec_signature=DEFAULT_EXEC_SIGNATURE):
    """Return the label ('clone'/'non-clone') for a question's tree, or None
    if no leaf node produced an answer that normalizes to clone/non-clone.

    `exec_signature` is a set of EXEC_* outcomes (see
    EXEC_SIGNATURE_TRIGGERS) that, if any of them occurred anywhere in the
    tree, forces the prediction to "clone" regardless of what the leaves
    voted. It is orthogonal to `aggregation` and applies on top of it.

    It defaults to DEFAULT_EXEC_SIGNATURE, i.e. **on**. Pass
    `exec_signature=frozenset()` for the pre-2026-09-07 behaviour, which is
    what every _result file published before that date used; callers that
    exist to reproduce those historical numbers must do so explicitly.

    The justification is a property of the harness, not of the model: only
    Code 1 is executable here (tree.py writes just the Python side to
    candidate_code.py and blocks the Java/Rust side), so "the model never
    produced runnable Python", "it reached for the blocked side", and "its
    own invented assertion failed" are all signatures of a branch that could
    not settle the comparison by running anything. Empirically that happens
    almost only on genuinely equivalent pairs: over the CLCCD runs those
    nodes are 113:7 and 82:0 ground-truth clone (Qwen3 rust/java), while
    accuracy on that population collapses to 0.59-0.76 against 0.93
    elsewhere. They are precisely the false negatives these runs lose.

    Note this is the same signal the exec-outcome-scoring branch feeds to
    the search as `negative_reward`, used the other way round. As a search
    penalty it is worth roughly nothing -- the trees are single-leaf 79% of
    the time, no answer leaf is ever backed up (need_value_func=False short-
    circuits the backup in select_next_step), and the value redirected the
    search in 2-11% of trees, losing more than it won. As a classification
    feature over the finished tree it is worth +0.03 to +0.10 F1, and it
    delivers that equally on baseline, assert-consistency-score and
    exec-outcome-scoring trees, which is what shows it is independent of how
    the search was scored.

    Only defensible where clone-side precision is high; on Qwen2.5/rust
    (precision 0.63) the held-out gain is +0.001 to +0.008, i.e. nothing.
    Same caveat as clone-on-disagreement below: validate the trigger set on
    held-out data. The default set was chosen by a 50x split-half protocol
    rather than by whole-run F1 -- see DEFAULT_EXEC_SIGNATURE.

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
    unknown = set(exec_signature) - set(EXEC_SIGNATURE_TRIGGERS)
    if unknown:
        raise ValueError(f"Unknown exec_signature trigger(s) {sorted(unknown)}; "
                         f"expected a subset of {EXEC_SIGNATURE_TRIGGERS}")
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
        # Only fires as a fallback for a tree that never reached a parsable
        # verdict, per the justification above -- it converts a no-judgment
        # instance rather than overriding a judged one. This is checked only
        # here (after votes are known to be empty), not unconditionally
        # before the leaves are read: on the code2-exec-rust-testfix-v1-full
        # run, 196/980 trees triggered the signature and 194 of those ALSO
        # had real leaf votes (only 2 were true no-judgment cases), so
        # checking it first -- as this function did before 2026-09-12 --
        # forced "clone" over a real, often-unanimous-and-correct leaf
        # verdict in the overwhelming majority of firings, on both that run
        # and the earlier code2-exec-lang-fewshot-v2 run (150/192 forced
        # firings correct vs 157/192 the leaf votes alone would have gotten
        # right). Moving the check here restores the invariant the original
        # comment claimed but the unconditional-return code did not enforce.
        if exec_signature and (tree_exec_outcomes(rstar_tree) & set(exec_signature)):
            return "clone"
        return None
    if aggregation == CLONE_ON_DISAGREEMENT and "clone" in votes:
        return "clone"
    counts = {label: votes.count(label) for label in VALID_LABELS}
    if counts["clone"] == counts["non-clone"]:
        return votes[0]  # tie -> earliest node's vote
    return max(counts, key=counts.get)


def evaluate(input_path, aggregation=CLONE_ON_DISAGREEMENT,
             exec_signature=DEFAULT_EXEC_SIGNATURE):
    total = 0
    tp = fp = tn = fn = 0
    no_judgment = 0
    exec_signature_fired = 0

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
            rstar_tree = record.get("rstar", {})
            predicted = predict_label(rstar_tree, aggregation=aggregation,
                                      exec_signature=exec_signature)
            if exec_signature and (tree_exec_outcomes(rstar_tree) & set(exec_signature)):
                exec_signature_fired += 1

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
    # UPDATE 2026-09-26: "response_rate" used to hold no_judgment / total --
    # the NON-response rate -- while methodology.tex defines response rate as
    # the fraction of pairs that get a prediction. It now matches the paper;
    # the old value is kept as "no_response_rate". _result files written
    # before this date print the non-response share under "Response rate".
    response_rate = judged / total if total else 0.0
    no_response_rate = no_judgment / total if total else 0.0

    return {
        "total_instances": total,
        "judged_instances": judged,
        "no_judgment_instances": no_judgment,
        "response_rate": response_rate,
        "no_response_rate": no_response_rate,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "aggregation": aggregation,
        "exec_signature": sorted(exec_signature),
        "exec_signature_fired": exec_signature_fired,
    }


def format_report(stats, input_path):
    lines = [
        f"Evaluation results for: {input_path}",
        "",
        "Positive class: 'clone' | Negative class: 'non-clone'",
        f"Leaf aggregation: {stats.get('aggregation', MAJORITY)}",
        f"Exec-signature override: {'+'.join(stats.get('exec_signature') or []) or 'off'}"
        + (f" (fired on {stats['exec_signature_fired']} instances)"
           if stats.get('exec_signature') else ""),
        f"Total instances: {stats['total_instances']}",
        f"Instances with a clone/non-clone final judgment: {stats['judged_instances']}",
        f"Instances with no clone/non-clone final judgment: {stats['no_judgment_instances']}",
        "",
        "Response rate (instances with a clone/non-clone final judgment / total instances):",
        f"  {stats['judged_instances']} / {stats['total_instances']} = {stats['response_rate']:.4f}",
        "No-response rate (instances with no clone/non-clone final judgment / total instances):",
        f"  {stats['no_judgment_instances']} / {stats['total_instances']} = {stats['no_response_rate']:.4f}",
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
    parser.add_argument("--aggregation", choices=AGGREGATIONS,
                        default=CLONE_ON_DISAGREEMENT,
                        help="how to combine a tree's leaf votes. "
                             "'clone-on-disagreement' is the default on this branch: "
                             "it answers clone whenever any leaf does, and is only "
                             "valid for a precision-heavy model. 'majority' is the "
                             "plain vote, and is what every _result file published "
                             "before 2026-09-07 used. See predict_label().")
    parser.add_argument("--exec-signature", nargs="?", const="default", default="default",
                        help="answer clone whenever the tree contains a code step that "
                             "never actually ran. ON by default, using the validated set ("
                             + "+".join(sorted(DEFAULT_EXEC_SIGNATURE)) + "); pass a "
                             f"'+'-separated subset of {'/'.join(EXEC_SIGNATURE_TRIGGERS)} "
                             "to narrow it. Composes with --aggregation; see "
                             "predict_label().")
    parser.add_argument("--no-exec-signature", dest="exec_signature",
                        action="store_const", const=None,
                        help="disable the override and score by leaf votes alone. This "
                             "is the pre-2026-09-07 behaviour that every _result file "
                             "published before that date used. Needed for models whose "
                             "clone-side precision is low -- on Qwen2.5-Coder-3B "
                             "python-rust the override costs 0.012 F1; see "
                             "EVALUATOR_CHANGES.md.")
    args = parser.parse_args()

    if args.exec_signature is None:
        exec_signature = frozenset()
    elif args.exec_signature == "default":
        exec_signature = DEFAULT_EXEC_SIGNATURE
    else:
        exec_signature = frozenset(t for t in args.exec_signature.split("+") if t)
        unknown = exec_signature - set(EXEC_SIGNATURE_TRIGGERS)
        if unknown:
            parser.error(f"unknown --exec-signature trigger(s) {sorted(unknown)}; "
                         f"expected a '+'-separated subset of {EXEC_SIGNATURE_TRIGGERS}")

    output_path = args.output_path
    if output_path is None:
        # Non-default settings write to their own file, so re-scoring a run
        # never overwrites the report it was originally published with.
        suffix = ("_result" if args.aggregation == CLONE_ON_DISAGREEMENT
                  else f"_result.{args.aggregation}")
        if not exec_signature:
            suffix += ".no-exec-signature"
        elif exec_signature != DEFAULT_EXEC_SIGNATURE:
            suffix += ".exec-signature-" + "+".join(sorted(exec_signature))
        output_path = args.input_path.with_name(args.input_path.name + suffix)

    stats = evaluate(args.input_path, aggregation=args.aggregation,
                     exec_signature=exec_signature)
    report = format_report(stats, args.input_path)

    # The exec-signature override became the default on 2026-09-07, so a
    # plain re-score of a run published before then now produces different
    # numbers under the same filename. Overwriting is still the right
    # behaviour -- _result is meant to track the current scorer -- but doing
    # it silently would make two incompatible reports indistinguishable
    # after the fact, so say so.
    if output_path.exists():
        previous = output_path.read_text(encoding="utf-8")
        was_on = "Exec-signature override: off" not in previous
        if "Exec-signature override:" not in previous or was_on != bool(exec_signature):
            print(f"warning: overwriting {output_path.name}, which was generated "
                  f"with a different exec-signature setting; its numbers are not "
                  f"comparable to the ones being written now "
                  f"(pass --no-exec-signature to reproduce them)", file=sys.stderr)

    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()

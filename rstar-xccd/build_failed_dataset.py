"""Build a question file of the samples every given run got wrong.

Motivation: the Code-2-execution harness (stage_code2() in
rstar_deepthink/tools/python_tool.py) lets the search compile and run Code 2
instead of only reasoning about it. The interesting question is whether that
recovers samples the models previously failed -- so we need those samples as
a standalone question file that main.py can consume unchanged.

A sample "failed" for a run if the frozen CLCCD configuration's prediction
for it is not the ground-truth label. That includes a tree that never reached
a parsable clone/non-clone verdict ("no judgment"): it did not get the sample
right, so it belongs in the retry set. Scoring is imported from
evaluate_clone_results rather than reimplemented, so this file cannot drift
from the evaluator -- see FINAL_FOR_CLCCD.md for what the defaults mean.

By default a sample must have failed in EVERY run given (--mode intersection),
which is what makes the output "hard for both models" rather than "hard for
one". --mode union takes anything that failed anywhere.

Runs need not cover the same samples: a run killed by its wall clock covers a
prefix. Only samples present in every run are eligible under intersection,
and the report states the shortfall explicitly rather than silently scoring a
smaller set.

The output carries exactly the source schema (index, question, answer) so it
drops straight into --qaf. `index` is the ORIGINAL index, so results map back
to the full run. Per-sample provenance -- what each model predicted and how it
failed -- goes to the sidecar report, not into the question file, to keep the
retry run's output schema identical to the original's.
"""
import argparse
import json
import sys
from collections import Counter

from evaluate_clone_results import (
    CLONE_ON_DISAGREEMENT,
    DEFAULT_EXEC_SIGNATURE,
    predict_label,
    tree_exec_outcomes,
)


def load_records(path):
    """{index: record} for lines that are questions, skipping the footer."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "question" not in record:
                continue
            records[record["index"]] = record
    return records


def outcome_for(record, aggregation, exec_signature):
    """(failed, kind, predicted) for one scored record."""
    truth = (record.get("answer") or "").strip().lower()
    tree = record.get("rstar", {})
    predicted = predict_label(tree, aggregation=aggregation,
                              exec_signature=exec_signature)
    if predicted is None:
        return True, "no_judgment", None
    if predicted == truth:
        return False, "correct", predicted
    kind = "false_positive" if predicted == "clone" else "false_negative"
    return True, kind, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                        help="original question file, the schema and text to emit")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                        help="a scored run; repeatable")
    parser.add_argument("--out", required=True, help="question file to write")
    parser.add_argument("--mode", choices=("intersection", "union"), default="intersection",
                        help="failed in every run (default) or in any run")
    parser.add_argument("--aggregation", default=CLONE_ON_DISAGREEMENT)
    parser.add_argument("--no-exec-signature", action="store_true",
                        help="score without the exec-signature override (pre-2026-09-07 behaviour)")
    args = parser.parse_args()

    exec_signature = frozenset() if args.no_exec_signature else DEFAULT_EXEC_SIGNATURE

    source = load_records(args.source)
    runs = {}
    for spec in args.run:
        if "=" not in spec:
            parser.error(f"--run expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        runs[label] = (path, load_records(path))

    # Eligibility is the intersection of coverage: a sample no run scored, or
    # that only some runs reached, cannot be said to have failed everywhere.
    covered = set(source)
    for _, records in runs.values():
        covered &= set(records)

    failures = {}   # label -> {index: kind}
    for label, (_, records) in runs.items():
        kinds = {}
        for index in covered:
            failed, kind, _ = outcome_for(records[index], args.aggregation, exec_signature)
            if failed:
                kinds[index] = kind
        failures[label] = kinds

    sets = [set(k) for k in failures.values()]
    selected = set.intersection(*sets) if args.mode == "intersection" else set.union(*sets)
    selected = sorted(selected)

    with open(args.out, "w", encoding="utf-8") as f:
        for index in selected:
            record = source[index]
            f.write(json.dumps({"index": record["index"],
                                "question": record["question"],
                                "answer": record["answer"]}) + "\n")

    report_path = args.out + "_provenance"
    with open(report_path, "w", encoding="utf-8") as f:
        for index in selected:
            entry = {"index": index,
                     "answer": source[index]["answer"],
                     "runs": {}}
            for label, (_, records) in runs.items():
                failed, kind, predicted = outcome_for(records[index], args.aggregation,
                                                      exec_signature)
                entry["runs"][label] = {
                    "failed": failed,
                    "kind": kind,
                    "predicted": predicted,
                    "exec_outcomes": sorted(tree_exec_outcomes(records[index].get("rstar", {}))),
                }
            f.write(json.dumps(entry) + "\n")

    lines = [
        f"Source:        {args.source} ({len(source)} samples)",
        f"Aggregation:   {args.aggregation}",
        f"Exec-signature:{'+'.join(sorted(exec_signature)) or ' off'}",
        f"Mode:          {args.mode}",
        "",
        "Per-run coverage and failures (over the eligible set):",
    ]
    for label, (path, records) in runs.items():
        missing = len(source) - len(records)
        note = f"  [MISSING {missing} of {len(source)} -- run is incomplete]" if missing else ""
        kinds = Counter(failures[label].values())
        lines.append(f"  {label}: scored {len(records)}{note}")
        lines.append(f"    failed {len(failures[label])} / {len(covered)}  "
                     + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    lines += [
        "",
        f"Eligible (scored by every run): {len(covered)} of {len(source)}",
        f"Selected ({args.mode}):          {len(selected)}",
        "",
        f"Wrote {args.out}",
        f"Wrote {report_path}",
    ]
    report = "\n".join(lines)
    print(report)
    with open(args.out + "_report", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    sys.exit(main())

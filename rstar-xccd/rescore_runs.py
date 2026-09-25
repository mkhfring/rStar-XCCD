"""Re-score finished CLCCD runs under ONE aggregation rule so that numbers
are comparable (MCTS_SEARCH_SCORING.md, "Aggregation audit, 2026-09-24").

Why this exists: the published Qwen3-4B baselines (FINAL_FOR_CLCCD.md, rust
0.8739 / java 0.9454) were scored before 2026-09-12 with the exec-signature
override applied UNCONDITIONALLY; every _result written since (all SFT
checkpoints) applies it only to trees without a leaf vote. Re-scored with
the current rule the same baseline trees give rust 0.7952 / java 0.9163, so
"checkpoint vs published baseline" compared two different metrics.

Rules reported for every run:
  old        pre-2026-09-12 predict_label (override first, unconditionally)
  current    evaluate_clone_results.predict_label defaults
  tested     current, but a NON-CLONE leaf whose path ran a real test (any
             code step that is not the prompt's print("not code clones")
             shortcut) votes CLONE. Model-dependent -- see --calibrate.

--calibrate: for each run, the tested rule is adopted only if it improves F1
on that run's rl_train questions (build_rl_split.py split); the reported
held-out number is then computed on rl_heldout with the rule chosen on
rl_train. This is the only way the tested rule should be reported.

Usage:
    python rescore_runs.py RUN.jsonl [...] --lang rust [--calibrate]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_clone_results import (DEFAULT_EXEC_SIGNATURE, node_sort_key, normalize_label,
                                    predict_label, tree_exec_outcomes)
from trace_checks import chain, code_steps, is_print_only, strip_code_tags


def tested(numeric, tag):
    steps = code_steps(chain(numeric, tag))
    return bool(steps) and not all(is_print_only(strip_code_tags(n.get("action_input"))) for _, n in steps)


def predict_tested(tree):
    numeric = {k: v for k, v in tree.items() if node_sort_key(k) is not None}
    votes = []
    for tag, n in sorted(numeric.items(), key=lambda kv: node_sort_key(kv[0])):
        fa = (n.get("final_answer") or "").strip()
        lab = normalize_label(fa) if fa else None
        if lab is None:
            continue
        votes.append("clone" if lab == "non-clone" and tested(numeric, tag) else lab)
    if not votes:
        return predict_label(tree)
    return "clone" if "clone" in votes else "non-clone"


RULES = {
    "old": lambda t: "clone" if tree_exec_outcomes(t) & set(DEFAULT_EXEC_SIGNATURE) else predict_label(t),
    "current": predict_label,
    "tested": predict_tested,
}


def f1(pairs):
    tp = sum(p == "clone" and a == "clone" for a, p in pairs)
    fp = sum(p == "clone" and a != "clone" for a, p in pairs)
    fn = sum(p != "clone" and a == "clone" for a, p in pairs)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return (2 * P * R / (P + R) if P + R else 0.0), P, R


def fmt(x):
    return f"{x[0]:.4f} (P {x[1]:.3f} R {x[2]:.3f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--lang", required=True, choices=["rust", "java"])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--report_out", help="also write the report to this file")
    args = ap.parse_args()
    lines = []

    def emit(msg=""):
        print(msg)
        lines.append(msg)

    split = {}
    for name in ("rl_train", "rl_heldout", "rl_heldout_strict"):
        p = Path(f"eval_data/{name}_python_{args.lang}_CLCCD.jsonl")
        if p.exists():
            split[name] = {json.loads(l)["index"] for l in open(p)}

    for run in args.runs:
        preds = {rule: [] for rule in RULES}
        for line in open(run, encoding="utf-8"):
            r = json.loads(line)
            if "rstar" not in r:
                continue
            for rule, fn in RULES.items():
                preds[rule].append((r["index"], r["answer"], fn(r["rstar"])))
        name = run.split(".mcts.")[-1][:60]
        n = len(preds["current"])
        emit(f"== {name}  (n={n})")
        for rule in RULES:
            parts = [f"full {fmt(f1([(a, p) for _, a, p in preds[rule]]))}"]
            for s, idx in split.items():
                parts.append(f"{s} {fmt(f1([(a, p) for i, a, p in preds[rule] if i in idx]))}")
            emit(f"   {rule:8s} " + " | ".join(parts))
        if args.calibrate and "rl_train" in split:
            tr = lambda rule: f1([(a, p) for i, a, p in preds[rule] if i in split["rl_train"]])[0]
            chosen = "tested" if tr("tested") > tr("current") else "current"
            out = [f"{s} {fmt(f1([(a, p) for i, a, p in preds[chosen] if i in idx]))}"
                   for s, idx in split.items() if s != "rl_train"]
            emit(f"   calibrated on rl_train -> '{chosen}' (rl_train F1 current {tr('current'):.4f} vs "
                  f"tested {tr('tested'):.4f}); held-out: " + " | ".join(out))

    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as w:
            w.write("Re-scored by rescore_runs.py (MCTS_SEARCH_SCORING.md, 'Scoring review, 2026-09-24').\n"
                    "old = pre-2026-09-12 aggregation; current = evaluate_clone_results default (what\n"
                    "_result reports); tested = NEW RULE (non-clone leaf that ran a real test votes clone).\n"
                    "calibrated = rule chosen on this run's rl_train questions, reported on held-out.\n\n")
            w.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

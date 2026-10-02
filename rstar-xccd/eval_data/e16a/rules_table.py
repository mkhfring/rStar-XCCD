"""F1 / precision / recall of search runs under every aggregation rule
(paper Tables tab:ablation-dual, tab:aggregation-effect, tab:ablation-ladder search rows).

Rules: majority, clone-on-disagreement (code name "current"), earlier ("old"),
tested non-clone ("tested"). Paper names in brackets.

Usage (../venv-qwen3): python eval_data/e16a/rules_table.py RUN.jsonl [RUN.jsonl ...]
  e.g. eval_data/test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.ablation-dualexec-fix.*.jsonl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_clone_results import predict_label  # noqa: E402
from rescore_runs import RULES  # noqa: E402

RULESET = [("majority", lambda t: predict_label(t, aggregation="majority")),
           ("clone-on-disagreement", RULES["current"]), ("earlier", RULES["old"]),
           ("tested non-clone", RULES["tested"])]


def main():
    for f in sys.argv[1:]:
        rows = [r for r in map(json.loads, open(f)) if "rstar" in r]
        print(f"== {Path(f).name}  (n={len(rows)})")
        for name, fn in RULESET:
            tp = fp = fn_ = 0
            for r in rows:
                p, a = fn(r["rstar"]), r["answer"]
                tp += p == "clone" and a == "clone"
                fp += p == "clone" and a != "clone"
                fn_ += p != "clone" and a == "clone"
            P, R = tp / max(1, tp + fp), tp / max(1, tp + fn_)
            print(f"   {name:22s} F1 {2 * tp / max(1, 2 * tp + fp + fn_):.4f}  P {P:.4f}  R {R:.4f}")


if __name__ == "__main__":
    main()

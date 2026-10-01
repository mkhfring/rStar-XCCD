"""Score the hard-negative pilot (build_hard_negatives.py) per pair kind.

Pair kinds (hard_python_{L}_codenet_meta.jsonl):
  clone      Accepted Python + Accepted Java/Rust, same problem
  hn_output  Accepted Python + Wrong Answer Java/Rust, same problem, outputs differ
             on an official sample input  (the hard, same-task non-clones)
  cross      Accepted Python + Accepted Java/Rust of a different problem (CLCCD-like)

For every system it reports
  clone recall        share of clones predicted clone
  FPR hard / cross    share of hn_output / cross pairs predicted clone
  F1 hard             F1 on clone + hn_output pairs
  F1 cross            F1 on clone + cross pairs (the CLCCD-like setting)
An unanswered pair counts as an error (never "clone" for FPR, a miss for recall).
Rows marked multi_answer_suspect are excluded by default (--keep_suspect to keep).

SCB systems: one per aggregation rule on the same trees (old / current /
majority / tested), plus the triage signal alone ("any leaf ran a real test",
which is what 'tested' reduces to on clone-leaning trees).
Literature-prompt systems: every paper_prompt_replication/hard_python_{L}_codenet.jsonl.*
file, re-parsed with the runner's parser (as summarize_baselines.py).

Usage: python eval_data/e16a/score_hard_negatives.py --lang java [--keep_suspect]
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval_data/e16a"))
from evaluate_clone_results import node_sort_key, predict_label  # noqa: E402
from rescore_runs import RULES, tested  # noqa: E402

E = "eval_data/e16a"
# summarize_baselines.NAME only knows dev/test files; same pattern for hard_*
NAME = re.compile(r"^(?:hard|hardeval)_python_(?P<L>java|rust)_codenet\.jsonl\.(?P<model>.+?)\."
                  r"(?P<prompt>sp1|sp2|sim|reas|inte|sl)\.(?:(?P<variant>think|nothink)(?:\.s\d+)?(?P<n>\.n\d+)?\.)?"
                  r"\d{14}\.jsonl$")


def any_tested(tree):
    numeric = {k: v for k, v in tree.items() if node_sort_key(k) is not None}
    return "clone" if any(tested(numeric, tag) for tag in numeric) else "non-clone"


def report(name, pred, meta):
    k = lambda kind: [m for m in meta if m["kind"] == kind]
    rec = sum(pred.get(m["index"]) == "clone" for m in k("clone")) / max(1, len(k("clone")))
    fpr = lambda kind: sum(pred.get(m["index"]) == "clone" for m in k(kind)) / max(1, len(k(kind)))

    def f1(neg):
        tp = sum(pred.get(m["index"]) == "clone" for m in k("clone"))
        fn = len(k("clone")) - tp
        fp = sum(pred.get(m["index"]) == "clone" for m in k(neg))
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    resp = sum(pred.get(m["index"]) is not None for m in meta) / len(meta)
    print(f"  {name:38s} clone-recall {rec:.3f} | FPR hard {fpr('hn_output'):.3f} cross {fpr('cross'):.3f} | "
          f"F1 hard {f1('hn_output'):.4f} cross {f1('cross'):.4f} | resp {resp:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["java", "rust"])
    ap.add_argument("--keep_suspect", action="store_true")
    ap.add_argument("--set", default="hard", choices=["hard", "hardeval"],
                    help="hard = 250-pair pilot (dev); hardeval = LOCKED evaluation set")
    args = ap.parse_args()
    L = args.lang
    meta = [json.loads(l) for l in open(f"{E}/{args.set}_python_{L}_codenet_meta.jsonl")]
    if not args.keep_suspect:
        meta = [m for m in meta if not m.get("multi_answer_suspect")]
    c = {kd: sum(m["kind"] == kd for m in meta) for kd in ("clone", "hn_output", "cross")}
    print(f"== {L}: {len(meta)} pairs {c}" + ("" if args.keep_suspect else " (multi-answer suspects excluded)"))

    for run in sorted(glob.glob(f"{E}/{args.set}_python_{L}_codenet_depth_16.jsonl.mcts.*.jsonl")):
        trees = {r["index"]: r["rstar"] for r in map(json.loads, open(run)) if "rstar" in r}
        print(f" SCB run {Path(run).name.split('.mcts.')[1][:60]} ({len(trees)} trees)")
        rules = dict(RULES)
        rules["majority"] = lambda t: predict_label(t, aggregation="majority")
        rules["triage only (any real test)"] = any_tested
        for rule, fn in rules.items():
            report(f"SCB {rule}", {i: fn(t) for i, t in trees.items()}, meta)

    files = sorted(glob.glob(f"eval_data/paper_prompt_replication/{args.set}_python_{L}_codenet.jsonl.*.jsonl"))
    if files:
        import summarize_baselines as S
        for f in files:
            m = NAME.match(Path(f).name)
            if not m:
                continue
            var = m["variant"] or "nothink"
            preds = {i: p for i, (_, p) in S.load_preds(f, m["prompt"], var).items()}
            report(f"{m['model']} {m['prompt']} {var}", preds, meta)


if __name__ == "__main__":
    main()

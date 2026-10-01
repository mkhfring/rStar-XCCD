"""Same-compute baseline: self-consistency (SC@k) over the n-sample literature-prompt runs
(run_clccd_paper_prompts.py --n_samples N), compared with SCB on the same pairs.

For each k, the first k samples of every pair are combined with two vote rules:
  majority   clone iff more answered samples say clone than non-clone
  any-clone  clone iff at least one answered sample says clone (= SCB's clone-on-disagreement)
A pair with no answered sample among its k is unanswered (counts as an error in F1 over
all pairs, like the bootstrap scripts). The rule is chosen on the dev file (--dev) and
reported on the test file; SCB tokens/pair (~1,100 generated for Qwen3-4B) sets the
compute-matched k.

Usage:
  python eval_data/e16a/score_self_consistency.py --lang java --dev DEV.jsonl --test TEST.jsonl [--scb SCB_RUN.jsonl]
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load(path):
    return {int(r["index"]): r for r in map(json.loads, open(path)) if "predictions" in r}


def vote(preds, k, rule):
    v = [p for p in (preds or [])[:k] if p]
    if not v:
        return None
    if rule == "any-clone":
        return "clone" if "clone" in v else "non-clone"
    return "clone" if v.count("clone") > v.count("non-clone") else "non-clone"


def f1(gold, pred):
    tp = sum(g == "clone" and p == "clone" for g, p in zip(gold, pred))
    fp = sum(g != "clone" and p == "clone" for g, p in zip(gold, pred))
    fn = sum(g == "clone" and p != "clone" for g, p in zip(gold, pred))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return (2 * P * R / (P + R) if P + R else 0.0), P, R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["java", "rust"])
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--scb", help="SCB run on the test pairs (default: the E5 python-only run)")
    ap.add_argument("--ks", default="1,3,5,7,9,11,13,15")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    dev, test = load(args.dev), load(args.test)

    print(f"== {args.lang}: SC@k F1 over all pairs (P/R) -- dev n={len(dev)}, test n={len(test)}")
    best = {}
    for k in ks:
        row = []
        for rule in ("majority", "any-clone"):
            dv = f1([r["answer"] for r in dev.values()], [vote(r["predictions"], k, rule) for r in dev.values()])
            tv = f1([r["answer"] for r in test.values()], [vote(r["predictions"], k, rule) for r in test.values()])
            row.append(f"{rule} dev {dv[0]:.4f} test {tv[0]:.4f} (P {tv[1]:.3f} R {tv[2]:.3f})")
            if k not in best or dv[0] > best[k][1]:
                best[k] = (rule, dv[0], tv[0])
        print(f"  k={k:2d}  " + " | ".join(row) + f"  -> dev picks {best[k][0]}")

    scb_path = args.scb or glob.glob(
        f"eval_data/test_python_{args.lang}_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.ablation-pyonly.*.jsonl")[0]
    from rescore_runs import predict_tested
    scb = {int(r["index"]): predict_tested(r["rstar"]) for r in map(json.loads, open(scb_path)) if "rstar" in r}
    keys = sorted(set(scb) & set(test))
    gold = [test[i]["answer"] for i in keys]
    a = [scb[i] for i in keys]
    random.seed(0)
    for k in ks:
        rule = best[k][0]
        b = [vote(test[i]["predictions"], k, rule) for i in keys]
        ds = []
        for _ in range(10000):
            ix = [random.randrange(len(keys)) for _ in keys]
            G = [gold[i] for i in ix]
            ds.append(f1(G, [a[i] for i in ix])[0] - f1(G, [b[i] for i in ix])[0])
        ds.sort()
        print(f"  SCB {f1(gold, a)[0]:.4f} vs SC@{k} ({rule}) {f1(gold, b)[0]:.4f}: "
              f"diff {f1(gold, a)[0] - f1(gold, b)[0]:+.4f} 95% CI [{ds[250]:+.4f}, {ds[9750]:+.4f}]")


if __name__ == "__main__":
    main()

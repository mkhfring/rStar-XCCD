"""Paired bootstrap BY PROBLEM for the CLCCD comparisons (completion plan, minimum item 4).

The pair-level bootstrap (summarize_baselines.py) resamples pairs, but pairs that share a
problem are correlated. Here the resampling unit is a cluster:
  - the CodeNet problem id of Code 1 when it was recovered (test_pids_{L}.json; all rust
    pairs, 600 java pairs);
  - otherwise the Python program itself (md5 of Code 1; the 1,200 XLCoST java pairs), so
    pairs that reuse one Python program stay together.
Cross-problem pairs are clustered by Code 1's problem (stated in the paper).

Systems: SCB (python-only, tested rule; seed-0 run, the paper's number) and every
literature-prompt baseline with its dev-chosen prompt (same choice as
summarize_baselines.py: F1 over all dev pairs, unanswered = error). Reports F1 over all
pairs and SCB - baseline with a 95% CI (default 10,000 resamples).

Usage (../venv-qwen3): python eval_data/e16a/bootstrap_by_problem.py [--boot 10000]
"""
import argparse
import glob
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval_data/e16a"))
import summarize_baselines as S  # noqa: E402
from rescore_runs import predict_tested  # noqa: E402


def clusters(L):
    pids = json.load(open(f"eval_data/e16a/test_pids_{L}.json"))
    out = {}
    for r in map(json.loads, open(f"eval_data/test_python_{L}_CLCCD.jsonl")):
        p = pids.get(str(r["index"]))
        if p and p["c1"]:
            out[r["index"]] = "P:" + sorted(p["c1"])[0]
        else:
            code1 = r["question"].split("```python\n", 1)[1].split("```", 1)[0]
            out[r["index"]] = "H:" + hashlib.md5(code1.encode()).hexdigest()
    return out


def baselines(L):
    runs = defaultdict(lambda: defaultdict(dict))
    for f in sorted(glob.glob(f"{S.D}/*.jsonl")):
        m = S.NAME.match(Path(f).name)
        if not m or m["n"] or m["L"] != L:
            continue
        split = "dev" if m["stem"].startswith("dev") else "test"
        runs[(m["model"], m["variant"] or "nothink")][split][m["prompt"]] = f
    out = {}
    for (model, var), sp in sorted(runs.items()):
        if "test" not in sp or "dev" not in sp:
            continue
        devf = {p: S.f1_all(*zip(*S.load_preds(f, p, var).values())) for p, f in sp["dev"].items()}
        prompt = max(devf, key=devf.get)
        if prompt in sp["test"]:
            out[f"{model} {prompt} {var}"] = {i: q for i, (_, q) in S.load_preds(sp["test"][prompt], prompt, var).items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    for L in ("java", "rust"):
        scb_file = sorted(glob.glob(f"eval_data/test_python_{L}_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.ablation-pyonly.*.jsonl"))[0]
        scb, gold = {}, {}
        for r in map(json.loads, open(scb_file)):
            if "rstar" in r:
                scb[r["index"]] = predict_tested(r["rstar"])
                gold[r["index"]] = r["answer"]
        cl = clusters(L)
        by = defaultdict(list)
        for i in gold:
            by[cl[i]].append(i)
        keys = sorted(by)
        base = baselines(L)
        f1 = lambda pred, ix: S.f1_all([gold[i] for i in ix], [pred.get(i) for i in ix])
        allix = sorted(gold)
        print(f"== {L}: {len(allix)} pairs in {len(keys)} clusters; SCB F1 {f1(scb, allix):.4f}")
        print(f"   {'baseline':40s} {'F1':>6s} | SCB-base  95% CI (by problem)   [pair-level CI for reference]")
        random.seed(0)
        samples = [[i for _ in keys for i in by[random.choice(keys)]] for _ in range(args.boot)]
        for name, pred in base.items():
            ds = sorted(f1(scb, ix) - f1(pred, ix) for ix in samples)
            lo, hi = ds[int(.025 * args.boot)], ds[int(.975 * args.boot) - 1]
            print(f"   {name:40s} {f1(pred, allix):6.4f} | {f1(scb, allix) - f1(pred, allix):+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()

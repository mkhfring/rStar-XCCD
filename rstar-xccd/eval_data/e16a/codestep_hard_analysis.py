"""Code-step v2 on the DEV hard-negative pilot (hard_python_{L}_codenet, not the locked hardeval set).
Rules fixed BEFORE the code-step output was read.

Systems, all on the same pairs (multi_answer_suspect rows excluded, as score_hard_negatives.py):
  SCB (tested rule)              main tree, branch hard-pyonly
  Extension it8+v3 (current)     main tree, branch stepb-hard-ext-it8-v3 (clone-on-disagreement)
  Code-step v2 / majority        codestep worktree, branch codestep-v2-hard (primary rule, as in the CLCCD pilot)
  Code-step v2 / current         same trees, clone-on-disagreement
Per system: clone recall, bug->clone (hn_output predicted clone), cross->clone, F1 / balanced accuracy
on clone + hn_output. Paired bootstrap BY PROBLEM (Python problem id) of BA, code-step(majority) minus
each other system. Unanswered pairs count as errors (never clone).

Usage (../venv-qwen3, codestep worktree): python eval_data/e16a/codestep_hard_analysis.py [--boot 5000]
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval_data/e16a"))
from evaluate_clone_results import predict_label  # noqa: E402
from rescore_runs import RULES  # noqa: E402
import codestep_pilot_analysis as A  # noqa: E402

MAIN = Path("/lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd/eval_data/e16a")
HERE = "eval_data/e16a/codestep_hard"


def trees(pattern):
    f = sorted(glob.glob(str(pattern)))
    return {r["index"]: r["rstar"] for r in map(json.loads, open(f[-1])) if "rstar" in r} if f else {}


def metrics(meta, p, ix):
    k = lambda i: meta[i]["kind"]  # noqa: E731
    cl = [i for i in ix if k(i) == "clone"]; hn = [i for i in ix if k(i) == "hn_output"]; cr = [i for i in ix if k(i) == "cross"]
    tp = sum(p.get(i) == "clone" for i in cl); fp = sum(p.get(i) == "clone" for i in hn)
    rec = tp / len(cl) if cl else 0.0
    b2c = fp / len(hn) if hn else 0.0
    c2c = sum(p.get(i) == "clone" for i in cr) / len(cr) if cr else 0.0
    f1 = 2 * tp / (2 * tp + fp + len(cl) - tp) if tp else 0.0
    return dict(rec=rec, b2c=b2c, c2c=c2c, F1=f1, BA=(rec + 1 - b2c) / 2, n=(len(cl), len(hn), len(cr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    for L in ("java", "rust"):
        meta = {r["index"]: r for r in map(json.loads, open(f"{HERE}/hard_python_{L}_codenet_meta.jsonl"))}
        cs = trees(f"{HERE}/hard_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.codestep-v2-hard.*.jsonl")
        ex = trees(MAIN / f"hard_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.stepb-hard-ext-it8-v3.*.jsonl")
        scb = trees(MAIN / f"hard_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.hard-pyonly.*.jsonl")
        ix = sorted(i for i in cs if i in ex and i in scb and not meta[i].get("multi_answer_suspect"))
        print(f"===== {L}: code-step {len(cs)}/{len(meta)} trees; compared on {len(ix)} pairs")
        if not ix:
            continue
        sysm = {
            "SCB (tested rule)": {i: RULES["tested"](scb[i]) for i in ix},
            "Extension it8+v3 (current)": {i: predict_label(ex[i]) for i in ix},
            "Code-step v2 / majority": {i: A.rules(cs[i])["majority"] for i in ix},
            "Code-step v2 / current": {i: A.rules(cs[i])["current"] for i in ix},
        }
        for name, p in sysm.items():
            m = metrics(meta, p, ix)
            print(f"  {name:28s} clone-rec {m['rec']:.3f} | bug->clone {m['b2c']:.3f} | cross->clone {m['c2c']:.3f} | F1 {m['F1']:.3f} BA {m['BA']:.3f}  (n clone/hn/cross {m['n']})")
        groups = {}
        for i in ix:
            groups.setdefault(meta[i].get("p_id1"), []).append(i)
        keys = sorted(groups)
        rng = random.Random(0)
        base = sysm["Code-step v2 / majority"]
        for other in ("SCB (tested rule)", "Extension it8+v3 (current)"):
            d = []
            for _ in range(a.boot):
                s = [i for g in (rng.choice(keys) for _ in keys) for i in groups[g]]
                d.append(metrics(meta, base, s)["BA"] - metrics(meta, sysm[other], s)["BA"])
            d.sort()
            obs = metrics(meta, base, ix)["BA"] - metrics(meta, sysm[other], ix)["BA"]
            print(f"  BA code-step(majority) - {other}: {obs:+.3f}  95% CI [{d[int(.025 * len(d))]:+.3f}, {d[int(.975 * len(d))]:+.3f}] (by problem, {len(keys)} problems)")
        gold = {i: meta[i]["answer"] for i in ix}
        b = A.behaviour({i: cs[i] for i in ix}, gold)
        P = max(b["paths"], 1)
        print(f"  behaviour: {b['paths']} paths, {b['steps'] / P:.1f} steps/path, {b['runboth_calls'] / P:.1f} run_both/path, "
              f"answer!=printed {b['paths_answer_ne_printed']}, diff-but-clone {b['paths_diff_but_clone']}, errors {b['steps_error']}/{b['steps']}")


if __name__ == "__main__":
    main()

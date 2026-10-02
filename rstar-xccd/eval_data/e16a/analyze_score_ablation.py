"""Why do the search scores not matter? (plan section 00b, analysis items 1-6)

Reads the score-ablation dev runs (scoreabl-it{4,8}-{on,off}) and the iteration-2 dev runs
(e1-dev-pyonly) and reports per arm:
  1. tree shape: nodes, leaves (final answers), root visits, distinct root children
     expanded, max depth; and how far ON and OFF trees differ for the same pair
  2. score informativeness: among internal nodes with >= 2 expanded children, the share
     whose children have DIFFERENT q-values (only then can PUCT prefer one branch);
     distribution of internal q-values
  3. tested-rule flips: non-clone pairs predicted clone because a tested trajectory
     voted non-clone (the precision loss), and the number of tested leaves per tree
  4. F1 under every aggregation rule (does the score effect show under verdict rules?)
Usage (../venv-qwen3): python eval_data/e16a/analyze_score_ablation.py
"""
import glob
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_clone_results import node_sort_key, normalize_label, predict_label  # noqa: E402
from rescore_runs import RULES, tested  # noqa: E402

E = "eval_data/e16a"
ARMS = ["e1-dev-pyonly", "scoreabl-it4-on", "scoreabl-it4-off", "scoreabl-it8-on", "scoreabl-it8-off"]


def load(L, arm):
    f = sorted(glob.glob(f"{E}/dev_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.{arm}.*.jsonl"))[0]
    return {r["index"]: r for r in map(json.loads, open(f)) if "rstar" in r}


def numeric(t):
    return {k: v for k, v in t.items() if node_sort_key(k) is not None and isinstance(v, dict)}


def shape(t):
    n = numeric(t)
    leaves = [k for k, v in n.items() if (v.get("final_answer") or "").strip()]
    kids = Counter(k.rsplit(".", 1)[0] for k in n if "." in k)
    root_children = sum(1 for k in n if k.count(".") == 1)
    diff = same = 0
    qs = []
    for p, c in kids.items():
        ch = [v for k, v in n.items() if k.rsplit(".", 1)[0] == p and "." in k and not (v.get("final_answer") or "").strip()]
        q = [float(v.get("q_value") or 0) for v in ch]
        if len(q) >= 2:
            if len(set(round(x, 6) for x in q)) > 1:
                diff += 1
            else:
                same += 1
    for k, v in n.items():
        if not (v.get("final_answer") or "").strip() and k != "0":
            qs.append(float(v.get("q_value") or 0))
    depth = max(k.count(".") for k in n)
    rootv = int(n.get("0", {}).get("visit_count") or 0)
    tested_leaves = sum(1 for k in leaves if tested(n, k))
    return dict(nodes=len(n), leaves=len(leaves), root_children=root_children, depth=depth, rootv=rootv,
                sib_diff=diff, sib_same=same, qs=qs, tested_leaves=tested_leaves)


def f1(pairs):
    tp = sum(a == "clone" and p == "clone" for a, p in pairs)
    fp = sum(a != "clone" and p == "clone" for a, p in pairs)
    fn = sum(a == "clone" and p != "clone" for a, p in pairs)
    return 2 * tp / max(1, 2 * tp + fp + fn), tp / max(1, tp + fp), tp / max(1, tp + fn)


def main():
    rules = {"tested": RULES["tested"], "clone-on-dis": RULES["current"], "majority": lambda t: predict_label(t, aggregation="majority")}
    for L in ("java", "rust"):
        runs = {a: load(L, a) for a in ARMS}
        ix = sorted(set.intersection(*(set(r) for r in runs.values())))
        print(f"\n===== {L}: {len(ix)} pairs in every arm")
        print(f"{'arm':18s} {'nodes':>6s} {'leaves':>6s} {'rootV':>6s} {'rootCh':>6s} {'depth':>5s} | "
              f"{'sibDiff%':>8s} {'q!=1 %':>7s} | {'testedLeaves':>12s} {'nonclone flips':>14s} | F1 tested / clone-on-dis / majority")
        for a in ARMS:
            S = [shape(runs[a][i]["rstar"]) for i in ix]
            sd = sum(s["sib_diff"] for s in S); ss = sum(s["sib_same"] for s in S)
            qs = [q for s in S for q in s["qs"]]
            flips = 0
            for i in ix:
                r = runs[a][i]
                if r["answer"] != "clone" and RULES["tested"](r["rstar"]) == "clone" and RULES["current"](r["rstar"]) != "clone":
                    flips += 1
            fs = {n: f1([(runs[a][i]["answer"], fn(runs[a][i]["rstar"])) for i in ix]) for n, fn in rules.items()}
            print(f"{a:18s} {st.mean(s['nodes'] for s in S):6.1f} {st.mean(s['leaves'] for s in S):6.2f} "
                  f"{st.mean(s['rootv'] for s in S):6.2f} {st.mean(s['root_children'] for s in S):6.2f} "
                  f"{st.mean(s['depth'] for s in S):5.1f} | {100 * sd / max(1, sd + ss):7.1f}% "
                  f"{100 * sum(abs(q - 1.0) > 1e-6 for q in qs) / max(1, len(qs)):6.1f}% | "
                  f"{st.mean(s['tested_leaves'] for s in S):12.2f} {flips:14d} | "
                  + " / ".join(f"{fs[n][0]:.3f} (P{fs[n][1]:.2f})" for n in rules))
        for it in (4, 8):
            on, off = runs[f"scoreabl-it{it}-on"], runs[f"scoreabl-it{it}-off"]
            same_shape = sum(sorted(numeric(on[i]["rstar"])) == sorted(numeric(off[i]["rstar"])) for i in ix)
            same_pred = sum(RULES["tested"](on[i]["rstar"]) == RULES["tested"](off[i]["rstar"]) for i in ix)
            print(f"  it{it}: ON vs OFF identical tree shape (same node ids) in {same_shape}/{len(ix)} pairs; "
                  f"same tested prediction in {same_pred}/{len(ix)}")


if __name__ == "__main__":
    main()

"""Step A of the improved-scoring plan (plan section 00b): OFFLINE re-aggregation of existing
search trees with evidence-aware, label-free rules. DEVELOPMENT DATA ONLY (no locked set, no
CLCCD test), so nothing here is tuned on evaluation data.

Per trajectory (root -> answer leaf) we record its vote and its evidence, using
diagnose_hard_negatives.classify on the path text:
  real      Code 1 ran on an input and printed output ("ran")
  harness   the harness compared Code 1 and Code 2 on the step's inputs (extension runs)
            -> SAME / DIFFERENT (clean) / INCONCLUSIVE / BOTH FAILED lines
  q         mean q-value of the path's internal nodes (the search scores)
Rules (all label-free):
  clone-on-dis       existing default (any clone vote -> clone, with the existing fallbacks)
  tested             existing SCB rule
  tested-real        like tested, but only trajectories with REAL evidence count as tested
  evidence-filtered  clone-on-disagreement over trajectories with real evidence only;
                     falls back to clone-on-dis if no trajectory has evidence
  value-weighted     sum of q over clone votes vs non-clone votes (ties -> clone-on-dis)
  harness            extension only: any clean DIFFERENT -> non-clone; else any SAME -> clone;
                     else clone-on-dis
Reports F1 (P/R) on the CLCCD-style dev set, and clone recall / bug->clone / BA on the
hard-negative DEV pilot.
Usage (../venv-qwen3): python eval_data/e16a/reaggregate_dev.py
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval_data/e16a"))
import diagnose_hard_negatives as D  # noqa: E402
from evaluate_clone_results import node_sort_key, normalize_label  # noqa: E402
from rescore_runs import RULES  # noqa: E402

E = "eval_data/e16a"


def trajectories(tree):
    n = {k: v for k, v in tree.items() if node_sort_key(k) is not None and isinstance(v, dict)}
    out = []
    for tag, v in n.items():
        fa = (v.get("final_answer") or "").strip()
        lab = normalize_label(fa) if fa else None
        if lab is None:
            continue
        parts = tag.split(".")
        chain = [n[".".join(parts[:i])] for i in range(1, len(parts) + 1) if ".".join(parts[:i]) in n]
        text = "\n".join(c.get("text", "") for c in chain)
        cat, flags = D.classify(text, "")
        lines = [l for l in text.splitlines() if l.startswith("Input ") and " -> Code 1: " in l]
        qs = [float(c.get("q_value") or 0) for c in chain[1:-1]] or [0.0]
        out.append(dict(vote=lab, real=(cat == "ran"),
                        h_diff=any(l.endswith("-> DIFFERENT") for l in lines),
                        h_same=any(l.endswith("-> SAME") for l in lines),
                        q=sum(qs) / len(qs)))
    return out


def rules(tree):
    T = trajectories(tree)
    cod = RULES["current"](tree)
    real = [t for t in T if t["real"]]
    ev = ("clone" if any(t["vote"] == "clone" for t in real) else "non-clone") if real else cod
    # tested-real: a non-clone vote from a trajectory with REAL evidence flips to clone
    tr_votes = ["clone" if (t["vote"] == "non-clone" and t["real"]) else t["vote"] for t in T]
    tested_real = ("clone" if "clone" in tr_votes else "non-clone") if tr_votes else cod
    sc = sum(t["q"] for t in T if t["vote"] == "clone")
    sn = sum(t["q"] for t in T if t["vote"] == "non-clone")
    vw = cod if abs(sc - sn) < 1e-9 else ("clone" if sc > sn else "non-clone")
    harness = "non-clone" if any(t["h_diff"] for t in T) else ("clone" if any(t["h_same"] for t in T) else cod)
    return {"clone-on-dis": cod, "tested": RULES["tested"](tree), "tested-real": tested_real,
            "evidence-filtered": ev, "value-weighted": vw, "harness": harness}


def f1(pairs):
    tp = sum(a == "clone" and p == "clone" for a, p in pairs)
    fp = sum(a != "clone" and p == "clone" for a, p in pairs)
    fn = sum(a == "clone" and p != "clone" for a, p in pairs)
    return 2 * tp / max(1, 2 * tp + fp + fn), tp / max(1, tp + fp), tp / max(1, tp + fn)


def main():
    names = ["clone-on-dis", "tested", "tested-real", "evidence-filtered", "value-weighted", "harness"]
    print("# CLCCD-style DEV set (cross-problem negatives): F1 (P / R)")
    for L in ("java", "rust"):
        for arm in ("e1-dev-pyonly", "scoreabl-it4-on", "scoreabl-it4-off", "scoreabl-it8-on", "scoreabl-it8-off"):
            f = sorted(glob.glob(f"{E}/dev_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.{arm}.*.jsonl"))[0]
            rows = [r for r in map(json.loads, open(f)) if "rstar" in r]
            preds = [(r["answer"], rules(r["rstar"])) for r in rows]
            print(f"  {L:4s} {arm:18s} " + " | ".join(
                f"{n} {f1([(a, p[n]) for a, p in preds])[0]:.3f}" for n in names if n != "harness"))
    print("\n# hard-negative DEV pilot (same-task bugs): clone recall / bug->clone / BA")
    for L, arms in (("java", ["hard-pyonly", "hard-autocode2-javav3"]), ("rust", ["hard-pyonly", "hard-autocode2-rustv4"])):
        meta = {m["index"]: m for m in map(json.loads, open(f"{E}/hard_python_{L}_codenet_meta.jsonl"))
                if not m.get("multi_answer_suspect")}
        for arm in arms:
            f = sorted(glob.glob(f"{E}/hard_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.{arm}.*.jsonl"))[-1]
            rows = {r["index"]: r for r in map(json.loads, open(f)) if "rstar" in r and r["index"] in meta}
            cl = [i for i in rows if meta[i]["kind"] == "clone"]
            hn = [i for i in rows if meta[i]["kind"] == "hn_output"]
            P = {i: rules(rows[i]["rstar"]) for i in rows}
            out = []
            for n in names:
                if n == "harness" and "autocode2" not in arm:
                    continue
                rec = sum(P[i][n] == "clone" for i in cl) / len(cl)
                fpr = sum(P[i][n] == "clone" for i in hn) / len(hn)
                out.append(f"{n} {rec:.2f}/{fpr:.2f}/{(rec + 1 - fpr) / 2:.3f}")
            print(f"  {L:4s} {arm:22s} " + " | ".join(out))


if __name__ == "__main__":
    main()

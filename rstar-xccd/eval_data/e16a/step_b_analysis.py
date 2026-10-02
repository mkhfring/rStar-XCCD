"""Step B analysis (plan section 00b): reward v3 vs v1, search budget, and the extension
(both programs executed). DEVELOPMENT DATA ONLY.

Sections
 1. CLCCD-style dev set (300 pairs/lang): F1 (P/R) per aggregation rule per arm
 2. hard-negative dev pilot (250 pairs/lang, multi-answer suspects excluded):
    clone recall / bug->clone / balanced accuracy per rule per arm
 3. paired bootstrap v3 - v1 at equal budget (main rule of each system:
    tested for python-only SCB, clone-on-disagreement for the extension)
 4. score diagnostics: share of branching points whose children have different values,
    and on TRUE CLONES the mean path value of correct "clone" votes vs wrong untested
    "non-clone" votes (v1 inverted these)
Usage (../venv-qwen3): python eval_data/e16a/step_b_analysis.py
"""
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval_data/e16a"))
from analyze_score_ablation import shape  # noqa: E402
from reaggregate_dev import rules, trajectories  # noqa: E402

E = "eval_data/e16a"
RULE_NAMES = ["tested", "clone-on-dis", "tested-real", "harness", "value-weighted"]


def load(ds, L, branch):
    fs = sorted(glob.glob(f"{E}/{ds}_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.{branch}.*.jsonl"))
    if not fs:
        return None
    return {r["index"]: r for r in map(json.loads, open(fs[-1])) if "rstar" in r}


def f1(pairs):
    tp = sum(a == "clone" and p == "clone" for a, p in pairs)
    fp = sum(a != "clone" and p == "clone" for a, p in pairs)
    fn = sum(a == "clone" and p != "clone" for a, p in pairs)
    return 2 * tp / max(1, 2 * tp + fp + fn), tp / max(1, tp + fp), tp / max(1, tp + fn)


DEV_ARMS = [("SCB py-only it2 v1", "e1-dev-pyonly"), ("SCB py-only it4 v1", "scoreabl-it4-on"),
            ("SCB py-only it4 v3", "stepb-dev-pyonly-it4-v3"), ("SCB py-only it8 v1", "scoreabl-it8-on"),
            ("SCB py-only it8 v3", "stepb-dev-pyonly-it8-v3"), ("EXT it2 v1", "stepb-dev-ext-it2-v1"),
            ("EXT it4 v1", "stepb-dev-ext-it4-v1"), ("EXT it4 v3", "stepb-dev-ext-it4-v3"),
            ("EXT it8 v1", "stepb-dev-ext-it8-v1"), ("EXT it8 v3", "stepb-dev-ext-it8-v3")]


def hard_arms(L):
    return [("SCB py-only it2 v1", "hard-pyonly"),
            ("EXT it2 v1", "hard-autocode2-javav3" if L == "java" else "hard-autocode2-rustv4"),
            ("EXT it4 v1", "stepb-hard-ext-it4-v1"), ("EXT it4 v3", "stepb-hard-ext-it4-v3"),
            ("EXT it8 v1", "stepb-hard-ext-it8-v1"), ("EXT it8 v3", "stepb-hard-ext-it8-v3")]


def boot(gold, pa, pb, ix, B=5000, score=None):
    random.seed(0)
    d = sorted(score(gold, pa, s) - score(gold, pb, s) for s in ([random.choice(ix) for _ in ix] for _ in range(B)))
    return d[int(.025 * B)], d[int(.975 * B) - 1]


def main():
    print("## 1. CLCCD-style DEV set: F1 (P/R) per rule")
    dev_pred = {}
    for L in ("java", "rust"):
        for name, br in DEV_ARMS:
            R = load("dev", L, br)
            P = {i: rules(R[i]["rstar"]) for i in R}
            dev_pred[(L, name)] = (R, P)
            cells = []
            for rn in RULE_NAMES:
                if rn == "harness" and not name.startswith("EXT"):
                    continue
                F, p, r = f1([(R[i]["answer"], P[i][rn]) for i in R])
                cells.append(f"{rn} {F:.3f} ({p:.2f}/{r:.2f})")
            print(f"  {L:4s} {name:20s} " + " | ".join(cells))

    print("\n## 2. hard-negative DEV pilot: clone recall / bug->clone / BA per rule")
    hard_pred = {}
    for L in ("java", "rust"):
        meta = {m["index"]: m for m in map(json.loads, open(f"{E}/hard_python_{L}_codenet_meta.jsonl"))
                if not m.get("multi_answer_suspect")}
        for name, br in hard_arms(L):
            R = {i: r for i, r in (load("hard", L, br) or {}).items() if i in meta}
            P = {i: rules(R[i]["rstar"]) for i in R}
            hard_pred[(L, name)] = (R, P, meta)
            cl = [i for i in R if meta[i]["kind"] == "clone"]
            hn = [i for i in R if meta[i]["kind"] == "hn_output"]
            cells = []
            for rn in RULE_NAMES:
                if rn == "harness" and not name.startswith("EXT"):
                    continue
                rec = sum(P[i][rn] == "clone" for i in cl) / len(cl)
                fpr = sum(P[i][rn] == "clone" for i in hn) / len(hn)
                cells.append(f"{rn} {rec:.2f}/{fpr:.2f}/{(rec + 1 - fpr) / 2:.3f}")
            print(f"  {L:4s} {name:20s} " + " | ".join(cells))

    print("\n## 3. paired bootstrap v3 - v1 at equal budget (5000 resamples)")
    def f1_score(gold, P, ix):
        return f1([(gold[i], P[i]) for i in ix])[0]

    def ba_score(gold, P, ix):
        cl = [i for i in ix if gold[i] == "clone"]
        hn = [i for i in ix if gold[i] == "hn_output"]
        rec = sum(P[i] == "clone" for i in cl) / max(1, len(cl))
        spec = sum(P[i] != "clone" for i in hn) / max(1, len(hn))
        return (rec + spec) / 2
    for L in ("java", "rust"):
        for it in (4, 8):
            for sysname, rn in (("SCB py-only", "tested"), ("EXT", "clone-on-dis"), ("EXT", "harness")):
                a, b = dev_pred[(L, f"{sysname} it{it} v3")], dev_pred[(L, f"{sysname} it{it} v1")]
                ix = sorted(set(a[0]) & set(b[0]))
                gold = {i: a[0][i]["answer"] for i in ix}
                pa = {i: a[1][i][rn] for i in ix}
                pb = {i: b[1][i][rn] for i in ix}
                lo, hi = boot(gold, pa, pb, ix, score=f1_score)
                d = f1_score(gold, pa, ix) - f1_score(gold, pb, ix)
                print(f"  dev  {L:4s} {sysname:11s} it{it} {rn:12s} F1 v3-v1 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            for rn in ("clone-on-dis", "harness"):
                a, b = hard_pred[(L, f"EXT it{it} v3")], hard_pred[(L, f"EXT it{it} v1")]
                meta = a[2]
                ix = sorted(i for i in set(a[0]) & set(b[0]) if meta[i]["kind"] in ("clone", "hn_output"))
                gold = {i: meta[i]["kind"] for i in ix}
                pa = {i: a[1][i][rn] for i in ix}
                pb = {i: b[1][i][rn] for i in ix}
                lo, hi = boot(gold, pa, pb, ix, score=ba_score)
                d = ba_score(gold, pa, ix) - ba_score(gold, pb, ix)
                print(f"  hard {L:4s} EXT         it{it} {rn:12s} BA v3-v1 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n## 4. score diagnostics (CLCCD-style dev): sibling-value differences; on TRUE CLONES "
          "mean path value of correct clone votes vs wrong untested non-clone votes")
    for L in ("java", "rust"):
        for name in ("SCB py-only it8 v1", "SCB py-only it8 v3", "EXT it8 v1", "EXT it8 v3"):
            R, _ = dev_pred[(L, name)]
            S = [shape(R[i]["rstar"]) for i in R]
            sd = sum(s["sib_diff"] for s in S); ss = sum(s["sib_same"] for s in S)
            good, bad = [], []
            for r in R.values():
                if r["answer"] != "clone":
                    continue
                for t in trajectories(r["rstar"]):
                    if t["vote"] == "clone":
                        good.append(t["q"])
                    elif not t["real"]:
                        bad.append(t["q"])
            print(f"  {L:4s} {name:20s} siblings differ {100 * sd / max(1, sd + ss):5.1f}% | correct clone votes "
                  f"q {st.mean(good) if good else float('nan'):+.2f} (n={len(good)}) vs wrong untested non-clone q "
                  f"{st.mean(bad) if bad else float('nan'):+.2f} (n={len(bad)})")


if __name__ == "__main__":
    main()

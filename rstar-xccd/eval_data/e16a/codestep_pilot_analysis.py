"""Code-step v1 pilot analysis (branch codestep-prompt). Rules fixed BEFORE the pilot output was read.

Accuracy on the 200-pair CLCCD pilot sample, code-step vs the frozen extension (clccd-ext-it8v3,
full-set run in the main tree, restricted to the same indices; clone-on-disagreement rule):
  code-step / majority      majority vote of the leaves' final labels (tie -> clone)
  code-step / current       evaluate_clone_results.predict_label (clone-on-disagreement)
  code-step / consistent    majority over leaves that pass the v4 leaf check (score_leaf_v4 > 0
                            or None, i.e. not contradicted); falls back to majority
Paired bootstrap (by pair) of F1 difference code-step(majority) minus extension.

Behaviour of the trees (is the model doing what the prompt asks?):
  steps per leaf path, run_both calls per path, share of steps with a Python error / no valid code,
  one-sided failures and retries, paths that observed a difference but answered clone,
  answers that differ from the printed label, Code 2 build failure (and what the model did then).

Usage (../venv-qwen3, from the codestep worktree): python eval_data/e16a/codestep_pilot_analysis.py [--boot 5000]
"""
import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_clone_results import node_sort_key, normalize_label, predict_label  # noqa: E402
from rstar_deepthink.agents.step_scoring import printed_label, runboth_counts, score_code_step_v4, score_leaf_v4  # noqa: E402

MAIN = Path("/lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd")
PILOT = "eval_data/e16a/codestep_pilot"


def load(pattern):
    f = sorted(glob.glob(pattern))
    if not f:
        return None, {}
    return f[-1], {r["index"]: r["rstar"] for r in map(json.loads, open(f[-1])) if "rstar" in r}


def leaves(tree):
    """(tag, node, ancestor code states root->leaf) for every node with a final answer."""
    num = {k: v for k, v in tree.items() if node_sort_key(k) is not None}
    out = []
    for tag, n in num.items():
        lab = normalize_label((n.get("final_answer") or "").strip()) if n.get("final_answer") else None
        if lab is None:
            continue
        chain, t = [], tag
        while t in num:
            chain.append(num[t])
            if "." not in t:
                break
            t = t.rsplit(".", 1)[0]
        chain = [s for s in reversed(chain) if s.get("action") == "python_interpreter"]
        out.append((tag, n, lab, chain))
    return out


def vote(labels):
    if not labels:
        return None
    c = collections.Counter(labels)
    return "clone" if c["clone"] >= c["non-clone"] else "non-clone"


def rules(tree):
    lv = leaves(tree)
    maj = vote([l for _, _, l, _ in lv]) or predict_label(tree)
    cons = vote([l for _, n, l, ch in lv if (score_leaf_v4(n.get("final_answer"), ch, 1.0, -1.0) or 0) >= 0]) or maj
    return {"majority": maj, "current": predict_label(tree), "consistent": cons}


def prf(gold, pred, ix):
    tp = sum(gold[i] == "clone" and pred.get(i) == "clone" for i in ix)
    fp = sum(gold[i] != "clone" and pred.get(i) == "clone" for i in ix)
    fn = sum(gold[i] == "clone" and pred.get(i) != "clone" for i in ix)
    tn = len(ix) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(F1=f, P=p, R=r, acc=(tp + tn) / len(ix) if ix else 0.0, tp=tp, fp=fp, fn=fn, tn=tn)


def behaviour(trees, gold):
    s = collections.Counter()
    for i, tree in trees.items():
        for _, n, lab, ch in leaves(tree):
            s["paths"] += 1
            s["steps"] += len(ch)
            obs = [c.get("observation") or "" for c in ch]
            b = [runboth_counts(o) for o in obs]
            calls = sum(x[0] for x in b if x)
            s["runboth_calls"] += calls
            s["paths_no_call"] += calls == 0
            s["steps_error"] += sum(bool(score_code_step_v4(o, "", 1.0, -1.0)[2]) for o in obs)
            s["steps_onesided"] += sum(1 for x in b if x and x[2] > 0)
            diff = any(x and x[4] > 0 for x in b)
            s["paths_diff_but_clone"] += diff and lab == "clone"
            pl = printed_label(obs[-1]) if obs else None
            s["paths_no_conclusion"] += pl is None
            s["paths_answer_ne_printed"] += pl is not None and pl != lab
            nobuild = any("could not be built" in o for o in obs)
            s["paths_code2_unbuilt"] += nobuild
            if nobuild:
                s[f"unbuilt_answer_{lab}_gold_{gold[i]}"] += 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    for L in ("java", "rust"):
        gold = {r["index"]: r["answer"] for r in map(json.loads, open(f"{PILOT}/pilot_python_{L}_CLCCD.jsonl"))}
        f_cs, cs = load(f"{PILOT}/pilot_python_{L}_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.codestep-v1-pilot.*.jsonl")
        f_ex, ex = load(str(MAIN / f"eval_data/test_python_{L}_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.clccd-ext-it8v3.*.jsonl"))
        print(f"===== {L}: code-step {len(cs)}/{len(gold)} trees ({f_cs}); extension {sum(i in ex for i in gold)}/{len(gold)} on the same indices")
        ix = sorted(i for i in gold if i in cs and i in ex)
        if not ix:
            continue
        nc = sum(gold[i] == "clone" for i in ix)
        print(f"compared on {len(ix)} pairs ({nc} clone / {len(ix) - nc} non-clone)")
        pc = {i: rules(cs[i]) for i in ix}
        pe = {i: predict_label(ex[i]) for i in ix}
        rows = [("extension (clone-on-disagreement)", pe)] + [(f"code-step / {r}", {i: pc[i][r] for i in ix}) for r in ("majority", "current", "consistent")]
        for name, p in rows:
            m = prf(gold, p, ix)
            print(f"  {name:36s} F1 {m['F1']:.3f}  P {m['P']:.3f}  R {m['R']:.3f}  acc {m['acc']:.3f}  tp {m['tp']} fp {m['fp']} fn {m['fn']} tn {m['tn']}")
        rng = random.Random(0)
        pm = rows[1][1]
        d = []
        for _ in range(a.boot):
            s = [rng.choice(ix) for _ in ix]
            d.append(prf(gold, pm, s)["F1"] - prf(gold, pe, s)["F1"])
        d.sort()
        print(f"  F1 code-step(majority) - extension: {prf(gold, pm, ix)['F1'] - prf(gold, pe, ix)['F1']:+.3f}  95% CI [{d[int(.025 * len(d))]:+.3f}, {d[int(.975 * len(d))]:+.3f}]")
        b = behaviour({i: cs[i] for i in ix}, gold)
        P = max(b["paths"], 1)
        print(f"  behaviour: {b['paths']} leaf paths, {b['steps'] / P:.1f} code steps/path, {b['runboth_calls'] / P:.1f} run_both calls/path")
        for k in ("paths_no_call", "paths_no_conclusion", "paths_answer_ne_printed", "paths_diff_but_clone", "paths_code2_unbuilt"):
            print(f"    {k:28s} {b[k]:5d} ({b[k] / P:.0%} of paths)")
        print(f"    steps with Python error / no code {b['steps_error']} of {b['steps']}; steps with a one-sided failure {b['steps_onesided']}")
        for k in sorted(k for k in b if k.startswith("unbuilt_answer")):
            print(f"    {k:40s} {b[k]}")


if __name__ == "__main__":
    main()

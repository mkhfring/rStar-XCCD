"""Metrics for the LOCKED same-task hard-negative sets (paper Table tab:hard-negatives, RQ3/RQ4).

For each system on hardeval_python_{L}_codenet.jsonl (clone / hn_output / cross pairs):
  clone recall, bug->clone (share of hard non-clones labeled clone, 95% Wilson interval),
  cross->clone, and F1 / balanced accuracy / MCC on the clone + hard non-clone pairs.
Systems (rules fixed BEFORE the locked sets were used):
  SCB            python-only search trees, tested non-clone rule   (branch hard-locked-pyonly)
  Extension      harness-side dual execution, clone-on-disagreement (branch hard-locked-ext)
  Qwen3-4B think / phi-4: literature prompt sp2 (dev-chosen), --split hardeval
With --bootstrap: paired bootstrap by PROBLEM (Extension minus each system) for F1, BA, MCC.
--set polyhuman scores the EXTERNAL PolyHuman set (polyhuman_python_java.jsonl, build_polyhuman_set.py; java
only) with the same frozen systems: branches polyhuman-pyonly / polyhuman-ext-it8v3, baselines run with
--datafile eval_data/e16a/polyhuman_python_{L}.jsonl.

Usage (../venv-qwen3): python eval_data/e16a/hardeval_metrics.py [--set hardeval|polyhuman] [--bootstrap 5000]
"""
import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval_data/e16a"))
import summarize_baselines as S  # noqa: E402
from rescore_runs import RULES  # noqa: E402

E = "eval_data/e16a"
SETS = {  # stem, SCB branch, extension branch (frozen it8 + v3), languages
    "hardeval": ("hardeval_python_{L}_codenet", "hard-locked-pyonly", "hard-locked-ext-it8v3", ("java", "rust")),
    "polyhuman": ("polyhuman_python_{L}", "polyhuman-pyonly", "polyhuman-ext-it8v3", ("java",)),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def systems(L, which="hardeval"):
    stem, scb, ext, _ = SETS[which]
    stem = stem.format(L=L)

    def tree(branch):
        f = sorted(glob.glob(f"{E}/{stem}_depth_16.jsonl.mcts.Qwen3-4B.{branch}.*.jsonl"))[-1]
        return {r["index"]: r["rstar"] for r in map(json.loads, open(f)) if "rstar" in r}
    out = {}
    t = tree(scb)
    out["SCB (tested rule)"] = {i: RULES["tested"](t[i]) for i in t}
    for model, var, name in (("Qwen3-4B", "think", "Qwen3-4B sp2 thinking"), ("phi-4", "nothink", "phi-4 sp2")):
        f = sorted(glob.glob(f"eval_data/paper_prompt_replication/{stem}.jsonl.{model}.sp2.*.jsonl"))[-1]
        out[name] = {i: p for i, (_, p) in S.load_preds(f, "sp2", var).items()}
    t = tree(ext)
    out["Extension (clone-on-disagreement)"] = {i: RULES["current"](t[i]) for i in t}
    return out


def metrics(meta, p, ix):
    tp = sum(meta[i]["kind"] == "clone" and p.get(i) == "clone" for i in ix)
    fn = sum(meta[i]["kind"] == "clone" and p.get(i) != "clone" for i in ix)
    fp = sum(meta[i]["kind"] == "hn_output" and p.get(i) == "clone" for i in ix)
    tn = sum(meta[i]["kind"] == "hn_output" and p.get(i) != "clone" for i in ix)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    rec, spec = tp / max(1, tp + fn), tn / max(1, tn + fp)
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, f1=f1, ba=(rec + spec) / 2, mcc=(tp * tn - fp * fn) / d if d else 0.0,
                rec=rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--set", default="hardeval", choices=list(SETS))
    args = ap.parse_args()
    for L in SETS[args.set][3]:
        stem = SETS[args.set][0].format(L=L)
        meta = {m["index"]: m for m in map(json.loads, open(f"{E}/{stem}_meta.jsonl"))}
        kinds = {k: [i for i in meta if meta[i]["kind"] == k] for k in ("clone", "hn_output", "cross")}
        sysp = systems(L, args.set)
        print(f"== {L}: clone {len(kinds['clone'])} / hard {len(kinds['hn_output'])} / cross {len(kinds['cross'])}")
        hard = kinds["clone"] + kinds["hn_output"]
        for name, p in sysp.items():
            m = metrics(meta, p, hard)
            lo, hi = wilson(m["fp"], m["fp"] + m["tn"])
            cross = sum(p.get(i) == "clone" for i in kinds["cross"]) / max(1, len(kinds["cross"]))
            print(f"  {name:36s} clone-rec {m['rec']:.3f} | bug->clone {m['fp'] / (m['fp'] + m['tn']):.3f} "
                  f"[{lo:.2f}, {hi:.2f}] | cross->clone {cross:.3f} | F1 {m['f1']:.3f} BA {m['ba']:.3f} MCC {m['mcc']:.3f}")
        if args.bootstrap:
            ref = "Extension (clone-on-disagreement)"
            pids = sorted({meta[i]["p_id1"] for i in hard})
            by = {p: [i for i in hard if meta[i]["p_id1"] == p] for p in pids}
            random.seed(0)
            samples = [[i for _ in pids for i in by[random.choice(pids)]] for _ in range(args.bootstrap)]
            r_all = metrics(meta, sysp[ref], hard)
            print(f"  paired bootstrap by problem ({len(pids)} problems, {args.bootstrap} resamples): {ref} minus system")
            for name, p in sysp.items():
                if name == ref:
                    continue
                o_all = metrics(meta, p, hard)
                ds = {k: [] for k in ("f1", "ba", "mcc")}
                for ix in samples:
                    a, b = metrics(meta, sysp[ref], ix), metrics(meta, p, ix)
                    for k in ds:
                        ds[k].append(a[k] - b[k])
                ci = {k: (sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v)) - 1]) for k, v in ds.items()}
                print("    " + f"{name:34s} " + "  ".join(
                    f"{k.upper()} {r_all[k] - o_all[k]:+.3f} [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]" for k in ("f1", "ba", "mcc")))


if __name__ == "__main__":
    main()

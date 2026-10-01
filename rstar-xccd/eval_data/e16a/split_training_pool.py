"""TRAINING_PLAN_2026-09-30 step 2: split the problem-level pool (build_training_pool.py)
BY PROBLEM into a LOCKED hard-negative evaluation set and a training pool.

Evaluation set (eval_data/e16a/hardeval_python_{L}_codenet.jsonl + _meta.jsonl), same
format and same pair kinds as the 250-pair pilot (hard_python_*), which is now DEV:
  per eval problem p (drawn at random among non-suspect problems with >=1 verified
  Wrong Answer program):
    clone      first verified Accepted Python + first agreeing Accepted target
    hn_output  same Python + one verified Wrong Answer target of p (prefers a
               difference other than letter case; case_only kept as a flag)
  cross       same Python of p + Accepted target of ANOTHER eval problem
              (--n_cross pairs, the CLCCD-like reference slice)
Training pool (pool_{L}_train.jsonl): every other pool problem, unchanged records.

LOCK: writes hardeval_{L}_MANIFEST.txt with sha256 of both files, the problem ids
and the creation time. Do not regenerate or tune anything against these files; any
method change after looking at results on them needs a new set.

Usage: python eval_data/e16a/split_training_pool.py --lang java --n_eval 150 --n_cross 75
"""
import argparse
import hashlib
import json
import random
import time

E = "eval_data/e16a"
SEED = 20261001
TEMPLATE = ("Determine whether the following two code snippets are semantic code clones.\n\n"
            "Code 1: Python\n```python\n{c1}```\n\nCode 2: {Lname}\n```{ltag}\n{c2}```")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["java", "rust"], required=True)
    ap.add_argument("--n_eval", type=int, default=150)
    ap.add_argument("--n_cross", type=int, default=75)
    ap.add_argument("--pool", default=None, help="pool file (default eval_data/e16a/pool_{L}.jsonl)")
    ap.add_argument("--note", default="", help="extra line for the manifest (e.g. length limit used)")
    args = ap.parse_args()
    L, Lname = args.lang, args.lang.capitalize()
    pool = [json.loads(l) for l in open(args.pool or f"{E}/pool_{L}.jsonl")]
    rng = random.Random(f"{SEED}-{L}")
    cand = sorted((r for r in pool if not r["multi_answer_suspect"] and r["hns"]), key=lambda r: r["pid"])
    rng.shuffle(cand)
    ev = cand[:args.n_eval]
    ev_ids = {r["pid"] for r in ev}
    pairs = []
    for r in ev:
        py, tg = r["pys"][0], r["tgs"][0]
        hn = next((h for h in r["hns"] if h["diff_type"] != "case_only"), r["hns"][0])
        pairs.append({"answer": "clone", "kind": "clone", "p1": r["pid"], "c1": py, "p2": r["pid"], "c2": tg})
        pairs.append({"answer": "non-clone", "kind": "hn_output", "p1": r["pid"], "c1": py, "p2": r["pid"],
                      "c2": [hn["s_id"], hn["code"]], "input": hn["input"], "expected": hn["expected"],
                      "got": hn["got"], "diff_type": hn["diff_type"]})
    for r in rng.sample(ev, min(args.n_cross, len(ev))):
        q = rng.choice([x for x in ev if x["pid"] != r["pid"]])
        pairs.append({"answer": "non-clone", "kind": "cross", "p1": r["pid"], "c1": r["pys"][0],
                      "p2": q["pid"], "c2": q["tgs"][0]})
    rng.shuffle(pairs)
    fq, fm = f"{E}/hardeval_python_{L}_codenet.jsonl", f"{E}/hardeval_python_{L}_codenet_meta.jsonl"
    with open(fq, "w") as wq, open(fm, "w") as wm:
        for i, p in enumerate(pairs):
            q = TEMPLATE.format(c1=p["c1"][1], c2=p["c2"][1], Lname=Lname, ltag=L)
            wq.write(json.dumps({"index": i, "question": q, "answer": p["answer"]}) + "\n")
            m = {"index": i, "answer": p["answer"], "kind": p["kind"], "p_id1": p["p1"], "s_id1": p["c1"][0],
                 "p_id2": p["p2"], "s_id2": p["c2"][0], "multi_answer_suspect": False}
            m.update({k: p[k] for k in ("input", "expected", "got", "diff_type") if k in p})
            wm.write(json.dumps(m) + "\n")
    train = [r for r in pool if r["pid"] not in ev_ids]
    with open(f"{E}/pool_{L}_train.jsonl", "w") as w:
        for r in train:
            w.write(json.dumps(r) + "\n")
    kinds = {k: sum(p["kind"] == k for p in pairs) for k in ("clone", "hn_output", "cross")}
    with open(f"{E}/hardeval_{L}_MANIFEST.txt", "w") as w:
        w.write(f"LOCKED hard-negative evaluation set, created {time.strftime('%Y-%m-%d %H:%M %Z')} by "
                f"split_training_pool.py (seed {SEED}). Do not tune against it.\n")
        w.write(f"pairs {len(pairs)} {kinds}; eval problems {len(ev_ids)}; pool {args.pool or 'pool_' + L + '.jsonl'}\n")
        if args.note:
            w.write(args.note + "\n")
        w.write(f"sha256 {sha(fq)}  {fq}\nsha256 {sha(fm)}  {fm}\n")
        w.write("eval problem ids: " + " ".join(sorted(ev_ids)) + "\n")
    print(f"{L}: eval {len(pairs)} pairs {kinds} from {len(ev_ids)} problems; "
          f"train pool {len(train)} problems ({sum(len(r['hns']) for r in train)} hard pairs)")


if __name__ == "__main__":
    main()

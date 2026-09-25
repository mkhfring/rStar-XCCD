"""Two-sided problem split of the CLCCD test files (SFT_DATA_PROCESS.txt
section 9), as an alternative to build_rl_split.py's rl_train / rl_heldout.

Why: build_rl_split.py splits by the PYTHON side's problem_id only. CLCCD
reuses every Code-2 program in several records -- its own clone pair plus
derangement-built non-clone pairs with OTHER problems' Python. Measured
(check_leakage.py / this script): 240/240 rust and 69/360 java rl_heldout
records have their exact Code 2 text somewhere in rl_train. A model tuned on
rl_train has therefore already seen every held-out rust Code 2, just paired
differently.

Fix: a Code-2 snippet's problem is the Python problem of the CLONE record it
appears in. Choose a held-out problem set H; a record is held out only if
BOTH its Python problem and its Code-2 problem are in H, kept for training
only if NEITHER is, and dropped otherwise (it would straddle the split).
H is 20% of problems; the seed is chosen deterministically as the one (in
0..199) that maximises the held-out size while keeping both labels -- the
same search reported in section 9 (rust: seed 122, 110 held out / 830 train /
260 dropped; java: seed 66, 404 / 1396 / 0).

Writes eval_data/rl_{train,heldout}_strict_python_{lang}_CLCCD.jsonl; the
original rl_train / rl_heldout files are untouched. `index` values are those
of the source CLCCD test file, so existing mining trees still line up.

Usage:
    python build_rl_split_strict.py --lang rust
"""
import argparse
import collections
import json
import random
import re

R = re.compile(r"Code 1: Python\n```python\n(.*?)```\s*\n+Code 2: (\w+)\n```\w*\n(.*?)```", re.S)


def norm(code):
    return " ".join(code.split())


def split(recs, frac, seed):
    P = [norm(R.search(r["question"]).group(1)) for r in recs]
    C = [norm(R.search(r["question"]).group(3)) for r in recs]
    prob_of_c2 = {c: p for p, c, r in zip(P, C, recs) if r["answer"] == "clone"}
    problems = sorted(set(P))
    H = set(random.Random(seed).sample(problems, int(frac * len(problems))))
    held, train = [], []
    for i, r in enumerate(recs):
        a, b = P[i] in H, prob_of_c2.get(C[i], P[i]) in H
        if a and b:
            held.append(r)
        elif not a and not b:
            train.append(r)
    return held, train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["rust", "java"])
    ap.add_argument("--frac", type=float, default=0.2)
    args = ap.parse_args()
    src = f"eval_data/test_python_{args.lang}_CLCCD.jsonl"
    recs = [json.loads(l) for l in open(src, encoding="utf-8")]
    best = None
    for seed in range(200):
        held, train = split(recs, args.frac, seed)
        labels = collections.Counter(r["answer"] for r in held)
        key = (labels["clone"] > 0 and labels["non-clone"] > 0, len(held))
        if best is None or key > best[0]:
            best = (key, seed, held, train)
    _, seed, held, train = best
    for name, rows in (("heldout", held), ("train", train)):
        out = f"eval_data/rl_{name}_strict_python_{args.lang}_CLCCD.jsonl"
        with open(out, "w", encoding="utf-8") as w:
            for r in rows:
                w.write(json.dumps(r) + "\n")
        print(f"{out}: {len(rows)} records {dict(collections.Counter(r['answer'] for r in rows))}")
    print(f"seed={seed}; dropped {len(recs) - len(held) - len(train)} straddling records")


if __name__ == "__main__":
    main()

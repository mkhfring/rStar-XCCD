"""Annotate hard_python_{L}_codenet_meta.jsonl (build_hard_negatives.py) in place with
  diff_type             how the Wrong Answer output differs from the expected one:
                        wrong_values | missing_output | extra_output | case_only | other
  multi_answer_suspect  True if up to 8 Accepted Python submissions of the problem
                        do not all print the same tokens on the sample inputs
                        (problem likely accepts several answers -> an output
                        difference need not mean a wrong program)
Applies to every row (clones and cross pairs get the multi_answer flag of p_id1).
Report subsets with and without the flagged rows.

Usage (../venv-data):  python eval_data/e16a/audit_hard_negatives.py --lang java
"""
import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

import duckdb

H = "eval_data/e16a/codenet_hf"
E = "eval_data/e16a"


def diff_type(e, g):
    e, g = e.split(), g.split()
    if [x.lower() for x in e] == [x.lower() for x in g]:
        return "case_only"
    if g[:len(e)] == e and len(g) > len(e):
        return "extra_output"
    if e[:len(g)] == g and len(e) > len(g):
        return "missing_output"
    if len(e) == len(g):
        return "wrong_values"
    return "other"


def consensus(job):
    pid, codes, samples = job
    outs = set()
    with tempfile.TemporaryDirectory() as d:
        for i, c in enumerate(codes):
            open(f"{d}/{i}.py", "w").write(c)
            res = []
            for x in samples:
                try:
                    p = subprocess.run(["python3", f"{d}/{i}.py"], input=x, capture_output=True, text=True,
                                       timeout=5)
                except subprocess.TimeoutExpired:
                    res = None
                    break
                if p.returncode:
                    res = None
                    break
                res.append(" ".join(p.stdout.split()))
            if res is not None:
                outs.add(tuple(res))
    return pid, len(outs) > 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    f = f"{E}/hard_python_{args.lang}_codenet{args.suffix}_meta.jsonl"
    rows = [json.loads(l) for l in open(f)]
    pids = sorted({r["p_id1"] for r in rows})
    samples = {k: [x for x in v if x.strip()][:4] for k, v in json.load(open(f"{E}/codenet_samples.json")).items()}
    con = duckdb.connect()
    plist = ",".join(f"'{p}'" for p in pids)
    codes = {}
    for p, c in con.execute(f"""select p_id, code from (select p_id, code, row_number() over
            (partition by p_id order by hash(s_id)) rn from '{H}/Python_*.parquet'
            where status='Accepted' and p_id in ({plist})) where rn <= 8""").fetchall():
        codes.setdefault(p, []).append(c)
    with ProcessPoolExecutor(args.workers) as ex:
        flag = dict(ex.map(consensus, [(p, codes.get(p, []), samples[p]) for p in pids]))
    for r in rows:
        r["multi_answer_suspect"] = flag[r["p_id1"]]
        if r["kind"].startswith("hn"):
            r["diff_type"] = diff_type(r["expected"], r["got"])
    with open(f, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    hn = [r for r in rows if r["kind"].startswith("hn")]
    print(f"{args.lang}: {sum(flag.values())}/{len(pids)} problems multi-answer suspects; "
          f"hard negatives flagged {sum(r['multi_answer_suspect'] for r in hn)}/{len(hn)}; "
          f"clones flagged {sum(r['multi_answer_suspect'] for r in rows if r['kind'] == 'clone')}")


if __name__ == "__main__":
    main()

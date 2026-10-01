"""TRAINING_PLAN_2026-09-30 step 2: problem-level pool of verified pairs for
training (rejection-sampling FT / RL) and for a larger locked hard eval set.

Same verification as build_hard_negatives.py (official CodeNet sample inputs
only, oracle = Accepted Python + Accepted target that agree on every sample;
float-output problems skipped), but per problem it keeps EVERYTHING verified:
  pys   up to 4 Accepted Python programs that run under python3 and print the
        oracle's output on every sample
  tgs   up to 3 Accepted Java/Rust programs that agree with the oracle
  hns   up to 8 Wrong Answer Java/Rust programs that run cleanly and print
        something different on >=1 sample (with that input, expected, got)
  multi_answer_suspect  the Accepted Pythons disagree among themselves
Excluded problems: CLCCD test, dev, E16a traces (excluded_pids()) AND the
250-pair hard pilot sets (hard_python_{L}_codenet_meta.jsonl), which become dev.
The eval/train split is made later, by problem, with split_training_pool.py.

Output: eval_data/e16a/pool_{L}.jsonl, one record per problem:
  {pid, pys:[[s_id, code]], tgs:[[s_id, code]], hns:[{s_id, code, input, expected,
   got, diff_type}], multi_answer_suspect, n_samples}
Run (../venv-data + java/17 + ~/.cargo/bin on PATH):
  python eval_data/e16a/build_training_pool.py --lang java --workers 30
"""
import argparse
import json
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hard_negatives as B  # noqa: E402
from audit_hard_negatives import diff_type  # noqa: E402

E = B.E
SEED = 20261001


def process(job):
    pid, lang, pys, tgs, was, samples = job
    rng = random.Random(f"{SEED}-{pid}")
    with tempfile.TemporaryDirectory() as d:
        py_ok = []
        for s, c in pys:
            cmd = B.build("python", c, d, f"py_{s}")
            outs = [B.run(cmd, x, d) for x in samples]
            if all(rc == 0 and o.strip() for rc, o in outs):
                py_ok.append((s, c, [o.split() for _, o in outs]))
        if not py_ok:
            return {"pid": pid, "skip": "no_python3"}
        ref = py_ok[0][2]
        if any(B.FLOAT.search(" ".join(o)) for o in ref):
            return {"pid": pid, "skip": "float"}
        multi = any(p[2] != ref for p in py_ok[1:])
        tg_ok = []
        for s, c in tgs:
            cmd = B.build(lang, c, d, f"tg_{s}")
            if cmd and all(r[0] == 0 and r[1].split() == o
                           for r, o in zip((B.run(cmd, x, d) for x in samples), ref)):
                tg_ok.append((s, c))
        if not tg_ok:
            return {"pid": pid, "skip": "no_agreeing_target"}
        hns = []
        rng.shuffle(was)
        for s, c in was:
            cmd = B.build(lang, c, d, f"wa_{s}")
            if not cmd:
                continue
            for x, exp in zip(samples, ref):
                rc, o = B.run(cmd, x, d)
                if rc == 0 and o.split() != exp:
                    e, g = " ".join(exp)[:500], o.strip()[:500]
                    hns.append({"s_id": s, "code": c, "input": x, "expected": e, "got": g,
                                "diff_type": diff_type(e, g)})
                    break
        return {"pid": pid, "pys": [[s, c] for s, c, _ in py_ok if not multi or s == py_ok[0][0]],
                "tgs": [list(t) for t in tg_ok], "hns": hns, "multi_answer_suspect": multi,
                "n_samples": len(samples)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["java", "rust"], required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_problems", type=int, default=100000)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--max_len", type=int, default=3000,
                    help="max characters per program (3000 ~ CLCCD test max 2994); larger values add "
                         "problems whose programs are longer than anything in CLCCD")
    args = ap.parse_args()
    L, Lname = args.lang, args.lang.capitalize()
    samples = {k: [x for x in v if x.strip()][:4] for k, v in json.load(open(f"{E}/codenet_samples.json")).items()}
    excl = B.excluded_pids()
    for LL in ("java", "rust"):
        for r in map(json.loads, open(f"{E}/hard_python_{LL}_codenet_meta.jsonl")):
            excl |= {r["p_id1"], r["p_id2"]}
    con = duckdb.connect()
    py = B.fetch(con, "Python", "Accepted", 4, args.max_len)
    tg = B.fetch(con, Lname, "Accepted", 3, args.max_len)
    wa = B.fetch(con, Lname, "Wrong Answer", B.MAX_WA, args.max_len)
    probs = sorted(p for p in set(py) & set(tg) & set(wa) if p not in excl and samples.get(p))
    random.Random(f"{SEED}-{L}").shuffle(probs)
    probs = probs[:args.max_problems]
    print(f"{L}: {len(probs)} eligible problems ({len(excl)} excluded ids)", flush=True)
    skips, n_ok, n_hn = {}, 0, 0
    out = open(f"{E}/pool_{L}{args.suffix}.jsonl", "w")
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process, (p, L, py[p], tg[p], list(wa[p]), samples[p])) for p in probs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"skip": f"exception:{type(e).__name__}"}
            if "skip" in r:
                skips[r["skip"]] = skips.get(r["skip"], 0) + 1
            else:
                out.write(json.dumps(r) + "\n")
                out.flush()
                n_ok += 1
                n_hn += len(r["hns"])
            if i % 50 == 0:
                print(f"  {i}/{len(probs)}: {n_ok} problems kept, {n_hn} hard negatives, skips {skips}", flush=True)
    out.close()
    print(f"{L} DONE: {n_ok} problems kept, {n_hn} hard negatives, skips {skips}", flush=True)


if __name__ == "__main__":
    main()

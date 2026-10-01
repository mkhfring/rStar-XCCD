"""E16a step 1: pick trace pairs from train_python_{L}_codenet.jsonl and run
BOTH programs offline on the official CodeNet sample inputs
(Elfsong/codenet_metadata test_cases) of both problems.

Keeps only pairs whose ground truth the execution confirms:
  clone     -- >=1 sample input on which both programs run cleanly and print
               the same thing, and none on which they disagree
  non-clone -- >=1 input on which Code 1 runs cleanly and Code 2 either runs
               cleanly and prints something DIFFERENT, or exits with an error
               (it cannot process an input Code 1 accepts); clean
               differences are preferred when the trace is built
Pairs flagged NEAR_DUP by check_leakage (leakage_new_{L}.txt) are skipped.

Output: batch_{name}_{L}.jsonl -- {index, question, answer, p_id1, p_id2,
cases:[{input, py_out, code2_out, py_rc, code2_rc, valid, match}]}
The eval venv is used (harness code); sample inputs are read from
codenet_samples.json (dumped once with ../venv-data, see --dump_samples).

Usage:
  python eval_data/e16a/select_and_verify.py --lang rust --n_clone 10 --n_non 10 --name pilot
"""
import argparse
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diff_test_pairs import test_question  # noqa: E402

E = ROOT / "eval_data/e16a"
MAX_SAMPLES = 4


def verify(job):
    q, inputs = job
    try:
        return test_question(q["question"], inputs)
    except Exception as exc:  # compile timeouts etc.
        return {"verdict": f"error {exc!r}"[:200], "cases": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--n_clone", type=int, default=10)
    ap.add_argument("--n_non", type=int, default=10)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[], help="earlier batch files whose indices to skip")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_chars", type=int, default=0, help="skip questions longer than this (0 = no cap)")
    args = ap.parse_args()
    L = args.lang

    samples = json.load(open(E / "codenet_samples.json"))
    near = {int(m) for m in re.findall(rf"train_python_{L}_codenet\.jsonl#(\d+)",
                                       open(E / f"leakage_new_{L}.txt").read())}
    used = set()
    for f in args.exclude:
        used |= {json.loads(l)["index"] for l in open(f)}
    meta = {m["index"]: m for m in map(json.loads, open(E / f"train_python_{L}_codenet_meta.jsonl"))}
    qs = [json.loads(l) for l in open(E / f"train_python_{L}_codenet.jsonl")]
    qs = [q for q in qs if q["index"] not in near and q["index"] not in used
          and samples.get(meta[q["index"]]["p_id1"])
          and (not args.max_chars or len(q["question"]) <= args.max_chars)]
    random.Random(f"{args.seed}-{L}-{args.name}").shuffle(qs)

    out, want = [], {"clone": args.n_clone, "non-clone": args.n_non}
    pos = 0
    with ProcessPoolExecutor(args.workers) as ex:
        while pos < len(qs) and any(want.values()):
            chunk = [q for q in qs[pos:pos + 4 * args.workers] if want[q["answer"]] > 0]
            pos += 4 * args.workers
            jobs = []
            for q in chunk:
                m = meta[q["index"]]
                ins = samples[m["p_id1"]][:MAX_SAMPLES]
                if m["p_id2"] != m["p_id1"]:
                    ins = ins + samples.get(m["p_id2"], [])[:MAX_SAMPLES]
                jobs.append((q, ins))
            for (q, _), r in zip(jobs, ex.map(verify, jobs)):
                valid = [c for c in r["cases"] if c["valid"]]
                n_match = sum(c["match"] for c in valid)
                ok = (n_match >= 1 and n_match == len(valid)) if q["answer"] == "clone" \
                    else any(c["py_rc"] == 0 and c["code2_rc"] != "timeout"
                             and (c["code2_rc"] != 0 or not c["match"]) for c in r["cases"])
                if ok and want[q["answer"]] > 0:
                    want[q["answer"]] -= 1
                    m = meta[q["index"]]
                    out.append({**q, "p_id1": m["p_id1"], "p_id2": m["p_id2"], "cases": r["cases"]})
                print(f"{q['index']} {q['answer']} {r['verdict']} valid={len(valid)} match={n_match} "
                      f"-> {'KEEP' if ok else 'skip'}", flush=True)
    path = E / f"batch_{args.name}_{L}.jsonl"
    with open(path, "w") as w:
        for r in out:
            w.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} verified pairs to {path}; still wanted {want}")


if __name__ == "__main__":
    main()

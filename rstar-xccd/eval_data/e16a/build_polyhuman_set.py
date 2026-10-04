"""External same-task hard-negative set from PolyHuman (Python-Java), built with the rules of our locked sets.

Source: PolyHuman (Sun et al., arXiv 2608.23961; Zenodo 10.5281/zenodo.21800077; CC BY 4.0), built on
CodeContests (CC BY 4.0). All problems are Codeforces, so none overlaps CodeNet/CLCCD/our dev or locked sets.
PolyHuman labels a solution incorrect if it failed ANY judge test (that can be a time limit only), so its
correct/incorrect pairs are not necessarily behaviourally different. We keep a pair only after verifying it,
as in build_hard_negatives.py:

  per problem (candidates: Py Pass1, Java Pass1, Java Fail all 50..3000 chars; tests from
  polyhuman_extract_tests.py = public + up to 8 private official tests, inputs <= 5000 chars):
    skip "float"        expected outputs contain decimals (judge tolerance)
    skip "oracle"       Py Pass1 or Java Pass1 does not exit 0 and print the expected tokens on EVERY test
    skip "multi_answer" Py Pass2 or Java Pass2 exits 0 but prints other tokens than expected on some test
    hn_output           Java Fail exits 0 and prints other tokens than expected on a test (public first);
                        the input, its source (public/private), expected and observed output are recorded
    hn_crash / wa_timeout_only / wa_agrees / wa_compile   no verified output difference -> not used
  clone      = Py Pass1 + Java Pass1 (the verified oracle pair) of the same problem
  hard       = Py Pass1 + Java Fail (hn_output only)
  cross      = Py Pass1 of problem p + Java Pass1 of another sampled problem q (reference slice)
Sample: N problems from the hn_output pool, allocated to difficulty levels in proportion to the pool
(largest remainder), seed SEED. Rules fixed before any model run; the set is used once.

Outputs (eval_data/e16a/): polyhuman_python_java.jsonl {index, question, answer} (eval format),
  polyhuman_python_java_meta.jsonl, polyhuman_java_MANIFEST.txt (counts, skips, sha256, attribution)
Run (from rstar-xccd/, ../venv-data, java/17):
  python eval_data/e16a/build_polyhuman_set.py --n 300 --n_cross 150 --workers 30
"""
import argparse
import collections
import hashlib
import json
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "eval_data/e16a")
from build_hard_negatives import FLOAT, TEMPLATE, build, run  # noqa: E402
from polyhuman_extract_tests import candidates  # noqa: E402

E, X = "eval_data/e16a", "eval_data/external/polyhuman"
SEED = 20261003


def process(job):
    rec, tests = job
    name = rec["ProblemName"]
    inputs = [(i, o.split(), "public") for i, o in tests.get("public", [])] + \
             [(i, o.split(), "private") for i, o in tests.get("private", [])]
    inputs = [x for x in inputs if x[1]]
    base = {"name": name, "difficulty": rec["Difficulty"]}
    if not inputs:
        return {**base, "status": "no_tests"}
    if any(FLOAT.search(" ".join(exp)) for _, exp, _ in inputs):
        return {**base, "status": "float"}
    with tempfile.TemporaryDirectory() as d:
        def ok_on_all(lang, code, tag):
            cmd = build(lang, code, d, tag)
            return cmd is not None and all(rc == 0 and out.split() == exp
                                           for (x, exp, _) in inputs for rc, out in [run(cmd, x, d)])
        if not (ok_on_all("python", rec["PYTHON3_Pass1"], "py1") and ok_on_all("java", rec["JAVA_Pass1"], "j1")):
            return {**base, "status": "oracle"}
        for lang, key, tag in (("python", "PYTHON3_Pass2", "py2"), ("java", "JAVA_Pass2", "j2")):
            if rec.get(key):
                cmd = build(lang, rec[key], d, tag)
                if cmd and any(rc == 0 and out.split() != exp
                               for (x, exp, _) in inputs for rc, out in [run(cmd, x, d)]):
                    return {**base, "status": "multi_answer"}
        cmd = build("java", rec["JAVA_Fail"], d, "jf")
        if cmd is None:
            return {**base, "status": "wa_compile"}
        crash = timeout = False
        for x, exp, src in inputs:
            rc, out = run(cmd, x, d)
            if rc == 0 and out.split() != exp:
                return {**base, "status": "hn_output", "input": x, "input_source": src,
                        "expected": " ".join(exp)[:500], "got": out.strip()[:500], "n_tests": len(inputs)}
            crash |= rc not in (0, "timeout")
            timeout |= rc == "timeout"
        return {**base, "status": "hn_crash" if crash else "wa_timeout_only" if timeout else "wa_agrees"}


def allocate(pool, n, rng):
    by = collections.defaultdict(list)
    for r in pool:
        by[r["difficulty"]].append(r)
    total = len(pool)
    quota = {k: n * len(v) / total for k, v in by.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - alloc[k], reverse=True)[:n - sum(alloc.values())]:
        alloc[k] += 1
    out = []
    for k in sorted(by):
        v = sorted(by[k], key=lambda r: r["name"])
        rng.shuffle(v)
        out += v[:alloc[k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n_cross", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="process only the first K candidates (smoke test)")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    tests = json.load(open(f"{X}/polyhuman_tests.json"))
    cands = sorted((r for r in candidates() if r["ProblemName"] in tests), key=lambda r: r["ProblemName"])
    rng = random.Random(SEED)
    rng.shuffle(cands)
    if args.limit:
        cands = cands[:args.limit]
    print(f"{len(cands)} candidates with tests", flush=True)
    recs = {r["ProblemName"]: r for r in cands}
    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(process, [(c, tests[c["ProblemName"]]) for c in cands], chunksize=2)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}: {dict(collections.Counter(x['status'] for x in results))}", flush=True)
    status = collections.Counter(x["status"] for x in results)
    print(f"DONE {dict(status)}", flush=True)
    pool = [r for r in results if r["status"] == "hn_output"]
    if len(pool) < args.n:
        print(f"WARNING: only {len(pool)} verified hard negatives (< {args.n})", flush=True)
    chosen = allocate(pool, min(args.n, len(pool)), rng)

    pairs = []
    for r in chosen:
        rec = recs[r["name"]]
        pairs.append({"answer": "non-clone", "kind": "hn_output", "p1": r["name"], "c1": rec["PYTHON3_Pass1"],
                      "p2": r["name"], "c2": rec["JAVA_Fail"], "s2": "JAVA_Fail", "difficulty": r["difficulty"],
                      **{k: r[k] for k in ("input", "input_source", "expected", "got")}})
        pairs.append({"answer": "clone", "kind": "clone", "p1": r["name"], "c1": rec["PYTHON3_Pass1"],
                      "p2": r["name"], "c2": rec["JAVA_Pass1"], "s2": "JAVA_Pass1", "difficulty": r["difficulty"]})
    for i in range(min(args.n_cross, len(chosen))):
        a = chosen[i]
        b = rng.choice([r for r in chosen if r["name"] != a["name"]])
        pairs.append({"answer": "non-clone", "kind": "cross", "p1": a["name"], "c1": recs[a["name"]]["PYTHON3_Pass1"],
                      "p2": b["name"], "c2": recs[b["name"]]["JAVA_Pass1"], "s2": "JAVA_Pass1",
                      "difficulty": a["difficulty"]})
    rng.shuffle(pairs)

    stem = f"{E}/polyhuman_python_java{args.suffix}"
    with open(f"{stem}.jsonl", "w") as fq, open(f"{stem}_meta.jsonl", "w") as fm:
        for i, p in enumerate(pairs):
            q = TEMPLATE.format(c1=p["c1"].rstrip() + "\n", c2=p["c2"].rstrip() + "\n", Lname="Java", ltag="java")
            fq.write(json.dumps({"index": i, "question": q, "answer": p["answer"]}) + "\n")
            meta = {"index": i, "answer": p["answer"], "kind": p["kind"], "p_id1": p["p1"], "s_id1": "PYTHON3_Pass1",
                    "p_id2": p["p2"], "s_id2": p["s2"], "difficulty": p["difficulty"]}
            meta.update({k: p[k] for k in ("input", "input_source", "expected", "got") if k in p})
            fm.write(json.dumps(meta) + "\n")
    sha = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (f"{stem}.jsonl", f"{stem}_meta.jsonl")}
    kinds = collections.Counter(p["kind"] for p in pairs)
    diff = collections.Counter(r["difficulty"] for r in chosen)
    src = collections.Counter(r["input_source"] for r in chosen)
    with open(f"{E}/polyhuman_java{args.suffix}_MANIFEST.txt", "w") as f:
        f.write(f"PolyHuman external hard-negative set (Python-Java), built {__file__}, seed {SEED}\n"
                f"Source: Sun, Uchoa, Gheyi, Assuncao, 'Evaluating Language Models on Cross-Language Code Functional\n"
                f"Equivalence', arXiv 2608.23961, Zenodo 10.5281/zenodo.21800077 (CC BY 4.0); tests from DeepMind\n"
                f"CodeContests (CC BY 4.0). Codeforces problems only (no overlap with CodeNet-based sets).\n"
                f"candidates processed: {len(cands)}; status counts: {dict(status)}\n"
                f"pairs: {len(pairs)} {dict(kinds)}\nhard-negative difficulty: {dict(sorted(diff.items()))}\n"
                f"distinguishing input source: {dict(src)}\n"
                + "".join(f"sha256 {v}  {k}\n" for k, v in sha.items()) +
                "LOCKED: rules fixed before any model run; use once; do not tune on it.\n")
    print(f"wrote {len(pairs)} pairs {dict(kinds)} -> {stem}.jsonl; difficulty {dict(sorted(diff.items()))}", flush=True)


if __name__ == "__main__":
    main()

"""Phase 0.3 checks: validity oracle on real CodeContests problems.

CF 146B (Lucky Mask, = PolyHuman pair 98): b must be a lucky number; accepted solutions loop
on other b, so such inputs must come out INVALID, while valid inputs get the agreed answer.

Usage (needs pyarrow -> venv-data, g++ and java on PATH; from rstar-xccd/):
    module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 java/17.0.6
    ../venv-data/bin/python tests/test_validity_oracle.py
"""
import glob
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from validity_oracle import ValidityOracle, classify_calls  # noqa: E402

CC = "/lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd/eval_data/external/code_contests"
LANG = {2: "CPP", 3: "PY3", 4: "JAVA"}
WANT = {"146_B. Lucky Mask", "4_A. Watermelon"}


def load(names):
    found = {}
    for f in sorted(glob.glob(f"{CC}/train-*.parquet")):
        t = pq.read_table(f, columns=["name"]).column("name").to_pylist()
        rows = [i for i, n in enumerate(t) if n in names]
        if not rows:
            continue
        tab = pq.read_table(f, columns=["name", "solutions", "public_tests", "private_tests", "time_limit"]).to_pylist()
        for i in rows:
            found[tab[i]["name"]] = tab[i]
        if len(found) == len(names):
            break
    return found


def problem_args(r):
    sols = [(LANG[l], s) for l, s in zip(r["solutions"]["language"], r["solutions"]["solution"]) if l in LANG]
    tests = list(zip(r["public_tests"]["input"], r["public_tests"]["output"]))
    tests += list(zip(r["private_tests"]["input"], r["private_tests"]["output"]))[:8]
    tl = (r["time_limit"] or {}).get("seconds") or 2
    return sols, tests, float(tl)


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    probs = load(WANT)
    check("T0 problems found in CodeContests", set(probs) == WANT, str(sorted(probs)))
    root = tempfile.mkdtemp(prefix="oracle_test_")
    try:
        o = ValidityOracle(root)
        t = time.monotonic()
        metas = {k: o.add_problem(k, *problem_args(r)) for k, r in probs.items()}
        for k, m in metas.items():
            print(f"     {k}: refs {[r['lang'] for r in m['refs']]} tried {m['tried']} rejected {m['rejected']}")
        check("T1 >= 2 references per problem", all(m["n_refs"] >= 2 for m in metas.values()),
              f"({time.monotonic() - t:.0f}s to build)")

        k = "146_B. Lucky Mask"
        cases = [("39999 4774\n", True, "40774"), ("100000 77777\n", True, "177777"), ("1 7\n", True, "7"),
                 ("1 1\n", False, None), ("10 0\n", False, None), ("abc\n", False, None)]
        for stdin, valid, exp in cases:
            t = time.monotonic()
            v = o.check(k, stdin)
            ok = v.valid == valid and (exp is None or v.expected == exp)
            check(f"T2 146B {stdin.strip()!r:16} -> {v.valid}", ok, f"{v.reason} expected={v.expected} ({time.monotonic() - t:.1f}s)")

        t = time.monotonic()
        v = o.check(k, "1 1\n")
        check("T3 verdict cache", time.monotonic() - t < 0.05 and v.valid is False)
        o2 = ValidityOracle(root)                           # fresh object: reads verdicts.jsonl
        t = time.monotonic()
        v = o2.check(k, "39999 4774\n")
        check("T4 verdict cache survives a new oracle", time.monotonic() - t < 0.5 and v.expected == "40774")

        w = "4_A. Watermelon"
        check("T5 Watermelon 8 -> YES", o.check(w, "8\n").expected == "YES")
        check("T5 Watermelon 2 -> NO", o.check(w, "2\n").expected == "NO")

        calls = [  # what python_tool._record stores on a node (pair 98: Code 2 is the buggy Java)
            dict(stdin="39999 4774\n", stdin_len=11, ok1=True, out1="40774", ok2=True, out2="40074", differ=True, skipped=False),
            dict(stdin="1 1\n", stdin_len=4, ok1=False, out1="", ok2=True, out2="1", differ=False, skipped=False),
            dict(stdin="5 5\n", stdin_len=4, ok1=False, out1="", ok2=False, out2="", differ=False, skipped=True),
        ]
        c = classify_calls(o, k, calls)
        check("T6 classify_calls: valid distinguishing input, Code 1 right, Code 2 wrong",
              c[0]["valid"] and c[0]["match1"] and not c[0]["match2"], str({x: c[0][x] for x in ("valid", "match1", "match2")}))
        check("T6 classify_calls: invalid input flagged; skipped call left unknown",
              c[1]["valid"] is False and c[2]["valid"] is None)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

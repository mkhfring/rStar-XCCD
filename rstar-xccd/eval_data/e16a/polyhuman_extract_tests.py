"""Extract CodeContests judge tests for the PolyHuman Python-Java candidate problems.

PolyHuman (Sun et al., arXiv 2608.23961, Zenodo 10.5281/zenodo.21800077, CC BY 4.0) gives per Codeforces
problem two correct and one incorrect solution per language, but no tests. CodeContests (CC BY 4.0) has the
tests. Candidates = problems whose PYTHON3_Pass1, JAVA_Pass1 and JAVA_Fail are all 50..3000 characters
(the CLCCD length limit). Only official tests are kept: public (= statement samples) and private (judge
tests); generated tests are NOT used (their validity is not guaranteed). Inputs over MAX_IN characters are
dropped.

Inputs : eval_data/external/polyhuman/polyhuman_py_java.json   (exported from the PolyHuman xlsx)
         eval_data/external/code_contests/train-*.parquet       (HF deepmind/code_contests)
Output : eval_data/external/polyhuman/polyhuman_tests.json      {name: {"public": [[in, out]...], "private": [...]}}
Run (from rstar-xccd/, ../venv-data): python eval_data/e16a/polyhuman_extract_tests.py
"""
import json

import duckdb

X = "eval_data/external"
MIN_LEN, MAX_LEN, MAX_IN, MAX_PRIVATE = 50, 3000, 5000, 8


def candidates():
    recs = json.load(open(f"{X}/polyhuman/polyhuman_py_java.json"))
    return [r for r in recs
            if all(r[k] and MIN_LEN <= len(r[k]) <= MAX_LEN for k in ("PYTHON3_Pass1", "JAVA_Pass1", "JAVA_Fail"))]


def main():
    names = sorted({r["ProblemName"] for r in candidates()})
    print(f"{len(names)} candidate problems", flush=True)
    con = duckdb.connect()
    con.execute("create temp table want(name varchar)")
    con.executemany("insert into want values (?)", [(n,) for n in names])
    rows = con.execute(f"""
        select t.name, t.public_tests.input, t.public_tests.output, t.private_tests.input, t.private_tests.output
        from '{X}/code_contests/train-*.parquet' t join want w on t.name = w.name""").fetchall()
    out = {}
    for name, pi, po, qi, qo in rows:
        pub = [[i, o] for i, o in zip(pi or [], po or []) if len(i) <= MAX_IN]
        prv = [[i, o] for i, o in zip(qi or [], qo or []) if len(i) <= MAX_IN][:MAX_PRIVATE]
        out[name] = {"public": pub, "private": prv}
    json.dump(out, open(f"{X}/polyhuman/polyhuman_tests.json", "w"))
    print(f"tests for {len(out)} / {len(names)} problems; "
          f"with >=1 public: {sum(bool(v['public']) for v in out.values())}", flush=True)


if __name__ == "__main__":
    main()

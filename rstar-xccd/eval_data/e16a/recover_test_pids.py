"""Recover CodeNet problem ids of the CLCCD test pairs (E16a / E1) by
matching each Code 1 / Code 2 snippet, whitespace-normalised, against the
CodeNet submissions in eval_data/e16a/codenet_hf (HF mirror iNeil77/CodeNet).
Writes eval_data/e16a/test_pids_{java,rust}.json: {index: {code1: [pids], code2: [pids]}}
and prints the set of test problem ids to exclude."""
import json, re, sys, duckdb
SN = re.compile(r"Code 1: Python\n```python\n(.*?)```\s*\n+Code 2: (\w+)\n```\w*\n(.*?)```", re.S)
def norm(s): return re.sub(r"[ \t\r\n\f\v]+", " ", s).strip()
H = "eval_data/e16a/codenet_hf"
con = duckdb.connect()

for L, CL in (("java", "Java"), ("rust", "Rust")):
    rows = []
    for l in open(f"eval_data/test_python_{L}_CLCCD.jsonl"):
        q = json.loads(l); m = SN.search(q["question"])
        rows.append((q["index"], norm(m.group(1)), norm(m.group(3))))
    con.execute("create or replace table t(idx varchar, c1 varchar, c2 varchar)")
    con.executemany("insert into t values (?,?,?)", rows)
    out = {}
    for col, src in (("c1", "Python"), ("c2", CL)):
        res = con.execute(f"""select t.idx, list(distinct s.p_id) from t join
            (select p_id, trim(regexp_replace(code, $$[ \t\r\n\f\v]+$$, $$ $$, $$g$$)) nc from '{H}/{src}_*.parquet' where length(code) < 20000) s
            on s.nc = t.{col} group by 1""").fetchall()
        for idx, pids in res: out.setdefault(str(idx), {})[col] = pids
    json.dump(out, open(f"eval_data/e16a/test_pids_{L}.json", "w"))
    allp = {p for v in out.values() for ps in v.values() for p in ps}
    print(L, "records", len(rows), "c1 matched", sum('c1' in v for v in out.values()),
          "c2 matched", sum('c2' in v for v in out.values()), "distinct pids", len(allp))

"""Build problem-DISJOINT python-{java,rust} clone-detection pools from full
Project CodeNet (HF mirror iNeil77/CodeNet, eval_data/e16a/codenet_hf) for
EXPERIMENT_PLAN E1 (dev set) and E16a (trace source).

Why: the old CodeNet pools (codenet_train_python_*_clean.jsonl) are ~2 AOJ
problems, useless as an SFT source; E1 needs a dev set whose problems are
disjoint from the CLCCD test pairs.

Construction mirrors the CLCCD test files:
  clone      = an Accepted Python submission + an Accepted Java/Rust
               submission of the SAME problem
  non-clone  = Accepted Python of problem A + Accepted Java/Rust of a
               DIFFERENT problem B
  both snippets 50..3000 chars (test max is 2994), same question template.
Excluded problems: every CodeNet problem id recovered from the CLCCD test
files of EITHER language (test_pids_{java,rust}.json, recover_test_pids.py).
Problems are split once: DEV problems and TRAIN problems never overlap, and
each non-clone draws both of its problems from the same side.
  dev   : java 150 clone / 150 non-clone; rust 75 / 225 (test ratios)
  train : one clone + one non-clone pair per remaining problem

Outputs (eval_data/e16a/):
  dev_python_{L}_codenet.jsonl, train_python_{L}_codenet.jsonl  {index, question, answer}
  *_meta.jsonl  {index, answer, p_id1, s_id1, p_id2, s_id2}
Run with ../venv-data (duckdb): module load StdEnv/2023 python/3.11
"""
import json
import random

import duckdb

H = "eval_data/e16a/codenet_hf"
OUT = "eval_data/e16a"
SEED = 20260928
TEMPLATE = ("Determine whether the following two code snippets are semantic code clones.\n\n"
            "Code 1: Python\n```python\n{c1}```\n\nCode 2: {Lname}\n```{ltag}\n{c2}```")
DEV = {"java": (150, 150), "rust": (75, 225)}


def test_pids():
    out = set()
    for L in ("java", "rust"):
        for v in json.load(open(f"{OUT}/test_pids_{L}.json")).values():
            for ps in v.values():
                out.update(ps)
    return out


def candidates(con, lang, k=3):
    """{p_id: [(s_id, code), ...]} -- up to k random Accepted submissions."""
    rows = con.execute(f"""
        select p_id, s_id, code from (
          select p_id, s_id, code,
                 row_number() over (partition by p_id order by hash(s_id || '{SEED}')) rn
          from '{H}/{lang}_*.parquet'
          where status = 'Accepted' and length(code) between 50 and 3000)
        where rn <= {k}""").fetchall()
    out = {}
    for p, s, c in rows:
        out.setdefault(p, []).append((s, c.rstrip() + "\n"))
    return out


def main():
    con = duckdb.connect()
    excl = test_pids()
    py = candidates(con, "Python")
    for L, Lname in (("java", "Java"), ("rust", "Rust")):
        tg = candidates(con, Lname)
        probs = sorted((set(py) & set(tg)) - excl)
        rng = random.Random(f"{SEED}-{L}")
        rng.shuffle(probs)
        n_clone, n_non = DEV[L]
        n_dev = max(n_clone, n_non) + 50
        splits = {"dev": probs[:n_dev], "train": probs[n_dev:]}
        for name, ps in splits.items():
            pairs = []
            if name == "dev":
                clone_ps, non_ps = ps[:n_clone], ps[:n_non]
            else:
                clone_ps = non_ps = ps
            for p in clone_ps:
                pairs.append(("clone", p, rng.choice(py[p]), p, rng.choice(tg[p])))
            for p in non_ps:
                q = rng.choice(ps)
                while q == p:
                    q = rng.choice(ps)
                pairs.append(("non-clone", p, rng.choice(py[p]), q, rng.choice(tg[q])))
            rng.shuffle(pairs)
            with open(f"{OUT}/{name}_python_{L}_codenet.jsonl", "w") as fq, \
                 open(f"{OUT}/{name}_python_{L}_codenet_meta.jsonl", "w") as fm:
                for i, (ans, p1, (s1, c1), p2, (s2, c2)) in enumerate(pairs):
                    q = TEMPLATE.format(c1=c1, c2=c2, Lname=Lname, ltag=L)
                    fq.write(json.dumps({"index": i, "question": q, "answer": ans}) + "\n")
                    fm.write(json.dumps({"index": i, "answer": ans, "p_id1": p1, "s_id1": s1,
                                         "p_id2": p2, "s_id2": s2}) + "\n")
            n_c = sum(a == "clone" for a, *_ in pairs)
            print(f"{L} {name}: {len(pairs)} pairs ({n_c} clone / {len(pairs) - n_c} non-clone), "
                  f"{len(ps)} problems")
        print(f"{L}: {len(probs)} eligible problems after excluding {len(excl)} test problem ids")


if __name__ == "__main__":
    main()

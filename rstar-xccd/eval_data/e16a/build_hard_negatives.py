"""Hard-negative (same-problem near-miss) evaluation set for the tested-rule
validity check (PAPER_COMPLETION_PLAN_2026-09-30, minimum plan item 2 / old E9).

Every CLCCD test and dev non-clone pairs two DIFFERENT problems. This set adds
non-clones that solve the SAME problem incorrectly:
  hard non-clone = Accepted Python of problem p + WRONG ANSWER Java/Rust of p,
                   kept only if we observe an input on which they differ.
  clone          = Accepted Python of p + Accepted Java/Rust of p (the oracle
                   pair below), for the same problems.
  cross          = Accepted Python of p + Accepted Java/Rust of another problem
                   q (the usual easy non-clone, as a reference slice).

Per problem:
  1. oracle = the first Accepted Python (python3) and the first Accepted target
     program that both run cleanly on every official sample input and print the
     same tokens. No agreement -> problem skipped (multi-answer / bad samples).
     Problems whose sample outputs contain decimals are skipped (float tolerance).
  2. input pool = the official sample inputs (default). With --n_mut > 0 it
     adds mutated samples (standalone integers replaced by values within the
     sample's own min..max range; first line of a multi-line input kept), kept
     only if both oracles (and a second Accepted Python) exit 0 and agree.
     OFF by default: the smoke test showed mutations that break problem
     constraints (duplicates in a permutation) while the oracles still agree.
  3. Wrong Answer submissions are tried in random order; the first one that runs
     cleanly and prints different tokens on a valid input -> kind hn_output.
     If none does, but one crashes / times out on a valid input -> hn_crash
     (reported separately; easier). Else the problem yields no hard negative.
The distinguishing input, expected and observed output are stored in *_meta.

Excluded problems: CLCCD test (test_pids_*), dev (dev_*_meta), and the
problems of the E16a SFT traces (batch_*).

Outputs (eval_data/e16a/):
  hard_python_{L}_codenet.jsonl       {index, question, answer}  (eval format)
  hard_python_{L}_codenet_meta.jsonl  {index, answer, kind, p_id1, s_id1, p_id2,
                                       s_id2, input, input_source, expected, got}
Run with ../venv-data (duckdb) plus java/17 and ~/.cargo/bin on PATH:
  python eval_data/e16a/build_hard_negatives.py --lang java --n 100 --workers 16
"""
import argparse
import json
import os
import random
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb

H = "eval_data/e16a/codenet_hf"
E = "eval_data/e16a"
SEED = 20260930
TEMPLATE = ("Determine whether the following two code snippets are semantic code clones.\n\n"
            "Code 1: Python\n```python\n{c1}```\n\nCode 2: {Lname}\n```{ltag}\n{c2}```")
TIMEOUT = 5
N_MUT = 0  # --n_mut; 0 = official sample inputs only (mutations can break constraints)
MAX_WA = 8
FLOAT = re.compile(r"\d\.\d")
INT = re.compile(r"(?<![\w.])-?\d+(?![\w.])")  # standalone integers only


def excluded_pids():
    out = set()
    for L in ("java", "rust"):
        for v in json.load(open(f"{E}/test_pids_{L}.json")).values():
            for ps in v.values():
                out.update(ps)
        for r in map(json.loads, open(f"{E}/dev_python_{L}_codenet_meta.jsonl")):
            out |= {r["p_id1"], r["p_id2"]}
        train = {r["index"]: r for r in map(json.loads, open(f"{E}/train_python_{L}_codenet_meta.jsonl"))}
        for b in ("pilot", "b1", "b2", "b3"):
            f = f"{E}/batch_{b}_{L}.jsonl"
            if os.path.exists(f):
                for r in map(json.loads, open(f)):
                    t = train.get(r["index"])
                    if t:
                        out |= {t["p_id1"], t["p_id2"]}
    return out


def fetch(con, lang, status, k, max_len=3000):
    rows = con.execute(f"""
        select p_id, s_id, code from (
          select p_id, s_id, code,
                 row_number() over (partition by p_id order by hash(s_id || '{SEED}')) rn
          from '{H}/{lang}_*.parquet'
          where status = '{status}' and length(code) between 50 and {max_len})
        where rn <= {k}""").fetchall()
    out = {}
    for p, s, c in rows:
        out.setdefault(p, []).append((s, c.rstrip() + "\n"))
    return out


# ---------------------------------------------------------------- execution
def run(cmd, inp, cwd):
    try:
        p = subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=TIMEOUT, cwd=cwd)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return "timeout", ""
    except Exception as e:  # noqa: BLE001
        return f"err:{e}", ""


def build(lang, code, d, name):
    """Compile into directory d; return the run command or None."""
    if lang == "python":
        open(f"{d}/{name}.py", "w").write(code)
        return ["python3", f"{d}/{name}.py"]
    if lang == "java":
        m = re.search(r"public\s+(?:final\s+)?class\s+(\w+)", code) or re.search(r"class\s+(\w+)", code)
        cls = m.group(1) if m else "Main"
        sub = f"{d}/{name}"
        os.makedirs(sub, exist_ok=True)
        open(f"{sub}/{cls}.java", "w").write(code)
        r = subprocess.run(["javac", "-nowarn", f"{cls}.java"], cwd=sub, capture_output=True, timeout=120)
        return None if r.returncode else ["java", "-Xss64m", "-cp", sub, cls]
    open(f"{d}/{name}.rs", "w").write(code)
    r = subprocess.run(["rustc", "-O", "--edition", "2018", f"{name}.rs", "-o", name], cwd=d,
                       capture_output=True, timeout=180)
    return None if r.returncode else [f"{d}/{name}"]


def mutate(inp, rng):
    lines = inp.split("\n")
    nums = [int(x) for x in INT.findall(inp)]
    if not nums:
        return None
    lo, hi = min(nums), max(nums)
    start = 1 if len([l for l in lines if l.strip()]) > 1 else 0

    def rep(m):
        return str(rng.randint(lo, hi)) if rng.random() < 0.5 else m.group(0)
    out = lines[:start] + [INT.sub(rep, l) for l in lines[start:]]
    new = "\n".join(out)
    return new if new != inp else None


def process(job):
    """One problem -> dict with the oracle, hard negative (if any) or a skip reason."""
    pid, lang, pys, tgs, was, samples = job
    rng = random.Random(f"{SEED}-{pid}")
    with tempfile.TemporaryDirectory() as d:
        # 1. oracle
        py_ok = []
        for s, c in pys:
            cmd = build("python", c, d, f"py_{s}")
            outs = [run(cmd, x, d) for x in samples]
            if all(rc == 0 and o.strip() for rc, o in outs):
                py_ok.append((s, c, cmd, [o.split() for _, o in outs]))
        if not py_ok:
            return {"pid": pid, "skip": "no_python3"}
        ps, pc, pcmd, pout = py_ok[0]
        if any(FLOAT.search(" ".join(o)) for o in pout):
            return {"pid": pid, "skip": "float"}
        tg = None
        for s, c in tgs:
            cmd = build(lang, c, d, f"tg_{s}")
            if cmd and all(r[0] == 0 and r[1].split() == o for r, o in
                           zip((run(cmd, x, d) for x in samples), pout)):
                tg = (s, c, cmd)
                break
        if not tg:
            return {"pid": pid, "skip": "no_agreeing_target"}
        ts, tc, tcmd = tg
        oracles = [pcmd, tcmd] + ([py_ok[1][2]] if len(py_ok) > 1 else [])
        # 2. inputs: samples first, then validated mutations
        pool = [(x, "sample", o) for x, o in zip(samples, pout)]
        tried = 0
        while tried < N_MUT * len(samples):
            tried += 1
            x = mutate(rng.choice(samples), rng)
            if x is None or any(x == y for y, _, _ in pool):
                continue
            res = [run(c, x, d) for c in oracles]
            if all(rc == 0 for rc, _ in res) and res[0][1].strip() and \
                    all(o.split() == res[0][1].split() for _, o in res):
                pool.append((x, "mutated", res[0][1].split()))
        # 3. wrong answers
        crash = None
        rng.shuffle(was)
        for s, c in was[:MAX_WA]:
            cmd = build(lang, c, d, f"wa_{s}")
            if not cmd:
                continue
            for x, src, exp in pool:
                rc, o = run(cmd, x, d)
                if rc == 0 and o.split() != exp:
                    return {"pid": pid, "py": (ps, pc), "tg": (ts, tc), "kind": "hn_output",
                            "wa": (s, c), "input": x, "input_source": src,
                            "expected": " ".join(exp)[:500], "got": o.strip()[:500], "n_inputs": len(pool)}
                if rc != 0 and crash is None:
                    crash = {"wa": (s, c), "input": x, "input_source": src,
                             "expected": " ".join(exp)[:500], "got": f"<exit {rc}>"}
        base = {"pid": pid, "py": (ps, pc), "tg": (ts, tc), "n_inputs": len(pool)}
        if crash:
            return {**base, "kind": "hn_crash", **crash}
        return {**base, "kind": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["java", "rust"], required=True)
    ap.add_argument("--n", type=int, default=100, help="hard negatives (= clones) wanted")
    ap.add_argument("--n_cross", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_problems", type=int, default=100000)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n_mut", type=int, default=0,
                    help="mutated inputs per sample; >0 adds inputs that may violate problem constraints")
    args = ap.parse_args()
    global N_MUT
    N_MUT = args.n_mut
    L, Lname = args.lang, args.lang.capitalize()

    samples = {k: [x for x in v if x.strip()][:4] for k, v in json.load(open(f"{E}/codenet_samples.json")).items()}
    con = duckdb.connect()
    excl = excluded_pids()
    py = fetch(con, "Python", "Accepted", 4)
    tg = fetch(con, Lname, "Accepted", 3)
    wa = fetch(con, Lname, "Wrong Answer", MAX_WA)
    probs = sorted(p for p in set(py) & set(tg) & set(wa) if p not in excl and samples.get(p))
    rng = random.Random(f"{SEED}-{L}")
    rng.shuffle(probs)
    probs = probs[:args.max_problems]
    print(f"{L}: {len(probs)} eligible problems ({len(excl)} excluded ids)", flush=True)

    results, skips = [], {}
    n_out = 0
    with ProcessPoolExecutor(args.workers) as ex:
        it = iter(probs)
        futs = {ex.submit(process, (p, L, py[p], tg[p], list(wa[p]), samples[p])): p
                for p in [next(it) for _ in range(min(len(probs), args.workers * 2))]}
        while futs:
            f = next(as_completed(futs))
            futs.pop(f)
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"skip": f"exception:{type(e).__name__}"}
            if "skip" in r:
                skips[r["skip"]] = skips.get(r["skip"], 0) + 1
            else:
                results.append(r)
                n_out += r["kind"] == "hn_output"
            done = len(results) + sum(skips.values())
            if done % 25 == 0:
                kinds = {k: sum(x["kind"] == k for x in results) for k in ("hn_output", "hn_crash", "none")}
                print(f"  {done} problems: {kinds} skips {skips}", flush=True)
            if n_out < args.n * 1.2:
                p = next(it, None)
                if p:
                    futs[ex.submit(process, (p, L, py[p], tg[p], list(wa[p]), samples[p]))] = p

    kinds = {k: [x for x in results if x["kind"] == k] for k in ("hn_output", "hn_crash", "none")}
    print(f"{L} DONE: {({k: len(v) for k, v in kinds.items()})} skips {skips}", flush=True)
    # selection: output-differences first, crash-only fills the rest
    hn = (kinds["hn_output"] + kinds["hn_crash"])[:args.n]
    rng.shuffle(hn)
    pairs = []
    for r in hn:
        pairs.append({"answer": "non-clone", "kind": r["kind"], "p1": r["pid"], "c1": r["py"], "p2": r["pid"],
                      "c2": r["wa"], **{k: r[k] for k in ("input", "input_source", "expected", "got")}})
        pairs.append({"answer": "clone", "kind": "clone", "p1": r["pid"], "c1": r["py"], "p2": r["pid"],
                      "c2": r["tg"]})
    pool = [r for r in results if "tg" in r]
    hn_p = [r["pid"] for r in hn]
    for i in range(min(args.n_cross, len(hn_p))):
        a = hn[i]
        b = rng.choice([r for r in pool if r["pid"] != a["pid"]])
        pairs.append({"answer": "non-clone", "kind": "cross", "p1": a["pid"], "c1": a["py"], "p2": b["pid"],
                      "c2": b["tg"]})
    rng.shuffle(pairs)
    suf = args.suffix
    with open(f"{E}/hard_python_{L}_codenet{suf}.jsonl", "w") as fq, \
         open(f"{E}/hard_python_{L}_codenet{suf}_meta.jsonl", "w") as fm:
        for i, p in enumerate(pairs):
            q = TEMPLATE.format(c1=p["c1"][1], c2=p["c2"][1], Lname=Lname, ltag=L)
            fq.write(json.dumps({"index": i, "question": q, "answer": p["answer"]}) + "\n")
            meta = {"index": i, "answer": p["answer"], "kind": p["kind"], "p_id1": p["p1"], "s_id1": p["c1"][0],
                    "p_id2": p["p2"], "s_id2": p["c2"][0]}
            meta.update({k: p[k] for k in ("input", "input_source", "expected", "got") if k in p})
            fm.write(json.dumps(meta) + "\n")
    c = {k: sum(p["kind"] == k for p in pairs) for k in ("clone", "hn_output", "hn_crash", "cross")}
    print(f"{L}: wrote {len(pairs)} pairs {c} -> {E}/hard_python_{L}_codenet{suf}.jsonl", flush=True)


if __name__ == "__main__":
    main()

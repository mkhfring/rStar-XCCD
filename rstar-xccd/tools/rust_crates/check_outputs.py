"""Phase 0.2: do the Rust Code 2 binaries the code-step harness now builds give right outputs?

Oracle: clone pairs only (Code 1 = Python solution of the SAME CodeNet problem). Both are
run on the problem's sample inputs (codenet_samples.json) through the real harness path,
python_tool.make_run_both (build cache, run timeout, output comparison `same`).
Programs are split by the committed harness's build outcome (from measure_build_rate.py):
  newly_built -- refused/failed before, built now (crates)  -> the group under test
  control     -- built before too (now with -O)             -> baseline agreement rate
Multi-answer problems and wrong "clone" labels disagree in both groups alike.

Run from rstar-xccd/ (venv-qwen3, ~/.cargo/bin on PATH):
  python tools/rust_crates/check_outputs.py --builds <measure out>/builds.jsonl --out <dir> [--workers 30]
"""
import argparse, collections, concurrent.futures as cf, hashlib, json, os, sys

sys.path.insert(0, ".")
from rstar_deepthink.tools import python_tool as pt

M = "../../rStar-XCCD/rstar-xccd/eval_data/"
E = M + "e16a/"
SETS = {   # name: (pairs file, pid source)
    "clccd_test": (M + "test_python_rust_CLCCD.jsonl", "clccd"),
    "dev": (E + "dev_python_rust_codenet.jsonl", E + "dev_python_rust_codenet_meta.jsonl"),
    "hard": (E + "hard_python_rust_codenet.jsonl", E + "hard_python_rust_codenet_meta.jsonl"),
    "hardeval": (E + "hardeval_python_rust_codenet.jsonl", E + "hardeval_python_rust_codenet_meta.jsonl"),
    "train_codenet": (E + "train_python_rust_codenet.jsonl", E + "train_python_rust_codenet_meta.jsonl"),
}
MAX_INPUTS = 5


def clone_pairs():
    """{rust md5: (set, pid, question)} -- first clone pair per distinct Rust program."""
    clccd_pids = json.load(open(E + "test_pids_rust.json"))
    out = {}
    for name, (path, src) in SETS.items():
        meta = {} if src == "clccd" else {str(r["index"]): r for r in map(json.loads, open(src))}
        for r in map(json.loads, open(path)):
            if r["answer"] != "clone":
                continue
            idx = str(r["index"])
            if src == "clccd":
                p = clccd_pids.get(idx, {})
                both = sorted(set(p.get("c1", [])) & set(p.get("c2", [])))
                pid = both[0] if both else None
            else:
                m = meta.get(idx, {})
                pid = m.get("p_id2") if m.get("p_id1") == m.get("p_id2") else None
            lang, code2 = pt.extract_code2_source(r["question"])
            k = hashlib.md5(code2.encode()).hexdigest()
            if pid and k not in out:
                out[k] = (name, pid, r["question"])
    return out


_SAMPLES = None


def check(args):
    global _SAMPLES
    k, name, pid, q, group = args
    if _SAMPLES is None:
        pt.set_rust_crate_check("strict")
        _SAMPLES = json.load(open(E + "codenet_samples.json"))
    inputs = [s for s in _SAMPLES.get(pid, []) if s.strip()][:MAX_INPUTS]
    row = {"key": k, "set": name, "pid": pid, "group": group, "n_inputs": len(inputs)}
    if not inputs:
        return {**row, "verdict": "no_samples"}
    log = []
    run_both, built, status = pt.make_run_both(q, log)
    if not built:
        return {**row, "verdict": "rust_unbuilt", "detail": status[:300]}
    res = [run_both(s) for s in inputs]
    ok1 = [r1.ok for r1, _ in res]
    if not any(ok1):
        return {**row, "verdict": "python_failed", "detail": res[0][0].err[:200]}
    usable = [(r1, r2) for r1, r2 in res if r1.ok]
    if any(not r2.ok for _, r2 in usable):
        r2 = next(r2 for _, r2 in usable if not r2.ok)
        return {**row, "verdict": "rust_runtime_fail", "detail": pt.err(r2)[:200] or r2.err[:200]}
    if all(pt.same(r1, r2) for r1, r2 in usable):
        return {**row, "verdict": "agree", "n_used": len(usable)}
    r1, r2 = next((a, b) for a, b in usable if not pt.same(a, b))
    return {**row, "verdict": "differ", "detail": f"py={r1.out[:80]!r} rs={r2.out[:80]!r}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    old = {r["key"]: r["outcome"] for r in map(json.loads, open(a.builds)) if r["mode"] == "old"}
    pairs = clone_pairs()
    jobs = [(k, n, p, q, "control" if old.get(k) == "built" else "newly_built")
            for k, (n, p, q) in sorted(pairs.items())]
    print(f"{len(jobs)} Rust programs with a same-problem clone pair and a pid", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    with cf.ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(check, jobs, chunksize=2))
    with open(os.path.join(a.out, "outputs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    V = ("agree", "differ", "rust_runtime_fail", "python_failed", "no_samples", "rust_unbuilt")
    print(f"\n{'set':14s} {'group':12s} {'n':>5s} " + " ".join(f"{v:>17s}" for v in V) + "   agree/(agree+differ+rs_fail)")
    for name in list(SETS) + ["ALL"]:
        for g in ("newly_built", "control"):
            s = [r for r in rows if r["group"] == g and (name == "ALL" or r["set"] == name)]
            c = collections.Counter(r["verdict"] for r in s)
            den = c["agree"] + c["differ"] + c["rust_runtime_fail"]
            print(f"{name:14s} {g:12s} {len(s):5d} " + " ".join(f"{c[v]:17d}" for v in V)
                  + f"   {c['agree'] / den if den else float('nan'):.3f}")
    print("\nnewly_built non-agreeing:")
    for r in rows:
        if r["group"] == "newly_built" and r["verdict"] in ("differ", "rust_runtime_fail", "rust_unbuilt"):
            print(f"  {r['set']:13s} {r['pid']} {r['key'][:8]} {r['verdict']}: {r.get('detail', '')[:150]}")


if __name__ == "__main__":
    main()

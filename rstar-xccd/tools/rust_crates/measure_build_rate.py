"""Phase 0.2: Rust Code 2 build rate, committed harness vs offline crate set.

Builds every distinct Rust Code 2 of the code-step data sets with the REAL
python_tool.stage_code2, three ways:
  old       -- committed HEAD (crate `use` -> refused; plain `rustc`, no -O)
  new_nocr  -- working-tree stage_code2 with the crate set disabled (-O only)
  new       -- working-tree stage_code2 with $RUST_CRATES_DIR linked in
rust_crate_check = strict in all three (as in every code-step config).

Run from rstar-xccd/ (venv-qwen3 python, ~/.cargo/bin on PATH):
  python tools/rust_crates/measure_build_rate.py --out <dir> [--workers 32] [--limit N]
Writes <dir>/builds.jsonl (one row per program x mode) and prints a per-set summary.
"""
import argparse, collections, concurrent.futures as cf, hashlib, importlib.util, json, os, re
import subprocess, sys, tempfile, time

sys.path.insert(0, ".")
MAIN = "../../rStar-XCCD/rstar-xccd/eval_data/"
SETS = {
    "clccd_test": MAIN + "test_python_rust_CLCCD.jsonl",
    "dev": MAIN + "e16a/dev_python_rust_codenet.jsonl",
    "hard": MAIN + "e16a/hard_python_rust_codenet.jsonl",
    "hardeval": MAIN + "e16a/hardeval_python_rust_codenet.jsonl",
    "train_codenet": MAIN + "e16a/train_python_rust_codenet.jsonl",
    "codenet_train_old": MAIN + "codenet_train_python_rust.jsonl",
}
MODES = ("old", "new_nocr", "new")
_MODS = {}


def _load(mode):
    if mode in _MODS:
        return _MODS[mode]
    if mode == "old":
        path = os.path.join(tempfile.gettempdir(), f"pt_head_{os.getpid()}.py")
        with open(path, "w") as f:
            f.write(subprocess.check_output(
                ["git", "show", "HEAD:rstar-xccd/rstar_deepthink/tools/python_tool.py"], text=True))
    else:
        path = "rstar_deepthink/tools/python_tool.py"
    spec = importlib.util.spec_from_file_location(f"pt_{mode}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.set_rust_crate_check("strict")
    if mode == "new_nocr":
        m._RUST_EXTERNS = {}
    _MODS[mode] = m
    return m


def classify(status):
    if status.startswith("Code 2 (Rust) compiled"):
        return "built", ""
    if status.startswith("Code 2 uses external crate"):
        return "refused", ""
    codes = re.findall(r"error\[(E\d{4})\]", status)
    first = next((l for l in status.splitlines() if l.startswith("error")), status.splitlines()[0][:120])
    return "failed", (codes[0] if codes else first[:120])


def build(args):
    mode, key, src = args
    m = _load(mode)
    with tempfile.TemporaryDirectory(prefix="rbm_") as d:
        t = time.time()
        status = m.stage_code2("rust", src, d)
        dt = time.time() - t
    outcome, err = classify(status)
    return {"mode": mode, "key": key, "outcome": outcome, "err": err, "sec": round(dt, 2),
            "status": status[:600]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from rstar_deepthink.tools.python_tool import extract_code2_source
    progs, members = {}, collections.defaultdict(set)
    for name, path in SETS.items():
        for line in open(path):
            r = json.loads(line)
            lang, src = extract_code2_source(r["question"])
            if lang != "rust" or not src:
                continue
            k = hashlib.md5(src.encode()).hexdigest()
            progs[k] = src
            members[name].add(k)
    keys = sorted(progs)[: a.limit or None]
    print(f"{len(progs)} distinct programs; measuring {len(keys)} x {len(MODES)} modes", flush=True)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "builds.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["mode"], r["key"]) for r in map(json.loads, open(out))}
    jobs = [(m, k, progs[k]) for k in keys for m in MODES if (m, k) not in done]
    with open(out, "a") as f, cf.ProcessPoolExecutor(a.workers) as ex:
        for i, row in enumerate(ex.map(build, jobs, chunksize=4), 1):
            f.write(json.dumps(row) + "\n")
            if i % 200 == 0:
                f.flush()
                print(f"  {i}/{len(jobs)}", flush=True)
    with open(os.path.join(a.out, "members.json"), "w") as f:
        json.dump({k: sorted(v) for k, v in members.items()}, f)
    summarize(out, members, set(keys))


def summarize(out, members, keys):
    res = {(r["mode"], r["key"]): r for r in map(json.loads, open(out))}
    print(f"\n{'set':18s} {'n':>5s}  " + "  ".join(f"{m:>22s}" for m in MODES))
    for name in list(SETS) + ["ALL"]:
        ks = (keys if name == "ALL" else members[name] & keys)
        cells = []
        for m in MODES:
            c = collections.Counter(res[(m, k)]["outcome"] for k in ks if (m, k) in res)
            n = sum(c.values()) or 1
            cells.append(f"built {c['built']/n:5.1%} ref {c['refused']:3d} fail {c['failed']:3d}")
        print(f"{name:18s} {len(ks):5d}  " + "  ".join(f"{x:>22s}" for x in cells))
    for m in MODES:
        errs = collections.Counter(r["err"] for (mm, _), r in res.items() if mm == m and r["outcome"] == "failed")
        secs = sorted(r["sec"] for (mm, _), r in res.items() if mm == m and r["outcome"] == "built")
        med = secs[len(secs) // 2] if secs else 0
        print(f"\n[{m}] median build {med:.1f}s, max {secs[-1] if secs else 0:.1f}s; top errors: {errs.most_common(8)}")


if __name__ == "__main__":
    main()

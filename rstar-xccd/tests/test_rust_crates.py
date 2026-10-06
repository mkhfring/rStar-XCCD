"""Phase 0.2 checks (training plan TRAINING_PLAN_CODESTEP_2026-10-05.txt): Rust Code 2 builds
against the offline crate set ($RUST_CRATES_DIR, built by tools/rust_crates/build.sh).

Usage (from rstar-xccd/, ~/.cargo/bin on PATH):
    ../venv-qwen3/bin/python tests/test_rust_crates.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rstar_deepthink.tools import python_tool as pt  # noqa: E402

pt.set_rust_crate_check("strict")

USE_PROCONIO = """use proconio::input;
use itertools::Itertools;
fn main() {
    input! { n: usize, a: [i64; n] }
    println!("{}", a.iter().sorted().join(" "));
}
"""
PATH_ONLY = """fn main() {
    proconio::input! { a: i64, b: i64 }
    println!("{}", num::integer::gcd(a, b));
}
"""
NO_CRATES = """use std::io::Read;
fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let v: Vec<i64> = s.split_whitespace().map(|x| x.parse().unwrap()).collect();
    println!("{}", v[0] * v[1]);
}
"""
REAL_ERROR = """use proconio::input;
fn main() {
    input! { n: usize }
    println!("{}", m);
}
"""


def build_and_run(src, stdin):
    with tempfile.TemporaryDirectory() as d:
        status = pt.stage_code2("rust", src, d)
        if not status.startswith("Code 2 (Rust) compiled"):
            return status, None
        out = subprocess.run([os.path.join(d, "candidate_code2")], input=stdin,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return status, out


def main():
    assert pt._rust_externs(), f"no externs.json under ${pt.RUST_CRATES_ENV} / {pt.RUST_CRATES_DEFAULT}"
    checks = []
    st, out = build_and_run(USE_PROCONIO, "4\n3 1 4 1\n")
    checks.append(("`use proconio` + itertools builds and runs", out == "1 1 3 4", st[:200]))
    st, out = build_and_run(PATH_ONLY, "12 18\n")
    checks.append(("path-only proconio:: / num:: builds", out == "6", st[:200]))
    st, out = build_and_run(NO_CRATES, "4000000000 4000000000\n")
    checks.append(("std-only program built with -O (overflow wraps, no debug panic)", out == str((4000000000 * 4000000000 + 2**63) % 2**64 - 2**63), st[:200]))
    st, out = build_and_run(REAL_ERROR, "1\n")
    checks.append(("failed crate build reports the real error, not 'unresolved import'",
                   out is None and "cannot find value `m`" in st and "unresolved import" not in st, st[:300]))
    for name, ok, detail in checks:
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"\n     {detail}"))
    if not all(ok for _, ok, _ in checks):
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()

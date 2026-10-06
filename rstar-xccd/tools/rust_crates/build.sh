#!/bin/bash
# Build the offline crate set (login node: needs network) and write externs.json:
#   { crate_name: rlib_path, ..., "_deps_dir": dir }  used by python_tool.stage_code2.
# Usage: RUST_CRATES_DIR=/project/.../rust_crates_atcoder2020 tools/rust_crates/build.sh
set -euo pipefail
: "${RUST_CRATES_DIR:?set RUST_CRATES_DIR}"
export PATH=$HOME/.cargo/bin:$PATH
unset LD_PRELOAD
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RUST_CRATES_DIR"
cp "$HERE/Cargo.toml" "$RUST_CRATES_DIR/"
mkdir -p "$RUST_CRATES_DIR/src" && cp "$HERE/src/main.rs" "$RUST_CRATES_DIR/src/"
cd "$RUST_CRATES_DIR"
cargo build --release --message-format=json > build.jsonl
python3 - <<'PY'
import json, os
direct = set()
meta = json.loads(os.popen("cargo metadata --format-version 1").read())
root = next(p for p in meta["packages"] if p["name"] == "codestep_crates")
direct = {d["name"].replace("-", "_") for d in root["dependencies"]}
ids = {}
for line in open("build.jsonl"):
    m = json.loads(line)
    if m.get("reason") != "compiler-artifact":
        continue
    t = m["target"]
    name = t["name"].replace("-", "_")
    if name in direct and ({"lib", "rlib", "proc-macro"} & set(t["kind"])):
        f = [x for x in m["filenames"] if x.endswith((".rlib", ".so"))]
        if f:
            ids[name] = f[0]
ids["_deps_dir"] = os.path.abspath("target/release/deps")
json.dump(ids, open("externs.json", "w"), indent=1)
print(f"{len(ids) - 1} crates:", sorted(k for k in ids if not k.startswith("_")))
missing = direct - set(ids)
print("missing:", sorted(missing))
PY

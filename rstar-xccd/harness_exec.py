"""Run code exactly the way the MCTS harness does, outside of a search run.

Used when curating or repairing SFT traces by hand (SFT_DATA_PROCESS.txt
section 9): a hand-edited or teacher-written <code> step must carry the
observation the real harness would have produced for it, never a typed-in
one. This module mirrors rstar_deepthink/agents/tree.py's _code_execution()
step for step -- same candidate_code.py staging, same Code 2 compile gate
(mentions_code2_execution + stage_code2, status appended to the
observation), same PythonInterpreter tool -- but runs in a throwaway
directory so nothing leaks into the repo.

Also exposes run_program()/compile_code2() for diff_test_pairs.py, which
runs both snippets of a pair directly on shared inputs.

Usage (CLI, for one-off checks):
    python harness_exec.py QUESTION_JSONL INDEX CODE_FILE
"""
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rstar_deepthink.tools.python_tool import (
    PythonInterpreter,
    extract_code1_python,
    extract_code2_source,
    is_python_code,
    mentions_code2_execution,
    rust_external_crates,
    sanitize_input,
    stage_code2,
)

NO_CODE_MESSAGE = "No valid Python code found in the response."
RUN_TIMEOUT = 10


@contextmanager
def scratch_dir():
    old = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="harness_exec_") as d:
        os.chdir(d)
        try:
            yield d
        finally:
            os.chdir(old)


def harness_observation(question: str, code: str) -> str:
    """The observation tree.py would attach to a python_interpreter step whose
    accumulated program is `code`."""
    sanitized = sanitize_input(code)
    if not is_python_code(sanitized):
        return NO_CODE_MESSAGE
    with scratch_dir():
        code1 = extract_code1_python(question)
        if code1:
            with open("candidate_code.py", "w") as f:
                f.write(code1)
        code2_status = ""
        if mentions_code2_execution(sanitized):
            lang, src = extract_code2_source(question)
            if src:
                code2_status = stage_code2(lang, src)
        observation = str(PythonInterpreter().run(code)).strip()
    if code2_status:
        observation = f"{observation}\n{code2_status}"
    return observation


ATCODER_DEPS = Path(__file__).resolve().parent / "eval_data/claude_review/atcoder_crates/target/release/deps"


def atcoder_extern_flags():
    if not ATCODER_DEPS.is_dir():
        return []
    flags = ["-L", f"dependency={ATCODER_DEPS}"]
    seen = set()
    for rlib in sorted(ATCODER_DEPS.glob("lib*.rlib")):
        name = rlib.name[3:].rsplit("-", 1)[0]
        if name not in seen:
            seen.add(name)
            flags += ["--extern", f"{name}={rlib}"]
    return flags


def compile_code2(question: str, workdir: str):
    """Compile Code 2 into workdir. Returns (argv or None, status string)."""
    lang, src = extract_code2_source(question)
    if not src:
        return None, "no Code 2 source found"
    if lang == "rust" and (rust_external_crates(src) or "proconio::" in src):
        # rust_external_crates() over-matches (a `use place` inside a comment,
        # a local `mod math`, `::std::io` paths), so for offline testing try
        # plain rustc anyway; only a real compile failure counts.
        # A real external crate (proconio etc.) is linked from the AtCoder
        # 2020 crate set built once under eval_data/claude_review/atcoder_crates
        # (cargo, login node) -- OFFLINE testing only; the MCTS harness itself
        # still cannot run these programs.
        src_path = os.path.join(workdir, "candidate_code2.rs")
        bin_path = os.path.join(workdir, "candidate_code2")
        Path(src_path).write_text(src)
        err = ""
        for extra in ([], ["--edition", "2018"], ["--edition", "2018"] + atcoder_extern_flags()):
            p = subprocess.run(["rustc", "-O", *extra, src_path, "-o", bin_path], cwd=workdir,
                               capture_output=True, text=True, timeout=180)
            if p.returncode == 0:
                return [bin_path], f"compiled offline ({' '.join(extra[:2]) or 'plain rustc'}); harness heuristic flags {sorted(rust_external_crates(src))}"
            err = p.stderr.strip()[:200]
        return None, f"external crates {sorted(rust_external_crates(src))}; rustc: {err}"
    status = stage_code2(lang, src, workdir=workdir)
    if lang == "rust" and "compiled to" in status:
        return [os.path.join(workdir, "candidate_code2")], status
    if lang == "java" and "compiled:" in status:
        cls = status.split("[\"java\", \"-cp\", ")[1].split(", ")[1].split("]")[0].strip("'\"")
        return ["java", "-cp", workdir, cls], status
    return None, status


def run_program(argv, stdin_text: str):
    """Run one program on one input. Returns (stdout, stderr, returncode or
    'timeout')."""
    try:
        p = subprocess.run(argv, input=stdin_text, capture_output=True, text=True,
                           timeout=RUN_TIMEOUT)
        return p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return "", "", "timeout"


if __name__ == "__main__":
    import json
    qfile, idx, code_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    for line in open(qfile, encoding="utf-8"):
        r = json.loads(line)
        if r.get("index") == idx:
            print(harness_observation(r["question"], open(code_file).read()))
            break

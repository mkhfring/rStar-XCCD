# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
import argparse
import ast
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional, Type, List
from pydantic import BaseModel, Field, root_validator
from timeout_decorator import timeout


TIMEOUT_SECONDS = 30
TIMEOUT_MESSAGE = f"Execution of the code snippet has timed out for exceeding {TIMEOUT_SECONDS} seconds."

# Wall-clock budget for staging Code 2 (javac/rustc + one run), separate from
# TIMEOUT_SECONDS which bounds the model's own python_interpreter calls.
CODE2_TOOL_TIMEOUT = 20

def truncate_string(text, max_length=1024, is_evalf=True):
    # print(text, file=sys.stderr)
    # print(type(text), file=sys.stderr)
    if is_evalf and isinstance(text, str):
        try:
            text_sympy = float(text)
            refine_text = str(round(text_sympy, 4)) + "\n"
            text = text if len(text) < len(refine_text) else refine_text
        except:
            pass

    if len(str(text)) > max_length:
        return str(text)[:max_length//2] + "..." + str(text)[-max_length//2:]
    return text

def extract_content(text):
    pattern = r'print\((.*?)\)'
    matches = re.findall(pattern, text)
    if len(matches) < 1:
        return ""
    return " ".join(matches)+":"


def __is_print_node(node: ast.AST) -> bool:
    
    if isinstance(node, ast.Expr) and \
        isinstance(node.value, ast.Call) and \
        isinstance(node.value.func, ast.Name) and \
        node.value.func.id == "print":
        return True
    elif isinstance(node, ast.If) or \
         isinstance(node, ast.While) or \
         isinstance(node, ast.For) or \
         isinstance(node, ast.FunctionDef):
        for sub_node in node.body:
            if __is_print_node(sub_node):
                return True
    return False


def find_print_node(body: List[ast.AST]) -> List[int]:
    """Find the python print node in the tree.body.

    Args:
        body (List[ast.AST]): The body of the AST

    Returns:
        List[int]: The index of the python print node
    """
    print_index = []
    for idx, node in enumerate(body):
        if __is_print_node(node):
            print_index.append(idx)
    return print_index


def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query


def is_python_code(query: str) -> bool:
    """Check whether a code snippet parses as valid, non-empty Python.

    Used to gate execution: prose, Java snippets, or empty extractions
    should not be handed to the interpreter.
    """
    try:
        tree = ast.parse(query)
    except SyntaxError:
        return False
    return len(tree.body) > 0


CODE2_EXECUTION_TOKEN_RE = re.compile(r"\b(javac?|rustc|cargo|candidate_code2)\b")


def mentions_code2_execution(query: str) -> bool:
    """Heuristic: does a python_interpreter query look like an attempt to
    compile/run Code 2 (Java or Rust)?

    Used only to decide whether it's worth spending a javac/rustc subprocess
    call staging Code 2 for this step -- javac/java and rustc are on PATH on
    this cluster (see stage_code2()), so unlike the pre-2026-09-08 version of
    this check, a match is no longer treated as doomed-to-fail and blocked;
    it just gates the (otherwise wasted, on every one of the many
    Python-only test steps) compile attempt. A plain substring/regex check
    is enough for that: false positives cost one extra compile, false
    negatives just mean the model falls back to static reasoning as before.
    """
    return bool(CODE2_EXECUTION_TOKEN_RE.search(query))


def extract_code1_python(question: str) -> str:
    """Extract the Python source of Code 1 from a clone-detection question.

    Looks for the block introduced by 'Code 1: Python\\n```python\\n' and
    returns everything up to the closing '```'.  Returns an empty string if
    the pattern is not found.
    """
    marker = "Code 1: Python\n```python\n"
    start = question.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = question.find("\n```", start)
    if end == -1:
        return ""
    return question[start:end]


CODE2_MARKERS = {
    "java": "Code 2: Java\n```java\n",
    "rust": "Code 2: Rust\n```rust\n",
}


def extract_code2_source(question: str):
    """Extract Code 2's language and source from a clone-detection question.

    Returns (language, source) with language in {"java", "rust"}, or
    (None, "") if Code 2 isn't in one of those languages, or the expected
    fenced block isn't found (e.g. a different language pair).
    """
    for lang, marker in CODE2_MARKERS.items():
        start = question.find(marker)
        if start == -1:
            continue
        start += len(marker)
        end = question.find("\n```", start)
        if end == -1:
            continue
        return lang, question[start:end]
    return None, ""


JAVA_CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_]\w*)")


def java_class_name(source: str) -> Optional[str]:
    """First top-level class name declared in a Java source, or None.

    Java requires the file to be named after a `public` class, but the
    snippets in this dataset are consistently package-private (`class Foo`,
    no `public`), so any filename compiles -- we still name the file after
    the class purely so `java <ClassName>` matches what a model would
    naturally guess.
    """
    match = JAVA_CLASS_RE.search(source)
    return match.group(1) if match else None


# Rust path/module keywords that can appear as the first segment of a `use`
# without naming an external crate.
RUST_NON_CRATE_PATH_ROOTS = {"std", "core", "alloc", "self", "super", "crate"}
# Not anchored to line start: attributes like `#[macro_use]` routinely share
# a line with the `extern crate`/`use` they annotate.
RUST_USE_CRATE_RE = re.compile(r"\buse\s+([A-Za-z_]\w*)")
RUST_EXTERN_CRATE_RE = re.compile(r"\bextern\s+crate\s+([A-Za-z_]\w*)")


# Char literals first so '"' does not open a string; lifetimes ('a) never
# match because they have no closing quote.
RUST_COMMENT_OR_STRING_RE = re.compile(r"""'(?:\\.|[^'\\])'|"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/""", re.S)
# Items a `use` can legitimately start from without being a crate: local
# modules (`mod io;` / `mod io {`) and local types whose variants/items get
# imported (`enum Node {..}` + `use Node::*;`).
RUST_LOCAL_ITEM_RE = re.compile(r"\b(?:mod|enum|struct|trait|type|union)\s+([A-Za-z_]\w*)")
# Names bound by an earlier `use a::b::name;` or `use a::b::{x, name}` (the
# 2015-edition relative form `use std::io; use io::*;`).
RUST_USE_TAIL_RE = re.compile(r"\buse\s+[\w:]*::(?:\{([^}]*)\}|(\w+))\s*;")

# "legacy" = the original regex over the raw source (every run before
# 2026-09-26, so old and in-flight evals stay comparable); "strict" =
# comments/strings stripped and locally declared names excluded. Carried in
# an environment variable, not just this module global: the Solver executes
# code in a *spawn* ProcessPool, whose workers re-import this module and only
# inherit the environment. main.py calls set_rust_crate_check() from
# config.rust_crate_check before the Solver (and its pool) is created.
RUST_CRATE_CHECK_ENV = "RSTAR_RUST_CRATE_CHECK"
RUST_CRATE_CHECK = os.environ.get(RUST_CRATE_CHECK_ENV, "legacy")


# Ablation switch (EXPERIMENT_PLAN E5 "python-only" arm): when off, Code 2 is
# never staged or compiled, and a step that tries to run it is told so --
# the same fallback the model gets for Rust with unavailable crates. Same
# environment-variable mechanism as RUST_CRATE_CHECK, for the spawn pool.
EXECUTE_CODE2_ENV = "RSTAR_EXECUTE_CODE2"
CODE2_DISABLED_MESSAGE = ("Code 2 cannot be executed in this setting. "
                          "Reason about its expected behavior instead of executing it.")


def execute_code2_enabled() -> bool:
    return os.environ.get(EXECUTE_CODE2_ENV, "1") == "1"


def set_execute_code2(enabled: bool) -> None:
    os.environ[EXECUTE_CODE2_ENV] = "1" if enabled else "0"


def set_rust_crate_check(mode: str) -> None:
    global RUST_CRATE_CHECK
    if mode not in ("legacy", "strict"):
        raise ValueError(f"rust_crate_check must be 'legacy' or 'strict', got {mode!r}")
    RUST_CRATE_CHECK = mode
    os.environ[RUST_CRATE_CHECK_ENV] = mode


def rust_external_crates(source: str, mode: Optional[str] = None) -> set:
    """Non-std crate names a Rust snippet 'use's or pulls in via 'extern crate'.

    These would need `cargo` to fetch from crates.io, which requires network
    access this sandbox's compute nodes don't have (only the cluster's login
    nodes do). A non-empty result means the snippet can't be compiled here
    with plain `rustc`.

    UPDATE 2026-09-26 ("strict" mode): on the 300 distinct Code 2 programs of
    test_python_rust_CLCCD.jsonl, "legacy" refuses 93, of which 4 (16 test
    pairs) compile fine with plain rustc: a `use` inside a comment
    ("// use priority_queue::..", "... use inherent method instead"),
    `use Node::*` for a local `enum Node`, and `use io::*` after
    `use std::io`. "strict" removes exactly those; it cannot catch crates
    used only through full paths (`proconio::input!` with no `use`), which
    fail to compile and reach the model as a compile error instead.
    """
    mode = mode or RUST_CRATE_CHECK
    if mode == "strict":
        source = RUST_COMMENT_OR_STRING_RE.sub(
            lambda m: '""' if m.group(0)[0] in "\"'" else " ", source)
    crates = set(RUST_USE_CRATE_RE.findall(source)) | set(RUST_EXTERN_CRATE_RE.findall(source))
    crates -= RUST_NON_CRATE_PATH_ROOTS
    if mode == "strict":
        crates -= set(RUST_LOCAL_ITEM_RE.findall(source))
        for group, single in RUST_USE_TAIL_RE.findall(source):
            names = group.split(",") if group else [single]
            crates -= {n.strip().split(" as ")[-1].strip() for n in names}
    return crates


# Suffixes of the code/build artifacts a step can leave in the working
# directory: our own staging (candidate_code.py, <Class>.java + the .class
# files javac emits, candidate_code2.rs + its binary) and whatever the model's
# test code writes for itself. The model names those files after the problem
# rather than by any convention -- past runs left pascal.py, dodecagonal.py,
# code_1.py, temp.rs and temp_rust_code.rs sitting in the repo root -- so
# matching on suffix is the only thing that catches them.
#
# The cost of including .py: a .py file that appears in the working directory
# during a step is deleted at the end of it, so don't write scratch scripts
# into the repo root while a run is in flight (a subdirectory is untouched --
# cleanup is non-recursive, and it only ever removes files that were not
# there when the step started).
# .o is here because rustc writes its codegen units as <name>.*.rcgu.o next
# to the source and only removes them once linking finishes -- a compile the
# step timeout kills partway through leaves them behind (three were found in
# the repo root, all stamped inside one run's window).
GENERATED_CODE_SUFFIXES = (".java", ".class", ".rs", ".py", ".o")


def snapshot_dir(workdir: str = ".") -> set:
    """Names currently in workdir, to diff against in cleanup_generated_code()."""
    try:
        return set(os.listdir(workdir))
    except OSError:
        return set()


def cleanup_generated_code(before: set, workdir: str = ".") -> None:
    """Delete generated code/build artifacts that appeared since `before`.

    Called after every execution so the repository doesn't silt up with one
    file per question. Deliberately diff-based rather than a fixed list:
    the model names its own scratch files, so a fixed list would miss them.
    Only regular files matching GENERATED_CODE_SUFFIXES -- or new
    extension-less executables, which is what rustc emits -- are removed;
    directories and everything else new are left untouched.
    """
    for name in snapshot_dir(workdir) - before:
        path = os.path.join(workdir, name)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(name)
        if ext not in GENERATED_CODE_SUFFIXES and not (ext == "" and os.access(path, os.X_OK)):
            continue
        try:
            os.remove(path)
        except OSError:
            # Best effort: a concurrent job on the same working directory may
            # have removed it already. Never let cleanup fail a search step.
            pass


def _compiler_env() -> dict:
    """os.environ minus LD_PRELOAD, for compiler subprocesses.

    The slurm scripts export LD_PRELOAD=libnccl.so.2 so vLLM links against a
    newer NCCL than its bundled PyTorch was built for. rustc inherits that
    when it shells out to link, and the shell it uses for that step is an old
    glibc-2.24 nix build which cannot satisfy libnccl's GLIBC_2.34
    requirement -- so every rustc invocation under a job script died with
    "error: linking with `cc` failed", and no Rust Code 2 has ever compiled
    in a real run. javac is unaffected (it is not a cc wrapper), which is why
    Java worked throughout.

    Dropping the variable for the compiler subprocess only is safe: this
    process already dlopened NCCL at vLLM start-up, so nothing here unloads
    it, and running the compiled binary under LD_PRELOAD is fine -- verified
    for the rust binary, python3 and java alike.
    """
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    return env


def stage_code2(language: Optional[str], source: str, workdir: str = ".") -> str:
    """Write Code 2 to disk and, for java/rust, compile it so the model's own
    subprocess calls (mirroring how it already runs candidate_code.py for
    Code 1) can invoke it directly instead of only reasoning about it.

    Returns a short status string describing what's ready to run, or why
    nothing is: this is meant to be appended to the tool observation so the
    model knows whether to attempt execution or fall back to static
    reasoning, without spending a turn discovering it by trial and error.
    """
    if not source:
        return ""

    if language == "java":
        class_name = java_class_name(source) or "Code2"
        src_path = os.path.join(workdir, f"{class_name}.java")
        with open(src_path, "w") as f:
            f.write(source)
        try:
            compile_proc = subprocess.run(
                ["javac", src_path], cwd=workdir, capture_output=True, text=True,
                env=_compiler_env(),
                timeout=CODE2_TOOL_TIMEOUT,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return f"Code 2 staged at {src_path}, but javac is unavailable: {e}"
        if compile_proc.returncode != 0:
            return (f"Code 2 staged at {src_path}, but `javac {src_path}` failed:\n"
                     f"{compile_proc.stderr.strip()[:500]}")
        return (f"Code 2 (Java) compiled: run it with "
                f"subprocess.run([\"java\", \"-cp\", {workdir!r}, {class_name!r}], "
                f"input=..., capture_output=True, text=True).")

    if language == "rust":
        external = rust_external_crates(source)
        if external:
            return (f"Code 2 uses external crate(s) {sorted(external)}, which this "
                     "offline sandbox cannot fetch (no network on compute nodes). "
                     "Reason about its expected behavior instead of executing it.")
        src_path = os.path.join(workdir, "candidate_code2.rs")
        bin_path = os.path.join(workdir, "candidate_code2")
        with open(src_path, "w") as f:
            f.write(source)
        try:
            compile_proc = subprocess.run(
                ["rustc", src_path, "-o", bin_path], cwd=workdir, capture_output=True,
                text=True, timeout=CODE2_TOOL_TIMEOUT, env=_compiler_env(),
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return f"Code 2 staged at {src_path}, but rustc is unavailable: {e}"
        if compile_proc.returncode != 0:
            return (f"Code 2 staged at {src_path}, but `rustc {src_path}` failed:\n"
                     f"{compile_proc.stderr.strip()[:500]}")
        return (f"Code 2 (Rust) compiled to {bin_path}: run it with "
                f"subprocess.run([{bin_path!r}], input=..., capture_output=True, text=True).")

    return ""


# ---------------------------------------------------------------------------
# Harness-side dual execution (TRAINING_PLAN_2026-09-30, step-1 follow-up).
# Hard-negative pilot: with dual-exec prompts Qwen3-4B defined run_both() but
# actually called Code 2 in only 10% (java) / 26% (rust) of trees. With
# auto_code2 on, every input the model's code feeds to a Python program via
# subprocess.run / subprocess.check_output is recorded; after the step the
# harness runs Code 1 AND Code 2 on those same inputs and appends both outputs
# to the observation. The model still chooses the inputs. Off by default, so
# every existing config behaves as before. Environment variable for the spawn
# pool, like EXECUTE_CODE2_ENV.
AUTO_CODE2_ENV = "RSTAR_AUTO_CODE2"
AUTO_CODE2_MAX_INPUTS = 4
AUTO_CODE2_RUN_TIMEOUT = 10
AUTO_CODE2_MARK = "[Harness]"


def auto_code2_enabled() -> bool:
    return os.environ.get(AUTO_CODE2_ENV, "0") == "1"


def set_auto_code2(enabled: bool) -> None:
    os.environ[AUTO_CODE2_ENV] = "1" if enabled else "0"


class capture_python_inputs:
    """Context manager: record the stdin `input=` of every subprocess.run /
    subprocess.check_output call that starts a Python program (Code 1 or the
    model's copy of it). Calls that run Code 2 themselves are not recorded."""

    def __init__(self):
        self.inputs: List[str] = []

    def _record(self, args, kwargs):
        cmd = args[0] if args else kwargs.get("args")
        cmd_str = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd)
        if "python" not in cmd_str or "candidate_code2" in cmd_str:
            return
        inp = kwargs.get("input")
        if inp is None:
            return
        if isinstance(inp, bytes):
            inp = inp.decode("utf-8", "replace")
        if inp not in self.inputs:
            self.inputs.append(inp)

    def __enter__(self):
        self._run, self._check_output = subprocess.run, subprocess.check_output
        rec, run, chk = self._record, self._run, self._check_output

        def run_patched(*args, **kwargs):
            rec(args, kwargs)
            return run(*args, **kwargs)

        def check_output_patched(*args, **kwargs):
            rec(args, kwargs)
            return chk(*args, **kwargs)
        subprocess.run, subprocess.check_output = run_patched, check_output_patched
        return self

    def __exit__(self, *exc):
        subprocess.run, subprocess.check_output = self._run, self._check_output
        return False


def _short(text: str, n: int = 160) -> str:
    text = text.strip()
    return repr(text if len(text) <= n else text[:n] + "...")


def auto_compare_code2(question: str, inputs: List[str], workdir: str = ".") -> str:
    """Run Code 1 and Code 2 on the same inputs and describe the outputs.
    Returns "" when there is nothing to do; a status line when Code 2 cannot run."""
    inputs = [x for x in inputs if isinstance(x, str)][:AUTO_CODE2_MAX_INPUTS]
    code1 = extract_code1_python(question)
    lang, source = extract_code2_source(question)
    if not inputs or not code1 or not source:
        return ""
    status = stage_code2(lang, source, workdir)
    if lang == "java" and status.startswith("Code 2 (Java) compiled"):
        cmd2 = ["java", "-cp", workdir, java_class_name(source) or "Code2"]
    elif lang == "rust" and status.startswith("Code 2 (Rust) compiled"):
        cmd2 = [os.path.join(workdir, "candidate_code2")]
    else:
        return f"{AUTO_CODE2_MARK} Could not run Code 2 on your test inputs: {status}"
    path1 = os.path.join(workdir, "candidate_code.py")
    with open(path1, "w") as f:
        f.write(code1)

    def _run(cmd, inp):
        try:
            p = subprocess.run(cmd, input=inp, capture_output=True, text=True,
                               timeout=AUTO_CODE2_RUN_TIMEOUT, cwd=workdir)
            return p.returncode, p.stdout
        except subprocess.TimeoutExpired:
            return "timeout", ""
    lines = [f"{AUTO_CODE2_MARK} Ran Code 1 (Python) and Code 2 ({lang.capitalize()}) "
             f"on the same {len(inputs)} input(s) your code used:"]
    for i, inp in enumerate(inputs, 1):
        rc1, o1 = _run([sys.executable, path1], inp)
        rc2, o2 = _run(cmd2, inp)
        ex1 = "" if rc1 == 0 else f" [exit {rc1}]"
        ex2 = "" if rc2 == 0 else f" [exit {rc2}]"
        if rc1 != 0 and rc2 != 0:
            verdict = "BOTH FAILED (the input is probably not valid for these programs)"
        elif rc1 != 0 or rc2 != 0:
            # 2026-10-01: was "DIFFERENT (only one program failed)". On the rust hard
            # pilot most such cases were INVALID model-made inputs (wrong format,
            # overflow) on true clones, which the model then rejected. A one-sided
            # crash is not evidence either way unless the input is known to be valid.
            verdict = ("INCONCLUSIVE (only one program failed; the input may not be "
                       "valid for these programs)")
        else:
            verdict = "SAME" if o1.split() == o2.split() else "DIFFERENT"
        lines.append(f"Input {i}: {_short(inp, 80)} -> Code 1: {_short(o1)}{ex1} | "
                     f"Code 2: {_short(o2)}{ex2} -> {verdict}")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# CODE-STEP MODE (branch codestep-prompt, 2026-10-03). rStar-Math-style prompt in which
# every reasoning step is a Python code step that checks a stated expectation and
# updates is_clone. Harness contract (prompt mcts_prompt_codestep_python_{L}_v1.json):
#   - run_both(stdin_text) is predefined -> (r1, r2), each Res(ok, out, err)
#   - v2 (2026-10-03): same(a, b) is predefined: both ran and the outputs are equal with ALL
#     whitespace ignored (CLCCD's tokenizer splits string literals, e.g. "Player- " vs "Player-");
#     the harness's "outputs differed" count uses the same comparison
#   - v3 (2026-10-03): err(r) is predefined: the informative error line of a failed run (Python: last
#     traceback line; Java: the "Exception in thread" line, skipping the JVM's JAVA_TOOL_OPTIONS notice;
#     Rust: the panic message). Res.err keeps head + tail of stderr so Java's first line survives.
#   - a step runs once; variables persist along ONE trajectory only: earlier code steps of
#     the same path are replayed silently in a fresh namespace before the current step
#   - the model sees only the current step's output, plus a one-line [run_both] summary
# Off by default (RSTAR_CODESTEP env, like the other switches).
import collections as _collections
import contextlib as _contextlib
import hashlib as _hashlib
import io as _io
import tempfile as _tempfile

CODESTEP_ENV = "RSTAR_CODESTEP"
CODESTEP_RUN_TIMEOUT = 10
Res = _collections.namedtuple("Res", "ok out err")
_CODESTEP_BUILDS = {}   # question key -> (cmd1, cmd2 or None, build status)
_CODESTEP_CACHE = {}    # (question key, stdin) -> (Res, Res)


def codestep_enabled() -> bool:
    return os.environ.get(CODESTEP_ENV, "0") == "1"


def set_codestep(enabled: bool) -> None:
    os.environ[CODESTEP_ENV] = "1" if enabled else "0"


def _codestep_build(question: str):
    code1 = extract_code1_python(question)
    lang, source = extract_code2_source(question)
    key = _hashlib.md5((code1 + "\x00" + source).encode()).hexdigest()
    if key not in _CODESTEP_BUILDS:
        workdir = os.path.join(_tempfile.gettempdir(), f"rstar_codestep_{os.getpid()}_{key}")
        os.makedirs(workdir, exist_ok=True)
        path1 = os.path.join(workdir, "candidate_code.py")
        with open(path1, "w") as f:
            f.write(code1)
        status = stage_code2(lang, source, workdir) if source else "Code 2 not found"
        if lang == "java" and status.startswith("Code 2 (Java) compiled"):
            cmd2 = ["java", "-cp", workdir, java_class_name(source) or "Code2"]
        elif lang == "rust" and status.startswith("Code 2 (Rust) compiled"):
            cmd2 = [os.path.join(workdir, "candidate_code2")]
        else:
            cmd2 = None
        _CODESTEP_BUILDS[key] = ([sys.executable, path1], cmd2, status, workdir)
    return key, _CODESTEP_BUILDS[key]


def _codestep_run(cmd, stdin_text, workdir):
    try:
        p = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True,
                           timeout=CODESTEP_RUN_TIMEOUT, cwd=workdir)
        e = p.stderr.strip()
        return Res(p.returncode == 0, p.stdout.strip(), e if len(e) <= 2000 else e[:1000] + "\n...\n" + e[-1000:])
    except subprocess.TimeoutExpired:
        return Res(False, "", "timeout")


def _norm_out(text):
    return "".join(str(text).split())


_ERR_LINE_RE = re.compile(r"(Error|Exception|panicked|called `|timeout)")


def err(r):
    """The informative error line of a failed run ("" if it ran without error)."""
    if r.ok:
        return ""
    lines = [l.strip() for l in (r.err or "").splitlines()
             if l.strip() and not l.startswith("Picked up") and not l.strip().startswith("at ")
             and not l.startswith("note: run with")]
    hits = [l for l in lines if _ERR_LINE_RE.search(l)]
    pick = (hits[0] if hits and ("Exception in thread" in hits[0] or "panicked" in hits[0]) else
            hits[-1] if hits else (lines[-1] if lines else "failed (no error message)"))
    if "panicked" in pick and lines.index(pick) + 1 < len(lines):   # rust: message is on the next line
        pick = pick.split(" at ")[0] + ": " + lines[lines.index(pick) + 1]
    return pick[:160]


def same(a, b):
    """True if both runs succeeded and printed the same output, ignoring all whitespace."""
    return bool(a.ok and b.ok and _norm_out(a.out) == _norm_out(b.out))


def make_run_both(question: str, log: list):
    key, (cmd1, cmd2, status, workdir) = _codestep_build(question)

    def run_both(stdin_text):
        stdin_text = str(stdin_text)
        if (key, stdin_text) not in _CODESTEP_CACHE:
            r1 = _codestep_run(cmd1, stdin_text, workdir)
            r2 = _codestep_run(cmd2, stdin_text, workdir) if cmd2 else Res(False, "", status[:300])
            _CODESTEP_CACHE[(key, stdin_text)] = (r1, r2)
        r1, r2 = _CODESTEP_CACHE[(key, stdin_text)]
        log.append(dict(stdin=stdin_text, ok1=r1.ok, ok2=r2.ok,
                        differ=r1.ok and r2.ok and not same(r1, r2)))
        return r1, r2
    return run_both, (cmd2 is not None), status


def codestep_summary(calls, built=True, status=""):
    if not calls:
        return ""
    both = sum(c["ok1"] and c["ok2"] for c in calls)
    one = sum(c["ok1"] != c["ok2"] for c in calls)
    none = sum(not c["ok1"] and not c["ok2"] for c in calls)
    diff = sum(c["differ"] for c in calls)
    line = (f"[run_both] {len(calls)} call(s): both ran {both}, only one failed {one}, "
            f"both failed {none}, outputs differed {diff}")
    if not built:
        line += f" | Code 2 could not be built: {status[:160]}"
    return line


def run_codestep(question: str, previous_steps, current_step: str) -> str:
    """Execute one code step of a code-step trajectory and return its observation."""
    log = []
    run_both, built, status = make_run_both(question, log)
    ns = {"__name__": "__main__", "run_both": run_both, "same": same, "err": err, "Res": Res}
    for prev in previous_steps:            # replay this trajectory's earlier steps silently
        try:
            with _contextlib.redirect_stdout(_io.StringIO()):
                exec(prev, ns)
        except BaseException:              # an earlier step may have failed; keep going
            pass
    n0 = len(log)
    buf = _io.StringIO()
    step_err = ""
    try:
        with _contextlib.redirect_stdout(buf):
            exec(current_step, ns)
    except BaseException as e:             # incl. SystemExit from model code
        step_err = "{}: {}".format(type(e).__name__, str(e))
    out = truncate_string(buf.getvalue(), max_length=1024).strip()
    parts = [p for p in (out, step_err, codestep_summary(log[n0:], built, status)) if p]
    return "\n".join(parts)

class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")
    

class PythonInterpreter(BaseModel):
    """A tool for running python code snippet."""

    name: str = "python_interpreter"
    description: str = (
        "A Python shell. Use this to execute python commands. "
    )
    description_zh: str = (
        "Python 交互式 shell。使用此工具来执行 Python 代码。"
    )
    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)
    sanitize_input: bool = True
    max_length: int = 1024
    is_evalf: bool = True
    args_schema: Type[BaseModel] = PythonInputs
    use_signals: bool = False 

    def _base_run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        def _sub_run(bodys):
            io_buffer = StringIO()
            module = ast.Module(bodys[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(bodys[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore

            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, self.globals, self.locals)
                    if ret is None:
                        return True, truncate_string(io_buffer.getvalue(), max_length=self.max_length, is_evalf=self.is_evalf)
                    else:
                        return True, truncate_string(ret, max_length=self.max_length, is_evalf=self.is_evalf)
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return False, truncate_string(io_buffer.getvalue(), max_length=self.max_length, is_evalf=self.is_evalf)
            
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
            tree = ast.parse(query)
            print_indexs = find_print_node(tree.body)
            if len(print_indexs) == 0:
                print_indexs = [len(tree.body) - 1]
            ret_strs = []
            if len(print_indexs) == 1:
                run_flag, ret = _sub_run(tree.body)
                return f"{ret}"
            for start_idx, end_idx in zip([-1] + print_indexs, print_indexs):
                node_source = ast.get_source_segment(query, tree.body[end_idx])
                run_flag, ret = _sub_run(tree.body[start_idx + 1:end_idx + 1])
                ret_strs.append(f"{extract_content(node_source)} {ret}")
            return "".join(ret_strs)
        except (Exception, SystemExit) as e:
            # SystemExit (2026-10-01): model code runs in THIS process, so a step that
            # calls exit()/sys.exit()/quit() used to terminate the whole search job
            # silently with status 0 (SCB seed-1 java run stopped at pair 740). Report
            # it as the step's output, as a separate Python process would have exited.
            return "{}: {}".format(type(e).__name__, str(e))
    
    def run(
        self,
        query: str,
    ) -> str:

        @timeout(TIMEOUT_SECONDS, use_signals=True, exception_message=TIMEOUT_MESSAGE)
        def base_run(query: str) -> str:
            return self._base_run(query)
        
        try:
            ret = base_run(query)
            return ret
        except Exception as e:
            print(e)
            print(" exec code error ")
            return "{}: {}".format(type(e).__name__, str(e))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--testcase', type=str, default="```python\nprint(1)\nprint(2)\n```")
    # input args
    args = args.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    tool = PythonInterpreter()
    print(tool.run(args.testcase))
    sys.exit(0)
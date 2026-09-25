"""Differential testing of clone pairs: run Code 1 (Python) and Code 2
(Rust/Java) on the SAME stdin inputs and compare what they print.

This is an execution-based check that does not depend on the model or on a
human/LLM judge (SFT_DATA_PROCESS.txt section 9). It is used three ways:
  1. label audit      -- a "clone" pair whose programs disagree on a valid
                          input, or a "non-clone" pair that agrees on every
                          input, is flagged for review (CodeNet labels are
                          derived from problem ids, so this catches noise);
  2. evidence audit   -- the model's own test inputs (the input="..." strings
                          in its subprocess calls) are replayed on Code 2, so
                          a trace's asserted expected outputs can be checked
                          against what Code 2 actually prints;
  3. repair / teacher -- gives the verified inputs used when a trace's code
                          step is rewritten (claude_review/, section 9).

Inputs per question come from (a) every input="..." literal the model wrote
anywhere in that question's mining trees and (b) an optional hand-written
file (--manual_inputs, {index: [input, ...]}).

Output: one JSON record per question with per-input outputs and a verdict:
  agree_all       -- every input ran on both sides and outputs matched
  disagree        -- >=1 input where both ran cleanly but outputs differ
  no_valid_inputs -- nothing ran cleanly on both sides
  code2_unrunnable-- Code 2 did not compile / uses external crates

Usage:
    python diff_test_pairs.py --questions eval_data/failed_codenet_python_rust_clean.jsonl \
        --trees <mining output> [<mining output> ...] \
        [--manual_inputs eval_data/claude_review/manual_inputs_rust.json] \
        --out eval_data/claude_review/diff_test_rust.jsonl
"""
import argparse
import ast
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_exec import compile_code2, run_program
from rstar_deepthink.tools.python_tool import extract_code1_python

INPUT_LITERAL_RE = re.compile(r"""input\s*=\s*((?:[rbuf]?)(?:"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'))""")
MAX_INPUTS = 40


def norm(out: str) -> str:
    return "\n".join(line.rstrip() for line in out.strip().splitlines())


def model_inputs(trees):
    found = []
    for tree in trees:
        for node in tree.values():
            if not isinstance(node, dict):
                continue
            for lit in INPUT_LITERAL_RE.findall(node.get("action_input") or ""):
                try:
                    val = ast.literal_eval(lit)
                except Exception:
                    continue
                if isinstance(val, bytes):
                    val = val.decode(errors="replace")
                if isinstance(val, str) and val not in found:
                    found.append(val)
    return found


def test_question(question, inputs):
    with tempfile.TemporaryDirectory(prefix="difftest_") as d:
        code1 = extract_code1_python(question)
        py_path = Path(d) / "candidate_code.py"
        py_path.write_text(code1)
        argv2, status = compile_code2(question, d)
        if argv2 is None:
            return {"verdict": "code2_unrunnable", "code2_status": status[:300], "cases": []}
        cases = []
        for inp in inputs[:MAX_INPUTS]:
            feed = inp if inp.endswith("\n") else inp + "\n"
            o1, e1, rc1 = run_program(["python3", str(py_path)], feed)
            o2, e2, rc2 = run_program(argv2, feed)
            both_ok = rc1 == 0 and rc2 == 0
            cases.append({
                "input": inp, "py_out": o1[-2000:], "code2_out": o2[-2000:],
                "py_rc": rc1, "code2_rc": rc2,
                "valid": both_ok, "match": both_ok and norm(o1) == norm(o2),
            })
    valid = [c for c in cases if c["valid"]]
    if not valid:
        verdict = "no_valid_inputs"
    elif all(c["match"] for c in valid):
        verdict = "agree_all"
    else:
        verdict = "disagree"
    return {"verdict": verdict, "code2_status": status[:300], "cases": cases}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--trees", nargs="*", default=[])
    ap.add_argument("--manual_inputs")
    ap.add_argument("--indices", help="comma-separated subset of indices")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trees_by_idx = {}
    for path in args.trees:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if "rstar" in r:
                    trees_by_idx.setdefault(r["index"], []).append(r["rstar"])
    manual = {}
    if args.manual_inputs:
        manual = {int(k): v for k, v in json.load(open(args.manual_inputs)).items()}
    only = {int(x) for x in args.indices.split(",")} if args.indices else None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for line in open(args.questions, encoding="utf-8"):
            r = json.loads(line)
            idx = r["index"]
            if only is not None and idx not in only:
                continue
            inputs = manual.get(idx, []) + [i for i in model_inputs(trees_by_idx.get(idx, []))
                                            if i not in manual.get(idx, [])]
            res = test_question(r["question"], inputs)
            n_valid = sum(c["valid"] for c in res["cases"])
            n_match = sum(c["match"] for c in res["cases"])
            print(f"{idx:5d} {r['answer']:9s} {res['verdict']:16s} valid={n_valid:2d}/{len(res['cases']):2d} "
                  f"match={n_match:2d}  manual={len(manual.get(idx, []))}")
            w.write(json.dumps({"index": idx, "answer": r["answer"], **res}) + "\n")


if __name__ == "__main__":
    main()

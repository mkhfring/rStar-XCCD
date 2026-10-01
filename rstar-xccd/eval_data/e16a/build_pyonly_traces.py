"""E16a step 2: assemble Claude-written traces for the PYTHON-ONLY method
(EXPERIMENT_PLAN E16a). Every execution result in a trace is REAL.

Division of labour (same principle as build_teacher_traces.py):
  * analysis prose is written by hand per pair after reading both programs
    (--analyses {index: {"c1", "c2", "cmp", "reason", "inputs": [...]}}),
    where "inputs" are extra hand-written test inputs;
  * test cases = the pair's official CodeNet sample inputs (batch file,
    select_and_verify.py) + the hand-written inputs, all re-run here on BOTH
    programs offline (diff_test_pairs.test_question);
  * the "expected Code 2 output" in each case is what Code 2 ACTUALLY printed
    offline -- the trace presents it as the result of tracing Code 2 by hand,
    which is what the python-only method asks the model to do;
  * the <output> block is harness_exec.harness_observation() with Code 2
    execution disabled, i.e. exactly what the python-only arm returns.

Code step: runs ONLY Code 1 (candidate_code.py) and compares with Code 2's
expected output. Clone traces need >=2 matching cases and no mismatch;
non-clone traces need >=1 case where Code 2 prints something different
(preferred) or cannot process an input Code 1 accepts; every printed line
must then be False. The code never mentions how to run Code 2 (the harness
would otherwise try to stage it).

Usage:
  python eval_data/e16a/build_pyonly_traces.py --batch eval_data/e16a/batch_pilot_rust.jsonl \
      --analyses eval_data/e16a/analyses_pilot_rust.json --out eval_data/e16a/traces_pilot_rust.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diff_test_pairs import norm, test_question  # noqa: E402
from harness_exec import harness_observation  # noqa: E402
from rstar_deepthink.tools.python_tool import mentions_code2_execution, set_execute_code2  # noqa: E402

MAX_CASES = 4
MAX_INPUT_CHARS = 300
MAX_OUTPUT_CHARS = 80
ERROR_NOTE = "Code 2 cannot process this input"

HEADER = (
    "# The Python implementation extracted from Code 1 has been saved as candidate_code.py.\n"
    "# Code 2 cannot be executed in this setting, so each case lists the output Code 2 is\n"
    "# expected to print, obtained by tracing Code 2 by hand on that input.\n"
    "import subprocess\n\n"
)


def short(c):
    return (c["input"].strip() != "" and len(c["input"]) <= MAX_INPUT_CHARS and len(c["py_out"].strip()) <= MAX_OUTPUT_CHARS
            and len(c["code2_out"].strip()) <= MAX_OUTPUT_CHARS and c["py_rc"] == 0)


def pick(cases, want):
    """Up to MAX_CASES cases of the wanted kind, earliest first, preferring
    distinct inputs and distinct Code 1 outputs."""
    if want == "clone":
        good = [c for c in cases if c["valid"] and c["match"] and short(c)]
    else:
        differ = [c for c in cases if c["valid"] and not c["match"] and short(c)]
        error = [c for c in cases if short(c) and c["code2_rc"] not in (0, "timeout")]
        good = differ + error
    seen_in, seen_out, out = set(), set(), []
    for prefer_new in (True, False):
        for c in good:
            k, o = c["input"].strip(), c["py_out"].strip()
            if k in seen_in or (prefer_new and o in seen_out):
                continue
            seen_in.add(k)
            seen_out.add(o)
            out.append(c)
            if len(out) == MAX_CASES:
                return out
    return out


def code_for(cases):
    rows = []
    for c in cases:
        exp = norm(c["code2_out"]) if c["code2_rc"] == 0 else None
        note = f"  # {ERROR_NOTE}" if exp is None else ""
        rows.append(f"    ({c['input']!r}, {exp!r}),{note}")
    return (HEADER + "cases = [\n" + "\n".join(rows) + "\n]\n"
            "for t, code2_expected in cases:\n"
            "    py = subprocess.run([\"python3\", \"candidate_code.py\"], input=t, capture_output=True, text=True).stdout.strip()\n"
            "    if code2_expected is None:\n"
            f"        print(False, repr(py), {ERROR_NOTE!r})\n"
            "    else:\n"
            "        print(py == code2_expected, repr(py), repr(code2_expected))\n")


def verdicts(observation, n):
    lines = [l for l in observation.splitlines() if l.startswith(("True", "False"))]
    return [l.startswith("True") for l in lines] if len(lines) == n else None


def assemble(a, code, obs, label):
    return (
        "<analysis>\n"
        f"{a['c1'].strip()}<end_of_step>\n\n"
        f"{a['c2'].strip()}<end_of_step>\n\n"
        f"{a['cmp'].strip()}<end_of_step>\n"
        "<end_of_analysis>\n\n"
        f"<code>\n{code}<end_of_code>\n"
        f"<output>\n{obs}\n<end_of_output><answer>\n"
        f"Final decision: \\boxed{{{label}}}\n"
        f"Reason: {a['reason'].strip()}\n"
        "<end_of_answer>"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--analyses", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    set_execute_code2(False)

    analyses = {int(k): v for k, v in json.load(open(args.analyses)).items()}
    n_ok = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for line in open(args.batch, encoding="utf-8"):
            q = json.loads(line)
            idx, label = q["index"], q["answer"]
            a = analyses.get(idx)
            if not a or "c1" not in a:
                continue
            sample_inputs = [c["input"] for c in q["cases"]]
            extra = [i for i in a.get("inputs", []) if i not in sample_inputs]
            cases = test_question(q["question"], sample_inputs + extra)["cases"]
            if label == "clone" and any(c["valid"] and not c["match"] for c in cases):
                print(f"{idx}: SKIP clone pair disagrees on a valid input -- check label/inputs")
                continue
            chosen = pick(cases, label)
            if len(chosen) < (2 if label == "clone" else 1):
                print(f"{idx}: SKIP only {len(chosen)} usable {label} case(s); add inputs")
                continue
            code = code_for(chosen)
            assert not mentions_code2_execution(code), idx
            obs = harness_observation(q["question"], code)
            v = verdicts(obs, len(chosen))
            if v is None or (label == "clone" and not all(v)) or (label == "non-clone" and any(v)):
                print(f"{idx}: SKIP harness observation does not support {label}:\n{obs[:300]}")
                continue
            w.write(json.dumps({
                "question": q["question"], "answer": label, "source_file": args.batch, "index": idx,
                "p_id1": q.get("p_id1"), "p_id2": q.get("p_id2"), "leaf_tag": "teacher",
                "origin": "teacher", "style": "pyonly", "target_text": assemble(a, code, obs, label),
            }) + "\n")
            n_ok += 1
            print(f"{idx}: ok ({label}, {len(chosen)} cases)")
    print(f"wrote {n_ok} traces to {args.out}")


if __name__ == "__main__":
    main()

"""Assemble teacher-written SFT traces (SFT_DATA_PROCESS.txt section 9) whose
every execution result is REAL.

Division of labour:
  * the analysis prose (what each program does, how they compare, the
    answer's reason) is written by hand, per question, in a JSON file
    (--analyses {index: {"c1", "c2", "cmp", "reason"}}), after reading
    both programs;
  * the test inputs are the analysis entry's optional "inputs" (official
    sample inputs, re-run on both programs here) followed by
    diff_test_pairs.py's cases -- only inputs on which Code 1 and Code 2 BOTH
    ran cleanly and printed the SAME thing are ever used;
  * the <output> block is produced by harness_exec.harness_observation(),
    i.e. the same staging + interpreter code path the MCTS harness uses.
    Nothing in an <output> block is ever typed by hand.

Two code-step styles, tried in order:
  run_both    -- run candidate_code.py and ./candidate_code2 (the harness
                 compiles Code 2 when a step mentions it) on each input and
                 print whether stdout matches. Used when the harness itself
                 can compile Code 2.
  python_only -- when the harness cannot run Code 2 (external crates such
                 as proconio, or the rust_external_crates() false positives):
                 run Code 1 only and compare against Code 2's output for
                 each input. Those expected outputs were verified by running
                 Code 2 offline (diff_test_pairs.py); the trace presents them
                 as what Code 2 prints, which is true.
A trace is kept only if the real observation reports a match on every case.

REPAIR MODE (--repairs REVIEW.jsonl): for every review row with
verdict "repair", the model's OWN leaf is kept verbatim up to and including
<end_of_analysis> and from <answer> on; only the code step in between (which
printed nothing) is replaced by a verified test and its real observation.
The analysis entry for that question only needs "inputs". These records get
"origin": "repair" and keep the original leaf_tag/source_file.

Output records match score_rl_rollouts.py --emit_sft, plus
"origin": "teacher" and "style", so build_full_prompts.py consumes them
unchanged and they stay separable from the model's own leaves.

Usage:
    python build_teacher_traces.py --questions Q.jsonl --diff_test D.jsonl \
        --analyses A.json --out OUT.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_exec import harness_observation
from diff_test_pairs import test_question

MAX_CASES = 4
MAX_INPUT_CHARS = 300
MAX_OUTPUT_CHARS = 80


def pick_cases(cases):
    """Up to MAX_CASES verified-matching cases, earliest first (hand-picked
    sample inputs come first), preferring cases whose output differs from
    those already picked so the test actually discriminates."""
    good = [c for c in cases if c["valid"] and c["match"]
            and len(c["input"]) <= MAX_INPUT_CHARS and len(c["py_out"].strip()) <= MAX_OUTPUT_CHARS]
    seen_in, seen_out, out = set(), set(), []
    for prefer_new_output in (True, False):
        for c in good:
            key, o = c["input"].strip(), c["py_out"].strip()
            if key in seen_in or (prefer_new_output and o in seen_out):
                continue
            seen_in.add(key)
            seen_out.add(o)
            out.append(c)
            if len(out) == MAX_CASES:
                return out
    return out


def code_run_both(cases):
    tests = ",\n    ".join(repr(c["input"]) for c in cases)
    return (
        "# The Python implementation extracted from Code 1 has been saved as candidate_code.py.\n"
        "# Code 2 is compiled by the sandbox to ./candidate_code2. Run both programs on the same\n"
        "# valid inputs and compare what they print.\n"
        "import subprocess\n\n"
        f"tests = [\n    {tests},\n]\n"
        "for t in tests:\n"
        "    py = subprocess.run([\"python3\", \"candidate_code.py\"], input=t, capture_output=True, text=True).stdout.strip()\n"
        "    rs = subprocess.run([\"./candidate_code2\"], input=t, capture_output=True, text=True).stdout.strip()\n"
        "    print(py == rs, repr(py), repr(rs))\n"
    )


def code_python_only(cases):
    rows = ",\n    ".join(f"({c['input']!r}, {c['code2_out'].strip()!r})" for c in cases)
    return (
        "# The Python implementation extracted from Code 1 has been saved as candidate_code.py.\n"
        "# Code 2 cannot be executed in this sandbox, so compare Code 1's output with what\n"
        "# Code 2 prints for each of these inputs.\n"
        "import subprocess\n\n"
        f"cases = [\n    {rows},\n]\n"
        "for t, code2_output in cases:\n"
        "    py = subprocess.run([\"python3\", \"candidate_code.py\"], input=t, capture_output=True, text=True).stdout.strip()\n"
        "    print(py == code2_output, repr(py))\n"
    )


def all_matched(observation, n_cases):
    lines = [l for l in observation.splitlines() if l.startswith(("True", "False"))]
    return len(lines) == n_cases and all(l.startswith("True") for l in lines)


def assemble(a, code, observation):
    return (
        "<analysis>\n"
        f"{a['c1'].strip()}<end_of_step>\n\n"
        f"{a['c2'].strip()}<end_of_step>\n\n"
        f"{a['cmp'].strip()}<end_of_step>\n"
        "<end_of_analysis>\n\n"
        f"<code>\n{code}<end_of_code>\n"
        f"<output>\n{observation}\n<end_of_output><answer>\n"
        "Final decision: \\boxed{clone}\n"
        f"Reason: {a['reason'].strip()}\n"
        "<end_of_answer>"
    )


def verified_block(question, a, diff_record):
    """(style, code, observation) for a verified code step, or None."""
    pool = []
    if a.get("inputs"):  # hand-picked (official sample) inputs, verified right here
        pool += test_question(question, a["inputs"])["cases"]
    n_hand = len(pick_cases(pool))
    if a.get("inputs") and n_hand < 2:
        return None
    if n_hand < MAX_CASES:
        pool += (diff_record or {}).get("cases", [])
    cases = pick_cases(pool)
    if len(cases) < 2:
        return None
    for style, builder in (("run_both", code_run_both), ("python_only", code_python_only)):
        code = builder(cases)
        obs = harness_observation(question, code)
        if all_matched(obs, len(cases)):
            return style, code, obs
    return None


def repair_leaf(original, block):
    style, code, obs = block
    head, sep, _ = original.partition("<end_of_analysis>")
    ans = original[original.find("<answer>"):]
    if not sep or not ans.startswith("<answer>"):
        return None
    return f"{head}<end_of_analysis>\n\n<code>\n{code}<end_of_code>\n<output>\n{obs}\n<end_of_output>{ans}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repairs", help="leaf review JSONL; rows with verdict 'repair' are repaired")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--diff_test", required=True, action="append")
    ap.add_argument("--analyses", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    analyses = {int(k): v for k, v in json.load(open(args.analyses)).items()}
    diffs = {}
    for p in args.diff_test:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            diffs[r["index"]] = r
    n_ok = 0
    with open(args.out, "w", encoding="utf-8") as w:
        if args.repairs:
            from evaluate_clone_results import node_sort_key
            from score_rl_rollouts import build_target_text
            qs = {json.loads(l)["index"]: json.loads(l) for l in open(args.questions, encoding="utf-8")}
            trees = {}
            for line in open(args.repairs, encoding="utf-8"):
                row = json.loads(line)
                if row["verdict"] != "repair":
                    continue
                src, idx = row["source_file"], row["index"]
                if src not in trees:
                    trees[src] = {json.loads(l)["index"]: json.loads(l) for l in open(src) if '"rstar"' in l}
                tree = trees[src][idx]["rstar"]
                numeric = {k: v for k, v in tree.items() if node_sort_key(k) is not None}
                block = verified_block(qs[idx]["question"], analyses.get(idx, {}), diffs.get(idx))
                fixed = repair_leaf(build_target_text(tree, row["leaf_tag"], numeric), block) if block else None
                if fixed is None:
                    print(f"repair {idx} {row['leaf_tag']}: skipped (no verified block or unexpected leaf shape)")
                    continue
                w.write(json.dumps({
                    "question": qs[idx]["question"], "answer": qs[idx]["answer"], "source_file": src,
                    "index": idx, "leaf_tag": row["leaf_tag"], "origin": "repair", "style": block[0],
                    "target_text": fixed,
                }) + "\n")
                n_ok += 1
                print(f"repair {idx} {row['leaf_tag']}: ok ({block[0]})")
        for line in open(args.questions, encoding="utf-8"):
            q = json.loads(line)
            idx = q["index"]
            if idx not in analyses or "c1" not in analyses[idx]:
                continue
            if q["answer"] != "clone":
                print(f"{idx}: skipped, teacher traces are only written for clone pairs")
                continue
            a = analyses[idx]
            pool = []
            if a.get("inputs"):  # hand-picked (official sample) inputs, verified right here
                pool += test_question(q["question"], a["inputs"])["cases"]
            n_hand = len(pick_cases(pool))
            if a.get("inputs") and n_hand < 2:
                print(f"{idx}: skipped, <2 hand-picked inputs verified on both programs")
                continue
            if n_hand < MAX_CASES:
                pool += diffs.get(idx, {}).get("cases", [])
            cases = pick_cases(pool)
            if len(cases) < 2:
                print(f"{idx}: skipped, <2 verified matching inputs")
                continue
            kept = None
            for style, builder in (("run_both", code_run_both), ("python_only", code_python_only)):
                code = builder(cases)
                obs = harness_observation(q["question"], code)
                if all_matched(obs, len(cases)):
                    kept = (style, code, obs)
                    break
            if kept is None:
                print(f"{idx}: skipped, harness observation did not confirm all cases:\n{obs[:300]}")
                continue
            style, code, obs = kept
            w.write(json.dumps({
                "question": q["question"], "answer": q["answer"], "source_file": args.questions,
                "index": idx, "leaf_tag": "teacher", "origin": "teacher", "style": style,
                "target_text": assemble(analyses[idx], code, obs),
            }) + "\n")
            n_ok += 1
            print(f"{idx}: ok ({style}, {len(cases)} cases)")
    print(f"wrote {n_ok} teacher traces to {args.out}")


if __name__ == "__main__":
    main()

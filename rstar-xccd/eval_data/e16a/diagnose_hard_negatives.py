"""Why does SCB call same-problem buggy programs clones? (hard-negative pilot, option 2)

For every trajectory (root -> leaf with a final answer) of the SCB trees on the
hard-negative set, classify what the model did, then aggregate per pair kind
(hn_output = same-problem Wrong Answer, clone, cross). Python-only setup: the
harness runs only Code 1, so a test can expose a Code-2 bug only if the model
compares Code 1's real output with a Code-2 output it predicts itself.

Per trajectory:
  no_code        no <code> step (decided from reading only)
  print_only     only the print("not code clones") shortcut
  hollow         code step(s) but none runs Code 1 on an input with a non-empty,
                 error-free output (e.g. just `import subprocess`)
  errored        a run of Code 1 crashed (Traceback / Error in the output)
  ran            Code 1 ran on >=1 input and printed something
    compared     ... and the code compares against an expected / Code-2 value
                 (assert, ==, 'expected', 'Code 2', 'match')
    sample       ... and one of its inputs is the official sample input on which
                 the Wrong Answer program is known to differ
    mismatch     ... and the observed output reports a mismatch (False, mismatch,
                 differ, not equal, FAIL)
  verdict        the leaf's final answer (clone / non-clone)
Dual-execution flags (--arm dualexec): crate_blocked (Rust needs a crate the
  offline sandbox lacks), code2_called (Code 2 was actually invoked on an input,
  not only defined), code2_mismatch (an assert/compare of the two outputs failed),
  code2_agree (Code 2 ran without error; NOT necessarily equal outputs).
auto_code2 flags (--arm autocode2): harness_compared (the harness ran both programs on
  the step's inputs), harness_different (it reported a clean output difference),
  harness_one_failed (only one program failed on an input). The leaf lines
  "harness_different->clone" = the model saw a difference and still said clone.
Tree level: a tree counts in a category if ANY of its trajectories does.

Usage: python eval_data/e16a/diagnose_hard_negatives.py --lang java
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_clone_results import node_sort_key, normalize_label  # noqa: E402
from rescore_runs import RULES  # noqa: E402

E = "eval_data/e16a"
CODE = re.compile(r"<code>(.*?)(?:<end_of_code>|<end_of_step>)", re.S)
OUT = re.compile(r"<output>(.*?)<end_of_output>", re.S)
RUNS = re.compile(r"subprocess\.(run|Popen|check_output)|candidate_code|os\.system")
COMPARE = re.compile(r"assert|==|expected|Code 2|code2|match", re.I)
MISMATCH = re.compile(r"\bFalse\b|mismatch|differ|not equal|FAIL|AssertionError", re.I)
ERROR = re.compile(r"Traceback|Error\b|error:", re.I)
PRINT_ONLY = re.compile(r"^\s*print\(\s*['\"]not code clones['\"]\s*\)\s*$", re.I)


def norm(s):
    return " ".join(s.replace("\\n", " ").split())


def paths(tree):
    numeric = {k: v for k, v in tree.items() if node_sort_key(k) is not None}
    for tag, n in numeric.items():
        fa = (n.get("final_answer") or "").strip()
        lab = normalize_label(fa) if fa else None
        if lab is None:
            continue
        parts = tag.split(".")
        chain = [numeric[".".join(parts[:i])] for i in range(1, len(parts) + 1) if ".".join(parts[:i]) in numeric]
        yield lab, "\n".join(c.get("text", "") for c in chain)


def code_text(text):
    """All code the trajectory wrote: from the first <code> on, outputs/answer removed.
    (Code can continue over several <end_of_step> chunks before <end_of_code>.)"""
    i = text.find("<code>")
    if i < 0:
        return None
    t = OUT.sub("", text[i:])
    t = re.sub(r"<answer>.*", "", t, flags=re.S)
    return re.sub(r"<code>|<end_of_code>|<end_of_step>", "\n", t)


def classify(text, sample):
    code_all = code_text(text)
    outs = OUT.findall(text)
    if code_all is None:
        return "no_code", set()
    stripped = "\n".join(l for l in (re.sub(r"#.*", "", x).strip() for x in code_all.splitlines()) if l)
    if not stripped:
        return "no_code", set()  # empty <code> block, straight to the answer
    if all(PRINT_ONLY.match(l) for l in stripped.splitlines()):
        return "print_only", set()
    flags = set()
    # dual execution: did the code actually CALL Code 2 (not just define run_both)?
    calls = code_all.count("run_both(") - code_all.count("def run_both(")
    direct = len(re.findall(r"candidate_code2|['\"]Main['\"]", code_all)) - code_all.count("def run_both(")
    # harness-side dual execution (auto_code2 arm): the observation carries both outputs
    harness = [l for o in outs for l in o.splitlines() if l.startswith("Input ") and " -> Code 1: " in l]
    if harness:
        flags.add("harness_compared")
        if any(l.endswith("-> DIFFERENT") for l in harness):
            flags.add("harness_different")
        elif any("DIFFERENT (only one" in l or "INCONCLUSIVE (only one" in l for l in harness):
            flags.add("harness_one_failed")
    if any("external crate" in o for o in outs):
        flags.add("crate_blocked")
    if calls > 0 or direct > 0:
        flags.add("code2_called")
        if any(re.search(r"AssertionError", o) or MISMATCH.search(o) for o in outs):
            flags.add("code2_mismatch")
        elif any(o.strip() and not ERROR.search(o) and "external crate" not in o for o in outs):
            flags.add("code2_agree")
    ran = RUNS.search(code_all) and any(o.strip() and not ERROR.search(o) for o in outs)
    if any(ERROR.search(o) for o in outs):
        flags.add("errored")
    if not ran:
        return ("errored" if "errored" in flags else "hollow"), flags
    if COMPARE.search(code_all):
        flags.add("compared")
    if sample and norm(sample) and norm(sample) in norm(code_all):
        flags.add("sample")
    if any(MISMATCH.search(o) for o in outs):
        flags.add("mismatch")
    return "ran", flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["java", "rust"])
    ap.add_argument("--arm", default="pyonly",
                    help="branch suffix: pyonly, dualexec, autocode2, dualexec-fix, autocode2-fix")
    ap.add_argument("--set", default="hard", choices=["hard", "hardeval"],
                    help="hard = 250-pair pilot (dev); hardeval = LOCKED evaluation set")
    args = ap.parse_args()
    L = args.lang
    meta = {m["index"]: m for m in map(json.loads, open(f"{E}/{args.set}_python_{L}_codenet_meta.jsonl"))
            if not m.get("multi_answer_suspect")}
    run = sorted(glob.glob(f"{E}/{args.set}_python_{L}_codenet_depth_16.jsonl.mcts.Qwen3-4B.hard-{args.arm}.*.jsonl"))[-1]
    samples = json.load(open(f"{E}/codenet_samples.json"))
    leaf = {k: Counter() for k in ("hn_output", "clone", "cross")}
    tree = {k: Counter() for k in ("hn_output", "clone", "cross")}
    n = Counter()
    examples = []
    for r in map(json.loads, open(run)):
        m = meta.get(r["index"])
        if not m or "rstar" not in r:
            continue
        kind = m["kind"]
        n[kind] += 1
        sample = m.get("input") or next((x for x in samples.get(m["p_id1"], []) if x.strip()), "")
        seen = set()
        for lab, text in paths(r["rstar"]):
            cat, flags = classify(text, sample)
            leaf[kind][f"{cat}->{lab}"] += 1
            leaf[kind]["TOTAL"] += 1
            for f in flags:
                leaf[kind][f"  {f}->{lab}"] += 1
            seen |= {cat} | flags | {f"{cat}->{lab}"}
            if kind == "hn_output" and cat == "ran" and "mismatch" in flags and lab == "clone":
                examples.append(m["p_id1"])
        for s in seen:
            tree[kind][s] += 1
        tree[kind]["pred_tested:" + str(RULES["tested"](r["rstar"]))] += 1
    print(f"== {L}  ({run.split('/')[-1][-40:]})  trees: {dict(n)}")
    for kind in ("hn_output", "clone", "cross"):
        print(f"-- {kind}: trajectories {leaf[kind]['TOTAL']}")
        for k, v in sorted(leaf[kind].items()):
            if k != "TOTAL":
                print(f"     leaf {k:24s} {v:4d} ({v / max(1, leaf[kind]['TOTAL']):.0%})")
        for k in ("no_code", "print_only", "hollow", "errored", "ran", "compared", "sample", "mismatch",
                  "crate_blocked", "code2_called", "code2_agree", "code2_mismatch", "harness_compared",
                  "harness_different", "harness_one_failed", "pred_tested:clone"):
            print(f"     tree {k:24s} {tree[kind][k]:4d} / {n[kind]} ({tree[kind][k] / max(1, n[kind]):.0%})")
    print(f"hard negatives where a run showed a mismatch but the leaf still said clone: {sorted(set(examples))}")


if __name__ == "__main__":
    main()

"""Process-level checks on one mined root-to-leaf trace.

Shared by score_rl_rollouts.py (strict SAFE filter), build_prm_labels.py
(step labels), build_dpo_pairs.py (chosen/rejected selection) and
process_reward.py (scalar reward) so all four agree on what a bad trace is.
See SFT_DATA_PROCESS.txt section 9 for how each flag was found.

A leaf that reaches the right label can still be a bad imitation target.
Section 4's SAFE/DANGEROUS split only looked at whether code execution
FAILED. These checks add the ways a trace can be wrong while every code
step "ran":

  empty_code_output  every code step's observation is empty, or holds only
                     print labels with no values ("result.stdout: " from a
                     subprocess call that was never given the program) --
                     the trace promised verification and never got any.
                     exec_outcome_of() maps an empty observation to None
                     ("never tried code"), which is how these passed
                     section 4's filter as pure reasoning.
  claims_unrun_tests the <answer> reason says tests passed / outputs were
                     verified, but no code step produced any output.
  divergent_evidence the trace tested an input on which diff_test_pairs.py
                     shows Code 1 and Code 2 print DIFFERENT outputs (both
                     exiting cleanly) and still concluded clone "because the
                     tests pass" -- its asserted expectation was never true
                     of Code 2. Needs a diff-test record for the question.
  copied_description the Code 2 summary step is a near-verbatim copy of the
                     Code 1 summary (language name swapped) -- the model did
                     not actually read Code 2.
  vacuous_clone      the only code is print("not code clones") (the prompt's
                     prescribed non-clone shortcut) yet the answer is clone.

Severity: HARD flags make a leaf unusable; SOFT flags keep it but rank it
below leaves without them (see leaf_tier()).
"""
import ast
import difflib
import re

from evaluate_clone_results import exec_outcome_of

HARD_FLAGS = {"claims_unrun_tests", "divergent_evidence", "vacuous_clone"}
SOFT_FLAGS = {"empty_code_output", "copied_description"}

# An <answer> reason that reports test evidence: "passes", "verified",
# "confirming", "produces the same/correct/expected output(s)", "tested ...".
CLAIMED_TEST_RE = re.compile(
    r"\b(pass(es|ed|ing)?\b|verif\w*|confirm\w*|tested\b|when (run|tested)\b|"
    r"produc(es|ed|ing) (the )?(same|correct|expected|identical|equivalent)\b)", re.I)
INPUT_LITERAL_RE = re.compile(r"""input\s*=\s*((?:[rbuf]?)(?:"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'))""")
LANG_WORDS_RE = re.compile(r"\b(python|rust|java|script|program|snippet|code ?[12])\b", re.I)


def chain(tree, tag):
    parts = tag.split(".")
    out = []
    for i in range(1, len(parts) + 1):
        node = tree.get(".".join(parts[:i]))
        if isinstance(node, dict):
            out.append((".".join(parts[:i]), node))
    return out


def code_steps(nodes):
    return [(t, n) for t, n in nodes if n.get("action") == "python_interpreter"]


def is_print_only(code):
    try:
        body = ast.parse(code).body
    except SyntaxError:
        return False
    return bool(body) and all(
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
        and getattr(s.value.func, "id", None) == "print"
        and all(isinstance(a, ast.Constant) for a in s.value.args)
        for s in body)


LABEL_ONLY_RE = re.compile(r"^\w+(?:\.\w+|\[[^\]]*\])+\(?:\s*$")  # e.g. "result.stdout:", "r.stdout.strip(:"


def has_values(observation):
    """False for an empty observation or one made only of the interpreter's
    per-print labels with nothing after them (e.g. 'result.stdout: ')."""
    lines = [l for l in observation.splitlines() if l.strip()]
    return any(not LABEL_ONLY_RE.match(l.strip()) for l in lines)


def strip_code_tags(code):
    return re.sub(r"<[^>]+>", "", code or "")


def answer_reason(leaf_text):
    m = re.search(r"Reason:(.*)", leaf_text or "", re.S)
    return m.group(1) if m else ""


def analysis_steps(nodes):
    return [n.get("text") or "" for _, n in nodes
            if n.get("action") != "python_interpreter" and not n.get("final_answer")
            and (n.get("text") or "").strip()]


def description_similarity(nodes):
    steps = analysis_steps(nodes)
    if len(steps) < 2:
        return 0.0
    a, b = (LANG_WORDS_RE.sub("", re.sub(r"<[^>]+>", "", s)).lower().split() for s in steps[:2])
    return difflib.SequenceMatcher(None, a, b).ratio()


def trace_inputs(nodes):
    found = []
    for _, n in code_steps(nodes):
        for lit in INPUT_LITERAL_RE.findall(n.get("action_input") or ""):
            try:
                v = ast.literal_eval(lit)
            except Exception:
                continue
            if isinstance(v, str):
                found.append(v)
    return found


def leaf_flags(tree, tag, label, diff_record=None):
    """Return the set of flags for the root-to-`tag` trace whose final label
    is `label` ('clone' / 'non-clone')."""
    nodes = chain(tree, tag)
    flags = set()
    steps = code_steps(nodes)
    observations = [(n.get("observation") or "").strip() for _, n in steps]
    any_output = any(has_values(o) for o in observations)

    if steps and not any_output:
        flags.add("empty_code_output")
    reason = answer_reason(nodes[-1][1].get("text") if nodes else "")
    if label == "clone" and not any_output and CLAIMED_TEST_RE.search(reason):
        flags.add("claims_unrun_tests")
    if label == "clone" and steps and all(is_print_only(strip_code_tags(n.get("action_input"))) for _, n in steps):
        flags.add("vacuous_clone")
    if description_similarity(nodes) >= 0.9:
        flags.add("copied_description")
    if label == "clone" and diff_record and diff_record.get("cases"):
        divergent = {c["input"] for c in diff_record["cases"] if c["valid"] and not c["match"]}
        if divergent and any(i in divergent for i in trace_inputs(nodes)):
            flags.add("divergent_evidence")
    return flags


def exec_outcomes(tree, tag):
    return {exec_outcome_of(n) for _, n in chain(tree, tag)} - {None}


def leaf_tier(flags, outcomes):
    """'A' = verified by real execution output and no soft flags,
    'B' = usable but weaker (soft flags / pure reasoning), 'X' = drop."""
    if flags & HARD_FLAGS:
        return "X"
    if outcomes and "ran" not in outcomes:
        return "X"  # section 4 DANGEROUS
    if flags & SOFT_FLAGS or not outcomes:
        return "B"
    return "A"

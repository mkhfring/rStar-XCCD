"""Search-time step scoring, version 2 (config.score_version == "v2").

Pure functions so the rules can be replayed offline over stored trees
(replay_step_scoring.py) as well as used live by MCTS.create_child() /
eval_final_answer(). Version "v1" (the default) is the unchanged
assert-consistency-score behaviour in mcts.py; nothing here runs unless a
config opts in. Rationale per rule: MCTS_SEARCH_SCORING.md, section
"score_version v2", and SFT_DATA_PROCESS.txt section 9 (the leaf audit these
rules come from). None of them reads the ground-truth label.

Code step (score_code_step):
  outcome            reward      deferred verdict   counts toward errors_threshold
  no_code            negative    -                  yes (unchanged)
  blocked / Code 2   negative    -                  no
    not runnable
  crash (exception)  negative    -                  yes
  AssertionError     0 (neutral) "non-clone" only if Code 2 was actually run in
                                 this program; otherwise none (the model's own
                                 guessed literal failed -- no evidence)
                                                    no
  ran, no output     0 (neutral) -                  no   (v1 paid +1: 44/110
    / label-only                                          rust hard leaves were
                                                          `import subprocess`
                                                          and nothing else)
  verdict echo       0 (neutral) -                  no   (print("not code clones"))
  ran, real output   positive    "clone" if the program asserted AND ran
                                 Code 2 (a real two-program comparison);
                                 otherwise no verdict (v1 tagged every passing
                                 self-written assert as clone evidence)
  ...and ran Code 2  +code2_bonus on top (running both programs on the same
                     input is the one behaviour that actually settles a pair)

Leaf (score_leaf): a clone answer whose reason claims tests passed / outputs
were verified while no ancestor code step produced any output, or whose only
"test" printed a verdict, gets negative_reward backed up the path.
"""
import re

from evaluate_clone_results import NO_CODE_MESSAGE, JAVA_BLOCKED_PREFIX, normalize_label
from rstar_deepthink.tools.python_tool import mentions_code2_execution

ASSERT_RE = re.compile(r"\bassert\b")
EXCEPTION_RE = re.compile(r"^(?:Traceback|[A-Za-z_]*(?:Error|Exception|Interrupt)\b)", re.M)
LABEL_ONLY_RE = re.compile(r"^\w+(?:\.\w+|\[[^\]]*\])+\(?:\s*$")
CODE2_UNAVAILABLE = ("uses external crate", "but `javac", "but `rustc", "is unavailable")
CODE2_READY = ("compiled to", "compiled:")
CLAIMED_TEST_RE = re.compile(
    r"\b(pass(es|ed|ing)?\b|verif\w*|confirm\w*|tested\b|when (run|tested)\b|"
    r"produc(es|ed|ing) (the )?(same|correct|expected|identical|equivalent)\b)", re.I)


def has_values(observation):
    lines = [l for l in observation.splitlines() if l.strip()]
    return any(not LABEL_ONLY_RE.match(l.strip()) for l in lines)


def is_verdict_echo(observation):
    obs = observation.strip()
    return 0 < len(obs) <= 40 and normalize_label(obs) is not None


def classify(observation, executed_code):
    """Return (outcome, code2_ran). outcome in: no_code, blocked, crash,
    assertion_failed, empty, echo, ok."""
    obs = (observation or "").strip()
    code2_attempt = mentions_code2_execution(executed_code or "")
    code2_ready = code2_attempt and any(s in obs for s in CODE2_READY)
    if obs == NO_CODE_MESSAGE:
        return "no_code", False
    if obs.startswith(JAVA_BLOCKED_PREFIX) or (code2_attempt and any(s in obs for s in CODE2_UNAVAILABLE)):
        return "blocked", False
    # The Code-2 staging status is appended after the program's own output;
    # classify on the program output only.
    body = obs
    for marker in ("Code 2 (Rust) compiled", "Code 2 (Java) compiled"):
        if marker in body:
            body = body[:body.index(marker)].strip()
    m = EXCEPTION_RE.search(body)
    if m:
        return ("assertion_failed" if "AssertionError" in body[m.start():] else "crash"), code2_ready
    if not has_values(body):
        return "empty", code2_ready
    if is_verdict_echo(body):
        return "echo", code2_ready
    return "ok", code2_ready


def score_code_step(observation, executed_code, positive, negative, code2_bonus=0.5):
    """Return (reward or None, deferred verdict or None, counts_as_error)."""
    outcome, code2_ran = classify(observation, executed_code)
    has_assert = bool(ASSERT_RE.search(executed_code or ""))
    if outcome == "no_code":
        return negative, None, True
    if outcome == "blocked":
        return negative, None, False
    if outcome == "crash":
        return negative, None, True
    if outcome == "assertion_failed":
        return (None, "non-clone", False) if code2_ran else (0.0, None, False)
    if outcome in ("empty", "echo"):
        return 0.0, None, False
    if code2_ran and has_assert:
        return None, "clone", False
    return positive + (code2_bonus if code2_ran else 0.0), None, False


def score_leaf(final_answer, leaf_text, ancestor_states, negative):
    """GT-free penalty for a terminal clone answer, or None. ancestor_states
    are the state dicts from the leaf's parent up to the root."""
    label = normalize_label(final_answer or "")
    if label != "clone":
        return None
    code_states = [s for s in ancestor_states if s.get("action") == "python_interpreter"]
    if not code_states:
        return None
    outputs = [(s.get("observation") or "").strip() for s in code_states]
    if all(is_verdict_echo(o) for o in outputs if o):
        if any(outputs):
            return negative
    if not any(has_values(o) for o in outputs):
        m = re.search(r"Reason:(.*)", leaf_text or "", re.S)
        if m and CLAIMED_TEST_RE.search(m.group(1)):
            return negative
    return None

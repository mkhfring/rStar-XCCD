"""Replay search-time step scoring (v1 vs v2) over stored MCTS trees and
measure how well each version's rewards predict whether a step leads to a
correct answer (MCTS_SEARCH_SCORING.md, "score_version v2").

For every python_interpreter node that has at least one answered leaf below
it, the node's score under each version is computed exactly as the search
would (immediate reward; deferred assert verdicts resolved against the
descendant leaves' labels, averaged, as _score_pending_assert_verdicts
does). The ground-truth label is used ONLY here, to evaluate the scores:
y = fraction of descendant leaves whose label is correct.

Reported per version: AUC of the score for separating steps whose subtree
is mostly right (y > 0.5) from mostly wrong, the mean y per reward value,
and the same restricted to ground-truth-clone questions (the population
these runs lose).

Usage:
    python replay_step_scoring.py TREES.jsonl [...] [--only_indices_from Q.jsonl]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_clone_results import NO_CODE_MESSAGE, node_sort_key, normalize_label
from rstar_deepthink.agents.step_scoring import ASSERT_RE, score_code_step
from rstar_deepthink.agents.tree import extract_program


def executed_program(numeric, tag):
    """Accumulated program as collect_action_inputs() + extract_program() see it."""
    inputs = []
    t = tag
    while t:
        n = numeric.get(t)
        if n is None:
            break
        if n.get("action") == "python_interpreter" and n.get("action_input"):
            inputs.append(n["action_input"])
        if t != tag and ("<end_of_output>" in (n.get("text") or "") or "<end_of_code>" in (n.get("text") or "")):
            break
        t = t.rsplit(".", 1)[0] if "." in t else None
    return extract_program("".join(reversed(inputs)))


def v1_score(observation, program):
    ok = "error" not in observation.lower() and observation != NO_CODE_MESSAGE
    if observation.startswith("AssertionError"):
        return None, "non-clone"
    if not ok:
        return -1.0, None
    if ASSERT_RE.search(program):
        return None, "clone"
    return 1.0, None


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trees", nargs="+")
    ap.add_argument("--only_indices_from", action="append", default=[])
    args = ap.parse_args()
    allowed = None
    for p in args.only_indices_from:
        allowed = (allowed or set()) | {json.loads(l)["index"] for l in open(p)}

    rows = []  # (version, score, y, truth)
    for path in args.trees:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if "rstar" not in r or (allowed is not None and r["index"] not in allowed):
                continue
            truth = r["answer"]
            numeric = {t: n for t, n in r["rstar"].items() if node_sort_key(t) is not None}
            leaves = {t: normalize_label((n.get("final_answer") or "").strip())
                      for t, n in numeric.items() if (n.get("final_answer") or "").strip()}
            leaves = {t: l for t, l in leaves.items() if l is not None}
            for tag, n in numeric.items():
                if n.get("action") != "python_interpreter":
                    continue
                below = [l for t, l in leaves.items() if t.startswith(tag + ".")]
                if not below:
                    continue
                y = sum(l == truth for l in below) / len(below)
                obs = (n.get("observation") or "").strip()
                prog = executed_program(numeric, tag)
                v2 = score_code_step(obs, prog, 1.0, -1.0)[:2]
                v2n = (0.0 if v2[0] is not None and v2[0] < 0 else v2[0], v2[1])
                for version, (reward, verdict) in (
                        ("v1", v1_score(obs, prog)),
                        ("v2", v2),
                        ("v2-nopenalty", v2n)):
                    if verdict is not None:
                        reward = sum(1.0 if l == verdict else -1.0 for l in below) / len(below)
                    rows.append((version, reward, y, truth))

    for subset, keep in (("all questions", lambda t: True), ("ground-truth clone questions", lambda t: t == "clone")):
        print(f"== {subset}")
        for version in ("v1", "v2", "v2-nopenalty"):
            rs = [(s, y) for v, s, y, t in rows if v == version and keep(t) and s is not None]
            unscored = sum(1 for v, s, y, t in rows if v == version and keep(t) and s is None)
            a = auc([s for s, y in rs if y > 0.5], [s for s, y in rs if y <= 0.5])
            by = defaultdict(list)
            for s, y in rs:
                by[round(s, 2)].append(y)
            dist = "  ".join(f"{k:+.2f}:n={len(v)},acc={sum(v)/len(v):.2f}" for k, v in sorted(by.items()))
            print(f"   {version}: AUC {a:.3f}  scored {len(rs)} unscored {unscored} | {dist}")


if __name__ == "__main__":
    main()

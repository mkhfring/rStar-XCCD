"""Format check of code-step search trees (Phase 0.5 smoke test (e); also usable per RFT round).

Reports, per run file: trees answered, leaves with a valid label, code steps that are not
valid/runnable Python, steps calling run_both, run_both calls per leaf path, steps per path.
Accuracy is printed for information only (20 pairs: not a result).

Usage: python train/format_check.py <run.jsonl> [<run.jsonl> ...]
"""
import json
import re
import sys

BAD_CODE = ("No valid Python code found", "SyntaxError", "IndentationError")
CALLS_RE = re.compile(r"\[run_both\] (\d+) call")
LABEL_RE = re.compile(r"\\boxed\{\s*(clone|non-clone)\s*\}")


def key(tag):
    return tuple(int(x) for x in tag.split("."))


def main():
    for path in sys.argv[1:]:
        trees = [json.loads(l) for l in open(path)]
        trees = [t for t in trees if "rstar" in t]
        n_answered = leaves = valid = steps = bad = with_call = calls = path_steps = correct_votes = 0
        for r in trees:
            t = r["rstar"]
            lv = [k for k, s in t.items() if isinstance(s, dict) and s.get("final_answer")]
            n_answered += bool(lv)
            for k, s in t.items():
                if isinstance(s, dict) and s.get("action") == "python_interpreter":
                    steps += 1
                    obs = s.get("observation") or ""
                    bad += any(b in obs for b in BAD_CODE)
                    with_call += bool(CALLS_RE.search(obs))
            votes = []
            for k in lv:
                leaves += 1
                m = LABEL_RE.search(t[k].get("text") or "")
                valid += bool(m)
                if m:
                    votes.append(m.group(1))
                kk = key(k)
                chain = [t[".".join(map(str, kk[:i]))] for i in range(1, len(kk) + 1)]
                code = [c for c in chain if c.get("action") == "python_interpreter"]
                path_steps += len(code)
                calls += sum(int(x) for c in code for x in CALLS_RE.findall(c.get("observation") or ""))
            if votes:
                maj = "clone" if votes.count("clone") >= votes.count("non-clone") else "non-clone"
                correct_votes += maj == r.get("answer")
        L = max(leaves, 1)
        print(f"== {path}")
        print(f"   trees {len(trees)} | answered {n_answered} ({n_answered / max(len(trees), 1):.0%}) | "
              f"leaves {leaves}, valid label {valid} ({valid / L:.0%})")
        print(f"   code steps {steps}: invalid code {bad} ({bad / max(steps, 1):.0%}), calling run_both "
              f"{with_call} ({with_call / max(steps, 1):.0%})")
        print(f"   per leaf path: {path_steps / L:.1f} code steps, {calls / L:.1f} run_both calls")
        print(f"   (info only) majority-vote correct {correct_votes}/{len(trees)}")


if __name__ == "__main__":
    main()

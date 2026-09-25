"""Check SFT/DPO/PRM training files against evaluation files for leakage
(SFT_DATA_PROCESS.txt section 9).

For every training record, Code 1 (Python) and Code 2 are extracted from the
question and compared with every snippet in the evaluation files:
  exact       -- identical after whitespace normalisation;
  near_dup    -- token-set Jaccard >= --jaccard (default 0.8), catching the
                 same submission with cosmetic edits.
Snippets shorter than --min_tokens tokens are skipped for near_dup (tiny
programs such as `print(input())` collide by accident).

Rules this project follows (section 2): Track B (CodeNet) data may be
evaluated on the full CLCCD test files; Track A (rl_train) data may be
evaluated ONLY on rl_heldout_*. Run this with the matching eval files and
expect zero exact matches; near_dup hits are listed for inspection.

Usage:
    python check_leakage.py --train T1.jsonl [T2.jsonl ...] \
        --eval eval_data/test_python_rust_CLCCD.jsonl [...] [--jaccard 0.8]
"""
import argparse
import json
import re
from collections import defaultdict

SNIPPET_RE = re.compile(r"Code 1: Python\n```python\n(.*?)```\s*\n+Code 2: (\w+)\n```\w*\n(.*?)```", re.S)
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+|\S")


def snippets(question):
    m = SNIPPET_RE.search(question)
    return (m.group(1), m.group(3)) if m else (None, None)


def norm(code):
    return " ".join(code.split())


def tokens(code):
    return frozenset(TOKEN_RE.findall(code))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--min_tokens", type=int, default=25)
    args = ap.parse_args()

    exact = {}
    by_token = defaultdict(list)
    eval_snips = []
    for path in args.eval:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            for side, code in zip(("code1", "code2"), snippets(r["question"])):
                if code is None:
                    continue
                exact.setdefault(norm(code), []).append((path, r.get("index"), side))
                toks = tokens(code)
                if len(toks) >= args.min_tokens:
                    k = len(eval_snips)
                    eval_snips.append((toks, path, r.get("index"), side))
                    for t in toks:
                        by_token[t].append(k)

    seen_q = set()
    n_q = n_exact = n_near = 0
    for path in args.train:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            q = r.get("question") or r.get("prompt") or ""
            if q in seen_q:
                continue
            seen_q.add(q)
            n_q += 1
            for side, code in zip(("code1", "code2"), snippets(q)):
                if code is None:
                    continue
                hit = exact.get(norm(code))
                if hit:
                    n_exact += 1
                    print(f"EXACT    {path}#{r.get('index')} {side} == {hit[0]}")
                    continue
                toks = tokens(code)
                if len(toks) < args.min_tokens:
                    continue
                counts = defaultdict(int)
                for t in toks:
                    for k in by_token.get(t, ()):
                        counts[k] += 1
                best = max(((c / (len(toks) + len(eval_snips[k][0]) - c), k) for k, c in counts.items()),
                           default=(0.0, None))
                if best[0] >= args.jaccard:
                    n_near += 1
                    _, ep, ei, es = eval_snips[best[1]]
                    print(f"NEAR_DUP {path}#{r.get('index')} {side} ~ {ep}#{ei} {es} (jaccard {best[0]:.2f})")
    print(f"checked {n_q} distinct training questions against {len(args.eval)} eval file(s): "
          f"{n_exact} exact snippet matches, {n_near} near-duplicates (jaccard >= {args.jaccard})")


if __name__ == "__main__":
    main()

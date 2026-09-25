"""Generate random stdin inputs for a Rust program from its proconio
`input! { ... }` declaration, for diff_test_pairs.py (--manual_inputs).

The declaration fixes the token sequence exactly; what it does not fix is
the LINE layout, which matters to the Python side (input().split() reads one
line at a time). Two layouts are emitted per random draw:
  row    -- consecutive scalars share a line, a 1-D array of scalars is one
            line, each tuple / each row of a 2-D array is its own line
            (the usual AtCoder layout);
  column -- every array element on its own line;
  tokens -- one token per line;  flat -- every token on one line.
diff_test_pairs.py only counts an input when BOTH programs exit cleanly on
it, so the wrong layout for a given Python parser is simply discarded.

Values are small (ints 1..9 unless a length, lengths 1..6, strings of
'a'..'c' length 1..6) -- small enough to stay inside typical constraints
and to exercise ties/duplicates, which is where clone pairs usually diverge
if they diverge at all. Inputs can still violate a problem-specific
constraint, so a disagreement is a lead to inspect, not a verdict.

Usage:
    python gen_inputs_proconio.py QUESTIONS.jsonl OUT.json [--per_question 12]
"""
import argparse
import json
import random
import re
import sys

INPUT_BLOCK_RE = re.compile(r"input!\s*\{(.*?)\}", re.S)
INT_TYPES = {"usize", "u64", "u32", "i64", "i32", "isize", "u128", "i128", "u16", "i16", "u8", "i8"}


def split_top(s, sep=","):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out if x.strip()]


def parse_decls(block):
    decls = []
    for item in split_top(block):
        if ":" not in item:
            continue
        name, ty = item.split(":", 1)
        decls.append((name.replace("mut", "").strip(), ty.strip()))
    return decls


class Gen:
    def __init__(self, rng):
        self.rng = rng
        self.env = {}

    def length(self, expr):
        expr = expr.strip()
        if expr.isdigit():
            return int(expr)
        try:
            return int(eval(expr, {}, dict(self.env)))
        except Exception:
            return self.rng.randint(1, 6)

    def value(self, ty, name=None):
        """Returns a nested structure: scalar str, ('tuple', [..]), ('arr', [..])."""
        ty = ty.strip()
        if ty.startswith("[") and ty.endswith("]"):
            inner, n = ty[1:-1].rsplit(";", 1)
            k = self.length(n)
            return ("arr", [self.value(inner) for _ in range(k)])
        if ty.startswith("(") and ty.endswith(")"):
            return ("tuple", [self.value(t) for t in split_top(ty[1:-1])])
        base = ty.split("::")[-1]
        if base in INT_TYPES:
            is_len = name is not None and re.fullmatch(r"[nmkqhwlt]|n\d|len|cnt|num", name.lower() or "")
            v = self.rng.randint(1, 6) if is_len else self.rng.randint(1, 9)
            if base.startswith("i") and not is_len and self.rng.random() < 0.15:
                v = -v
            return str(v)
        if base in ("Usize1", "Isize1"):
            n = self.env.get("n") or self.env.get("N") or 3
            return str(self.rng.randint(1, max(1, int(n))))
        if base in ("f64", "f32"):
            return str(self.rng.randint(1, 9))
        if base in ("String", "Chars", "Bytes"):
            return "".join(self.rng.choice("abc") for _ in range(self.rng.randint(1, 6)))
        if base in ("char", "u8char"):
            return self.rng.choice("abc")
        return str(self.rng.randint(1, 9))


def render(values, layout):
    if layout in ("tokens", "flat"):
        toks = [t for v in values for t in ([v] if isinstance(v, str) else _flat(v))]
        return ("\n" if layout == "tokens" else " ").join(toks) + "\n"
    lines, cur = [], []

    def flush():
        if cur:
            lines.append(" ".join(cur))
            cur.clear()

    def flat(v):
        if isinstance(v, str):
            return [v]
        return [t for x in v[1] for t in flat(x)]

    for v in values:
        if isinstance(v, str):
            cur.append(v)
        elif v[0] == "tuple":
            cur.extend(flat(v))
        else:
            flush()
            elems = v[1]
            if layout == "row" and all(isinstance(e, str) for e in elems):
                lines.append(" ".join(elems))
            else:
                for e in elems:
                    lines.append(" ".join(flat(e)))
    flush()
    return "\n".join(lines) + "\n"


def _flat(v):
    if isinstance(v, str):
        return [v]
    return [t for x in v[1] for t in _flat(x)]


def inputs_for(rust_src, per_question, seed):
    m = INPUT_BLOCK_RE.search(rust_src)
    if not m:
        return []
    decls = parse_decls(m.group(1))
    if not decls:
        return []
    rng = random.Random(seed)
    out = []
    for _ in range(per_question):
        g = Gen(rng)
        vals = []
        for name, ty in decls:
            v = g.value(ty, name)
            if isinstance(v, str) and re.fullmatch(r"-?\d+", v):
                g.env[name] = int(v)
            vals.append(v)
        for layout in ("row", "column", "tokens", "flat"):
            s = render(vals, layout)
            if s not in out:
                out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--per_question", type=int, default=12)
    args = ap.parse_args()
    res = {}
    for line in open(args.questions, encoding="utf-8"):
        r = json.loads(line)
        q = r["question"]
        rust = q[q.find("Code 2"):]
        ins = inputs_for(rust, args.per_question, seed=r["index"])
        if ins:
            res[str(r["index"])] = ins
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"generated inputs for {len(res)} questions -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

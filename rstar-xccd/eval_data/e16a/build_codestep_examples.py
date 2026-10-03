"""Build the DRAFT code-step prompt examples (2026-10-02, for user review) with REAL outputs.

Semantics the examples assume (to be implemented in the harness for the pilot):
  - run_both(stdin_text) is predefined; returns (r1, r2), each with .ok (exit 0), .out (stdout)
  - each code step is executed once, state persists across the steps of one trajectory, and the
    model sees only that step's own output
Each step below is executed in order in one namespace with Code 1 / Code 2 really built and run.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
from collections import namedtuple

OUT_DIR = sys.argv[1]
Res = namedtuple("Res", "ok out err")

PY_SUM = "n = int(input())\ntotal = 0\nfor i in range(1, n + 1):\n    if i % 3 == 0 or i % 5 == 0:\n        total += i\nprint(total)\n"
PY_DIV = "a = int(input())\nb = int(input())\nprint(a // b)\n"
PY_PARITY = 'n = int(input())\nif n % 2 == 0:\n    print("Even")\nelse:\n    print("Odd")\n'
JAVA = {
    "sum": "import java.util.*;\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int n = sc.nextInt();\n        int total = 0;\n        for (int i = 1; i <= n; i++) {\n            if (i % 3 == 0 || i % 5 == 0) {\n                total += i;\n            }\n        }\n        System.out.println(total);\n    }\n}\n",
    "div": "import java.util.*;\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int a = sc.nextInt();\n        int b = sc.nextInt();\n        System.out.println(a / b);\n    }\n}\n",
    "rev": "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        String s = sc.next();\n        StringBuilder sb = new StringBuilder(s);\n        sb.reverse();\n        System.out.println(sb.toString());\n    }\n}\n",
}
RUST = {
    "sum": "use std::io;\n\nfn main() {\n    let mut buf = String::new();\n    io::stdin().read_line(&mut buf).unwrap();\n    let n: i64 = buf.trim().parse().unwrap();\n    let mut total: i64 = 0;\n    for i in 1..=n {\n        if i % 3 == 0 || i % 5 == 0 {\n            total += i;\n        }\n    }\n    println!(\"{}\", total);\n}\n",
    "div": "use std::io;\n\nfn read() -> i64 {\n    let mut buf = String::new();\n    io::stdin().read_line(&mut buf).unwrap();\n    buf.trim().parse().unwrap()\n}\n\nfn main() {\n    let a = read();\n    let b = read();\n    println!(\"{}\", a / b);\n}\n",
    "rev": "use std::io;\n\nfn main() {\n    let mut s = String::new();\n    io::stdin().read_line(&mut s).unwrap();\n    let r: String = s.trim().chars().rev().collect();\n    println!(\"{}\", r);\n}\n",
}
LANG_READ = {"java": {"sum": "sc.nextInt()", "div": "two sc.nextInt() calls", "rev": "sc.next(), a string"},
             "rust": {"sum": "one line parsed as i64", "div": "two lines parsed as i64", "rev": "one line kept as a string"}}

EXAMPLES = [
    ("sum", PY_SUM, "clone", [
        "# Step 1: Both programs read one integer n from standard input (Code 1: int(input()); Code 2: {read}).\n"
        "# Check that a typical input is accepted by both programs.\n"
        "r1, r2 = run_both(\"10\")\nis_clone = r1.ok and r2.ok\nprint(is_clone)",
        "# Step 2: Both should print one integer, the sum. Check that both observed outputs are integers.\n"
        "is_int = lambda s: s.strip().lstrip(\"-\").isdigit()\nis_clone = is_clone and is_int(r1.out) and is_int(r2.out)\nprint(is_clone)",
        "# Step 3: The programs could differ at the loop bounds (is n itself included?) and for small n.\n"
        "# Test n = 1, n = 15 (a multiple of both 3 and 5) and n = 100. Compare only inputs both programs accept.\n"
        "for s in [\"1\", \"15\", \"100\"]:\n    a, b = run_both(s)\n    print(repr(s), a.out, b.out)\n"
        "    if a.ok and b.ok:\n        is_clone = is_clone and a.out.split() == b.out.split()\nprint(is_clone)",
        "# Step 4: Conclusion.\nprint(\"\\\\boxed{clone}\" if is_clone else \"\\\\boxed{non-clone}\")",
    ], "Both programs accepted the same inputs, printed integers, and printed identical sums for n = 1, 10, 15 and 100, including the bound n itself."),
    ("div", PY_DIV, "non-clone", [
        "# Step 1: Both programs read two integers from standard input (Code 1: two input() calls; Code 2: {read}).\n"
        "# Check that a typical input is accepted by both programs.\n"
        "r1, r2 = run_both(\"7\\n2\")\nis_clone = r1.ok and r2.ok\nprint(is_clone)",
        "# Step 2: Both should print one integer, the quotient. Check that both observed outputs are integers.\n"
        "is_int = lambda s: s.strip().lstrip(\"-\").isdigit()\nis_clone = is_clone and is_int(r1.out) and is_int(r2.out)\nprint(is_clone)",
        "# Step 3: Code 1 uses // (rounds down) and Code 2 uses / on integers (rounds toward zero).\n"
        "# They can differ only when the quotient is negative, so test negative operands and an exact division.\n"
        "for s in [\"-7\\n2\", \"7\\n-2\", \"6\\n3\"]:\n    a, b = run_both(s)\n    print(repr(s), a.out, b.out)\n"
        "    if a.ok and b.ok:\n        is_clone = is_clone and a.out.split() == b.out.split()\nprint(is_clone)",
        "# Step 4: Conclusion.\nprint(\"\\\\boxed{clone}\" if is_clone else \"\\\\boxed{non-clone}\")",
    ], "On -7 and 2, Code 1 printed -4 and Code 2 printed -3: the programs round negative quotients differently, so they do not compute the same function."),
    ("rev", PY_PARITY, "non-clone", [
        "# Step 1: Code 1 reads one integer; Code 2 reads one token ({read}). Both accept a digit string,\n"
        "# so check that a typical input is accepted by both programs.\n"
        "r1, r2 = run_both(\"5\")\nis_clone = r1.ok and r2.ok\nprint(is_clone)",
        "# Step 2: Code 1 should print a word (Even or Odd); Code 2 prints the reversed input.\n"
        "# Check that both observed outputs are of the same kind (both words or both numbers).\n"
        "is_int = lambda s: s.strip().lstrip(\"-\").isdigit()\nis_clone = is_clone and is_int(r1.out) == is_int(r2.out)\nprint(repr(r1.out), repr(r2.out), is_clone)",
        "# Step 3: is_clone is already False: the programs do not even produce the same kind of output,\n"
        "# so they solve different tasks and no behavioural test is needed. Conclusion.\n"
        "print(\"\\\\boxed{clone}\" if is_clone else \"\\\\boxed{non-clone}\")",
    ], "Code 1 prints a parity word while Code 2 prints its input reversed; on the same input 5 they printed 'Odd' and '5', different kinds of output for unrelated tasks."),
]


def build(lang, key, py_src, d):
    open(f"{d}/candidate_code.py", "w").write(py_src)
    if lang == "java":
        open(f"{d}/Main.java", "w").write(JAVA[key])
        subprocess.run(["javac", "Main.java"], cwd=d, check=True, capture_output=True)
        cmd2 = ["java", "-cp", d, "Main"]
    else:
        open(f"{d}/candidate_code2.rs", "w").write(RUST[key])
        subprocess.run(["rustc", "-O", "candidate_code2.rs", "-o", "candidate_code2"], cwd=d, check=True, capture_output=True)
        cmd2 = [f"{d}/candidate_code2"]

    def run_both(stdin_text):
        def run(cmd):
            p = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, timeout=10, cwd=d)
            return Res(p.returncode == 0, p.stdout.strip(), p.stderr.strip()[:200])
        return run(["python3", f"{d}/candidate_code.py"]), run(cmd2)
    return run_both


def main():
    out = {}
    for lang, label2 in (("java", "Java"), ("rust", "Rust")):
        shots = []
        for key, py_src, answer, steps, reason in EXAMPLES:
            d = os.path.join(OUT_DIR, f"{lang}_{key}")
            os.makedirs(d, exist_ok=True)
            ns = {"__name__": "__main__", "run_both": build(lang, key, py_src, d)}
            code2 = (JAVA if lang == "java" else RUST)[key]
            q = (f"Question: Determine whether the following two code snippets are semantic code clones.\n\n"
                 f"Code 1: Python\n```python\n{py_src}```\n\nCode 2: {label2}\n```{lang}\n{code2}```\n")
            body = "<code>\n"
            for k, step in enumerate(steps):
                step = step.replace("{read}", LANG_READ[lang][key])
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(step, ns)
                last = k == len(steps) - 1
                body += step + ("\n<end_of_code>\n" if last else "\n<end_of_step>\n") + f"<output>\n{buf.getvalue().rstrip()}\n<end_of_output>\n"
            body += f"<answer>\nFinal decision: \\boxed{{{answer}}}\nReason: {reason}\n<end_of_answer>"
            assert ("\\boxed{" + answer + "}") in buf.getvalue(), (lang, key, buf.getvalue())
            shots.append(q + body)
        out[lang] = shots
    json.dump(out, open(os.path.join(OUT_DIR, "examples.json"), "w"), indent=1, ensure_ascii=False)
    print(out["java"][1])


if __name__ == "__main__":
    main()

"""Build the CODE-STEP prompt examples (branch codestep-prompt, prompt version v3).

Seven worked examples per language, every output produced by the REAL harness function
python_tool.run_codestep (so the examples show exactly what the model will see):
  1 clone                 sum of multiples of 3 or 5
  2 non-clone, same task  a // b vs a / b (rounding of negative quotients)
  3 non-clone, diff. task parity word vs reversed input (early stop at the output comparison)
  4 clone, one-sided crash first input fails only in Code 1 (wrong input format) -> retry
  5 clone, no input       both call a function on fixed arguments (CLCCD style) -> compare outputs
  6 clone, broken program Code 1 fails on every input (Python 2 division) -> decide by reading,
                          and write the decision into is_clone
  7 clone, both fail      sentinel-terminated input: first input lacks the "0 0" line, BOTH programs fail;
                          the step prints err() for both, the input is fixed (not a difference, not read)
v2 (2026-10-03, after 28+26 v1 pilot trees): outputs compared with the predefined same(a, b)
instead of a model-written "same kind" check; examples 5 and 6 added. v1 = commit 19011e8.
v3 (2026-10-03, after ~48 trees per language of v2 on the DEV hard-negative pilot): every run step prints
r1.ok, r2.ok and the predefined err(r1), err(r2); example 7 added. v2 = commit 0e44f55.
Writes rstar_deepthink/few_shots/few_shot_codestep_python_{java,rust}_v3.json.
Run from rstar-xccd/ with java/17 and ~/.cargo/bin on PATH:
  python eval_data/e16a/build_codestep_examples.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rstar_deepthink.tools.python_tool import run_codestep  # noqa: E402

TEMPLATE = ("Determine whether the following two code snippets are semantic code clones.\n\n"
            "Code 1: Python\n```python\n{c1}```\n\nCode 2: {Lname}\n```{ltag}\n{c2}```")
PY = {
    "sum": "n = int(input())\ntotal = 0\nfor i in range(1, n + 1):\n    if i % 3 == 0 or i % 5 == 0:\n        total += i\nprint(total)\n",
    "div": "a = int(input())\nb = int(input())\nprint(a // b)\n",
    "rev": 'n = int(input())\nif n % 2 == 0:\n    print("Even")\nelse:\n    print("Odd")\n',
    "list": "n = int(input())\na = list(map(int, input().split()))\nprint(sum(a))\n",
    "digits": "def countDigits ( n ) :\n    count = 0\n    while n > 0 :\n        count += 1\n        n //= 10\n    return count\nprint ( countDigits ( 12345 ) )\nprint ( countDigits ( 7 ) )\n",
    "pairs": "while True:\n    a, b = map(int, input().split())\n    if a == 0 and b == 0:\n        break\n    print(a + b)\n",
    "middle": "def middle ( arr ) :\n    n = len ( arr )\n    return arr [ n / 2 ]\nprint ( middle ( [ 4 , 8 , 15 , 16 , 23 ] ) )\n",
}
CODE2 = {
    "java": {
        "sum": "import java.util.*;\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int n = sc.nextInt();\n        int total = 0;\n        for (int i = 1; i <= n; i++) {\n            if (i % 3 == 0 || i % 5 == 0) {\n                total += i;\n            }\n        }\n        System.out.println(total);\n    }\n}\n",
        "div": "import java.util.*;\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int a = sc.nextInt();\n        int b = sc.nextInt();\n        System.out.println(a / b);\n    }\n}\n",
        "rev": "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        String s = sc.next();\n        StringBuilder sb = new StringBuilder(s);\n        sb.reverse();\n        System.out.println(sb.toString());\n    }\n}\n",
        "digits": "class GFG {\n    static int countDigits(int n) {\n        int count = 0;\n        while (n > 0) {\n            count++;\n            n /= 10;\n        }\n        return count;\n    }\n    public static void main(String[] args) {\n        System.out.println(countDigits(12345));\n        System.out.println(countDigits(7));\n    }\n}\n",
        "pairs": "import java.util.*;\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        while (true) {\n            int a = sc.nextInt();\n            int b = sc.nextInt();\n            if (a == 0 && b == 0) {\n                break;\n            }\n            System.out.println(a + b);\n        }\n    }\n}\n",
        "middle": "class GFG {\n    static int middle(int[] arr) {\n        int n = arr.length;\n        return arr[n / 2];\n    }\n    public static void main(String[] args) {\n        System.out.println(middle(new int[]{4, 8, 15, 16, 23}));\n    }\n}\n",
        "list": "import java.util.*;\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int n = sc.nextInt();\n        long total = 0;\n        for (int i = 0; i < n; i++) {\n            total += sc.nextInt();\n        }\n        System.out.println(total);\n    }\n}\n",
    },
    "rust": {
        "sum": "use std::io;\n\nfn main() {\n    let mut buf = String::new();\n    io::stdin().read_line(&mut buf).unwrap();\n    let n: i64 = buf.trim().parse().unwrap();\n    let mut total: i64 = 0;\n    for i in 1..=n {\n        if i % 3 == 0 || i % 5 == 0 {\n            total += i;\n        }\n    }\n    println!(\"{}\", total);\n}\n",
        "div": "use std::io;\n\nfn read() -> i64 {\n    let mut buf = String::new();\n    io::stdin().read_line(&mut buf).unwrap();\n    buf.trim().parse().unwrap()\n}\n\nfn main() {\n    let a = read();\n    let b = read();\n    println!(\"{}\", a / b);\n}\n",
        "rev": "use std::io;\n\nfn main() {\n    let mut s = String::new();\n    io::stdin().read_line(&mut s).unwrap();\n    let r: String = s.trim().chars().rev().collect();\n    println!(\"{}\", r);\n}\n",
        "digits": "fn count_digits(mut n: i64) -> i64 {\n    let mut count = 0;\n    while n > 0 {\n        count += 1;\n        n /= 10;\n    }\n    count\n}\n\nfn main() {\n    println!(\"{}\", count_digits(12345));\n    println!(\"{}\", count_digits(7));\n}\n",
        "pairs": "use std::io::{self, Read};\n\nfn main() {\n    let mut s = String::new();\n    io::stdin().read_to_string(&mut s).unwrap();\n    let mut it = s.split_whitespace().map(|x| x.parse::<i64>().unwrap());\n    loop {\n        let a = it.next().unwrap();\n        let b = it.next().unwrap();\n        if a == 0 && b == 0 {\n            break;\n        }\n        println!(\"{}\", a + b);\n    }\n}\n",
        "middle": "fn middle(arr: &[i64]) -> i64 {\n    let n = arr.len();\n    arr[n / 2]\n}\n\nfn main() {\n    println!(\"{}\", middle(&[4, 8, 15, 16, 23]));\n}\n",
        "list": "use std::io::{self, Read};\n\nfn main() {\n    let mut s = String::new();\n    io::stdin().read_to_string(&mut s).unwrap();\n    let mut it = s.split_whitespace().map(|x| x.parse::<i64>().unwrap());\n    let n = it.next().unwrap() as usize;\n    let total: i64 = it.take(n).sum();\n    println!(\"{}\", total);\n}\n",
    },
}
READS = {"java": {"sum": "sc.nextInt()", "div": "two sc.nextInt() calls", "rev": "sc.next(), one string token",
                  "list": "n and then n integers with sc.nextInt(), on any lines", "digits": "", "middle": "", "pairs": ""},
         "rust": {"sum": "one line parsed as an integer", "div": "two lines, each parsed as an integer",
                  "rev": "one line kept as a string", "list": "all of standard input as whitespace-separated integers",
                  "digits": "", "middle": "", "pairs": ""}}
CONCLUDE = '# Conclusion: print the label that is_clone supports.\nprint("clone" if is_clone else "non-clone")'
TEST_LOOP = ("for s in {inputs}:\n    a, b = run_both(s)\n    print(repr(s), repr(a.out), repr(b.out))\n"
             "    if a.ok and b.ok:\n        is_clone = is_clone and same(a, b)\nprint(is_clone)")
SHOW = "print(r1.ok, r2.ok, err(r1), err(r2))"
COMPARE = "is_clone = is_clone and same(r1, r2)\nprint(repr(r1.out), repr(r2.out), is_clone)"

EXAMPLES = [
    ("sum", "clone", [
        "# Step 1: Both programs read one integer n (Code 1: int(input()); Code 2: {reads}).\n"
        "# Check that a typical input is accepted by both programs.\n"
        'r1, r2 = run_both("10")\nis_clone = r1.ok and r2.ok\n' + SHOW,
        "# Step 2: Both should print one integer, the sum. Compare the observed outputs directly.\n" + COMPARE,
        "# Step 3: The programs could differ at the loop bounds (is n itself included?) and for small n.\n"
        "# Test n = 1, n = 15 (a multiple of both 3 and 5) and n = 100. Compare only inputs both programs accept.\n"
        + TEST_LOOP.format(inputs='["1", "15", "100"]'),
        CONCLUDE,
    ], "Both programs accepted the same inputs, printed integers, and printed identical sums for n = 1, 10, 15 and 100, including the bound n itself."),
    ("div", "non-clone", [
        "# Step 1: Both programs read two integers (Code 1: two input() calls; Code 2: {reads}).\n"
        "# Check that a typical input is accepted by both programs.\n"
        'r1, r2 = run_both("7\\n2")\nis_clone = r1.ok and r2.ok\n' + SHOW,
        "# Step 2: Both should print one integer, the quotient. Compare the observed outputs directly.\n" + COMPARE,
        "# Step 3: Code 1 uses // (rounds down) and Code 2 uses integer / (rounds toward zero). Both read any\n"
        "# integers, so negative operands are valid input; the programs can differ only when the quotient is\n"
        "# negative, so test negative operands and an exact division.\n"
        + TEST_LOOP.format(inputs='["-7\\n2", "7\\n-2", "6\\n3"]'),
        CONCLUDE,
    ], "On -7 and 2, Code 1 printed -4 and Code 2 printed -3: the programs round negative quotients differently, so they do not compute the same function."),
    ("rev", "non-clone", [
        "# Step 1: Code 1 reads one integer; Code 2 reads {reads}. Both accept a digit string,\n"
        "# so check that a typical input is accepted by both programs.\n"
        'r1, r2 = run_both("5")\nis_clone = r1.ok and r2.ok\n' + SHOW,
        "# Step 2: Code 1 should print a word (Even or Odd); Code 2 prints the reversed input.\n"
        "# Compare the observed outputs directly.\n" + COMPARE,
        "# is_clone is already False: the programs do not even print the same kind of output, so they\n"
        "# solve different tasks and no behavioural test is needed.\n" + CONCLUDE.split("\n", 1)[1],
    ], "Code 1 prints a parity word while Code 2 prints its input reversed; on the same input 5 they printed 'Odd' and '5', different kinds of output for unrelated tasks."),
    ("list", "clone", [
        "# Step 1: Code 1 reads n, then the n numbers on the next line; Code 2 reads {reads}.\n"
        "# Check a typical input on both programs.\n"
        'r1, r2 = run_both("3 1 2 3")\nis_clone = r1.ok and r2.ok\n' + SHOW,
        "# Step 2: Only Code 1 failed, with a ValueError on int(): the input was not valid for it, because\n"
        "# Code 1 reads n on a line of its own.\n"
        "# A one-sided failure is not evidence of a difference. Retry with the format both programs accept.\n"
        'r1, r2 = run_both("3\\n1 2 3")\nis_clone = r1.ok and r2.ok\n' + SHOW,
        "# Step 3: Both should print one integer, the sum. Compare the observed outputs directly.\n" + COMPARE,
        "# Step 4: Any integers are valid here. The programs could differ for a single number, for negative\n"
        "# numbers and for zeros. Test those cases, written in the format both programs accept.\n"
        + TEST_LOOP.format(inputs='["1\\n5", "3\\n-1 -2 4", "2\\n0 0"]'),
        CONCLUDE,
    ], "The first input was written in a format only Code 2 accepts; with valid input both programs printed identical sums, including for a single number, negative numbers and zeros."),
    ("digits", "clone", [
        "# Step 1: Neither program reads input: both call countDigits on the fixed arguments 12345 and 7\n"
        "# and print the results. Run both programs with empty input.\n"
        'r1, r2 = run_both("")\nis_clone = r1.ok and r2.ok\n' + SHOW,
        "# Step 2: Both should print two integers, the digit counts of 12345 and 7. With fixed arguments\n"
        "# the printed outputs are the whole observable behaviour, so compare them directly.\n" + COMPARE,
        CONCLUDE,
    ], "Neither program reads input; both count the digits of 12345 and 7 and printed the same results, 5 and 1."),
    ("middle", "clone", [
        "# Step 1: Neither program reads input: both print the middle element of [4, 8, 15, 16, 23].\n"
        "# Run both programs with empty input.\n"
        'r1, r2 = run_both("")\nis_clone = r1.ok and r2.ok\n' + SHOW,
        "# Step 2: Only Code 1 failed, with a TypeError rather than an input error. Retry once with another\n"
        "# input to see whether the input was the problem.\n"
        'r1, r2 = run_both("5")\n' + SHOW,
        "# Code 1 fails again with the same TypeError while Code 2 runs: n / 2 is integer division in Python 2 but\n"
        "# a float in Python 3, so Code 1 itself is broken in this runtime and execution cannot decide.\n"
        "# Decide by reading the code: both return arr[n / 2] with integer division, the middle element of\n"
        "# the same fixed list, so they implement the same function. Record that decision in is_clone.\n"
        "is_clone = True  # decided by reading the code; Code 1 cannot run in Python 3\nprint(is_clone)",
        CONCLUDE,
    ], "Code 1 fails on every input only because n / 2 is a float in Python 3; read as intended (integer division), both programs return the middle element of the same list, 15."),
    ("pairs", "clone", [
        "# Step 1: Both programs read pairs of integers a b and print a + b for each pair.\n"
        "# Check a typical input on both programs.\n"
        'r1, r2 = run_both("1 2\\n3 4")\nis_clone = r1.ok and r2.ok\n' + SHOW,
        "# Step 2: BOTH programs failed at the end of the input, so the input is wrong for the task, not a\n"
        "# difference between the programs: both keep reading pairs until the line 0 0. Fix the input by\n"
        "# adding that terminating line.\n"
        'r1, r2 = run_both("1 2\\n3 4\\n0 0")\nis_clone = r1.ok and r2.ok\n' + SHOW,
        "# Step 3: Both should print one sum per pair. Compare the observed outputs directly.\n" + COMPARE,
        "# Step 4: Valid inputs always end with 0 0. The programs could differ when the input has no pairs\n"
        "# before 0 0, when only one number of a pair is zero, and for large sums. Test those cases.\n"
        + TEST_LOOP.format(inputs='["0 0", "5 0\\n0 7\\n0 0", "1000000 2000000\\n0 0"]'),
        CONCLUDE,
    ], "The first input lacked the terminating line 0 0 that both programs read up to; with valid input both printed the same sums, including no pairs, pairs with one zero and large values."),
]


def main():
    out_dir = ROOT / "rstar_deepthink/few_shots"
    for L, Lname in (("java", "Java"), ("rust", "Rust")):
        shots = []
        for key, answer, steps, reason in EXAMPLES:
            question = TEMPLATE.format(c1=PY[key], c2=CODE2[L][key], Lname=Lname, ltag=L)
            text = f"Question: {question}\n"
            done = []
            for step in steps:
                step = step.replace("{reads}", READS[L][key])
                obs = run_codestep(question, done, step)
                text += f"<code>\n{step}\n<end_of_step>\n<output>\n{obs}\n<end_of_output>\n"
                done.append(step)
            assert obs.splitlines()[0] == answer, (L, key, obs)
            text += f"<answer>\nFinal decision: \\boxed{{{answer}}}\nReason: {reason}\n<end_of_answer>"
            shots.append(text)
        path = out_dir / f"few_shot_codestep_python_{L}_v3.json"
        json.dump(shots, open(path, "w"), indent=1, ensure_ascii=False)
        print(f"wrote {path} ({len(shots)} examples)")


if __name__ == "__main__":
    main()

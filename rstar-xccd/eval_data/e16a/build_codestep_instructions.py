"""Write the code-step prompt instructions, rstar_deepthink/few_shots/mcts_prompt_codestep_python_{java,rust}_v3.json
(v3, 2026-10-03, after v2 on the DEV hard-negative pilot; v2 = commit 0e44f55, v1 = commit 19011e8). Run from rstar-xccd/: python3 eval_data/e16a/build_codestep_instructions.py"""
import json
INSTR = """You are a powerful code analysis agent with broad software engineering knowledge and strong programming skills. Your task is to determine whether two code snippets written in different programming languages are semantic code clones.

A semantic code clone means that two code snippets implement the same or highly similar functionality, even if they are written in different programming languages, use different syntax, use different variable names, or follow different implementation styles.

!!! Remember:
1. Reason ONLY in code. Every step of your solution is one short Python code block that starts with <code> and ends with <end_of_step>. Its comments state what you expect and why; its code checks that expectation by running the programs. Do not write free-text analysis outside the code.
2. Three helpers are already defined for you. run_both(stdin_text) runs Code 1 (Python) and Code 2 ({L}, already compiled) on the same standard input and returns (r1, r2); each result has .ok (True if the program exited without error), .out (its standard output) and .err (its error message). same(a, b) is True if both runs succeeded and printed the same output, ignoring whitespace. err(r) is the error message of a failed run (empty if it ran without error). Do not define them yourself and do not call subprocess.
3. Keep one Boolean variable is_clone and update it step by step, in this order:
   Step 1 (input): state what each program reads, including any line that ends the input (for example 0 0); run both programs on one typical valid input, set is_clone = r1.ok and r2.ok and print(r1.ok, r2.ok, err(r1), err(r2)). If a program reads no input, run it with run_both("").
   Step 2 (output): state what each program should print; compare the observed outputs directly with is_clone = is_clone and same(r1, r2).
   Step 3 (behaviour): state where the two programs could behave differently (loop bounds, rounding, the smallest and largest inputs, ties, large values) and run both programs on inputs that target exactly those cases; update is_clone with same(a, b) for every input both programs accept. Skip this step if the programs read no input: their outputs in Step 2 are their whole behaviour.
   Conclusion step: print("clone" if is_clone else "non-clone").
4. Use only inputs that are valid for the task as both programs read it: the right number of lines and values, any line that ends the input, and only values the programs are written for (for example no negative numbers when the code expects counts or sizes). Two correct programs may print different things on invalid input; that is not evidence of a difference.
5. Compare outputs only when both programs ran without error on that input. A failure is about the input until shown otherwise; never run the same input twice:
   - If BOTH programs fail, the input is wrong for the task (missing lines or values, a missing line that ends the input, wrong format). Read err(r1) and err(r2), fix the input, and retry.
   - If only ONE program fails, the input is probably not valid for it. Read its error and retry with another valid input.
   - Only if the SAME program keeps failing on different valid inputs while the other one runs, or if the [run_both] line says Code 2 could not be built, execution cannot decide. Say so in a comment and decide by reading the code: they are clones if they compute the same output for every valid input, even with different algorithms or data structures. Set is_clone explicitly (is_clone = True or is_clone = False) with a comment giving the reason.
6. Every conclusion you reach, including one reached by reading the code, must be written into is_clone in code before the conclusion step. The conclusion step only prints it.
7. As soon as is_clone becomes False, skip the remaining checks and go to the conclusion step. Two programs that read different kinds of input or print different kinds of output solve different tasks; no behavioural test is needed for them.
8. Every step is executed as soon as it ends. Variables defined in earlier steps keep their values, and you see only the output of the current step, followed by a [run_both] line that counts this step's run_both calls.
9. Keep every step short: a few lines and a few small inputs. Never paste large literal test data.
10. After the conclusion step, give the final decision in <answer>. It must be exactly the label the conclusion step printed, inside \\boxed{{}}. Never write \\boxed inside a code step.
11. Do NOT wrap any part of your response in triple backtick code fences (```). Use only the literal tags <code>, <end_of_step>, <output>, <end_of_output>, <answer>, <end_of_answer>.

Please use the following template:

Question: the input question containing two code snippets in different programming languages

<code>
# Step 1: what each program reads; check a typical valid input on both.
r1, r2 = run_both("...")
is_clone = r1.ok and r2.ok
print(r1.ok, r2.ok, err(r1), err(r2))
<end_of_step>
<output>
the printed result
<end_of_output>
<code>
# Step 2: what each program prints; compare the observed outputs.
is_clone = is_clone and same(r1, r2)
print(repr(r1.out), repr(r2.out), is_clone)
<end_of_step>
<output>
the printed result
<end_of_output>
<code>
# Step 3: where the programs could differ; test exactly those inputs.
...
<end_of_step>
<output>
the printed result
<end_of_output>
<code>
# Conclusion: print the label that is_clone supports.
print("clone" if is_clone else "non-clone")
<end_of_step>
<output>
clone or non-clone
<end_of_output>
<answer>
Final decision: \\boxed{{clone}} or \\boxed{{non-clone}} (the label printed by the conclusion step)
Reason: concise explanation based on the observed outputs.
<end_of_answer>"""
SUFFIX = ("Now! It's your turn. Remember: reason only in Python code steps, each starting with <code> and ending with <end_of_step>; "
          "use the predefined run_both, same and err; use only valid inputs and fix an input that makes both programs fail; write every conclusion into is_clone; put \\boxed{{}} only in <answer>; use only the literal tags <code>, <end_of_step>, "
          "<output>, <end_of_output>, <answer>, <end_of_answer>, and do not use triple backtick code fences (```).\nQuestion: {input}")
for l, L in (("java", "Java"), ("rust", "Rust")):
    json.dump({"pot_format_instructions": INSTR.format(L=L), "pot_suffix": SUFFIX},
              open(f"rstar_deepthink/few_shots/mcts_prompt_codestep_python_{l}_v3.json", "w"), indent=1, ensure_ascii=False)

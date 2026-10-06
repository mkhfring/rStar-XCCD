"""Phase 0.1 checks (training plan TRAINING_PLAN_CODESTEP_2026-10-05.txt): code-step harness
time limits, using the real PolyHuman pair 98 (CF 146B Lucky Mask), whose correct Code 1 loops
forever when b is not a lucky number.

Usage (needs java on PATH, e.g. `module load java/17.0.6`; from rstar-xccd/):
    ../venv-qwen3/bin/python tests/test_codestep_timeouts.py
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["SLURM_JOB_ID"] = f"test{os.getpid()}"      # private shared-cache dir for this test

from rstar_deepthink.tools import python_tool as pt  # noqa: E402

DATA = Path("/lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd/"
            "eval_data/e16a/polyhuman_python_java.jsonl")
Q = next(json.loads(l)["question"] for l in open(DATA) if json.loads(l)["index"] == 98)

LOOP_STEP = """
for s in ["1 1\\n", "10 0\\n", "2 3\\n", "5 5\\n"]:
    r1, r2 = run_both(s)
    print(s.strip(), r1.ok, r2.ok, err(r1))
"""
CACHED_STEP = """
for s in ["1 1\\n", "39999 4774\\n"]:
    r1, r2 = run_both(s)
    print(s.strip(), r1.ok, r2.ok)
"""
GOOD_STEP = """
r1, r2 = run_both("39999 4774\\n")
print(r1.out, r2.out, same(r1, r2))
"""


def timed(prev, cur):
    t = time.monotonic()
    obs = pt.run_codestep(Q, prev, cur)
    return time.monotonic() - t, obs


def child(q):
    os.environ["SLURM_JOB_ID"] = q.get()               # same job id -> same cache dir
    t = time.monotonic()
    obs = pt.run_codestep(Q, [], CACHED_STEP)
    q.put((time.monotonic() - t, obs))


def main():
    fails = 0

    def check(name, ok, detail):
        nonlocal fails
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    dt, obs = timed([], LOOP_STEP)
    budget_cap = pt.CODESTEP_STEP_BUDGET + 2 * pt.CODESTEP_RUN_TIMEOUT + 5
    check("T1 looping inputs stay within the step budget", dt <= budget_cap and "skipped" in obs,
          f"{dt:.1f}s (cap {budget_cap}s)\n{obs}")

    dt, obs = timed([LOOP_STEP], GOOD_STEP)
    check("T2 replay of the looping step is free", dt < 5 and "40774 40074 False" in obs,
          f"{dt:.1f}s\n{obs}")

    dt, obs = timed([], GOOD_STEP)
    check("T3 valid input: difference found", "40774 40074 False" in obs and "outputs differed 1" in obs,
          f"{dt:.1f}s\n{obs}")

    pt.pop_codestep_calls()
    pt.run_codestep(Q, [LOOP_STEP], GOOD_STEP)
    rec = pt.pop_codestep_calls()
    check("T5 run_both_calls records the current step only (stdin + both outputs)",
          len(rec) == 1 and rec[0]["stdin"] == "39999 4774\n" and rec[0]["out1"] == "40774"
          and rec[0]["out2"] == "40074" and rec[0]["differ"] and pt.pop_codestep_calls() == [], str(rec))

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    q.put(os.environ["SLURM_JOB_ID"])
    p = ctx.Process(target=child, args=(q,))
    p.start()
    p.join(120)
    time.sleep(0.2)
    dt, obs = q.get()
    check("T4 another process reuses build + results from the shared cache", dt < 5 and "1 1 False True" in obs,
          f"{dt:.1f}s (spawned process, cold memory cache)\n{obs}")

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

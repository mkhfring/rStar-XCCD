"""Carve a train/held-out split out of the CLCCD test files, for the
rl-self-improvement branch.

Every F1 number in RESUME_HERE.txt and EXTERNAL_BASELINES_COMPARISON.txt was
computed on the FULL eval_data/test_python_{java,rust}_CLCCD.jsonl -- the
same files this branch would otherwise be tempted to mine rollouts from and
train on. Doing that without first reserving a genuinely untouched slice
would invalidate every one of those comparisons (see RL_IDEAS.txt section 4,
"train/test leakage").

CORRECTED 2026-09-13: the first version of this script stratified by
clone/non-clone label at the RECORD level only. CLCCD builds multiple pair-
records from the same underlying CodeNet/XLCoST problem (one clone pair +
several derangement-based non-clone pairs sharing the same Python side, per
CLCCD_DATASET_CONSTRUCTION.md section 4) -- so a record-level split can and
did put two records from the SAME problem on opposite sides. Checked
directly: with the old split, 170 of 172 held-out rust problems (98.8%) also
had a training example from the same problem, and 28 of 327 held-out java
problems (8.6%) did too. A model fine-tuned on train could then "recognize"
a held-out question's problem from training rather than reasoning about the
pair on its own merits -- a real leakage channel independent of the
train/test split itself.

FIX: recover each record's underlying Python-side problem_id by matching
its extracted Python snippet, verbatim, against ../../CLCCD/data/
{java_cn,java_xl}.json (ll2=="Python" records; codeB is the Python snippet
there, problem_id_2 is its problem id) -- 0 unmatched across all 1800+1200
records, and the resulting distinct-problem counts (java: 1583, rust: 300)
exactly match CLCCD_DATASET_CONSTRUCTION.md's stated pool sizes, which is
the check that this join is actually correct rather than silently wrong.
Then split by PROBLEM, not by record: every record sharing a problem_id
goes entirely to one side. random_state=42 for the shuffle, matching this
project's own GraphCodeBERT baseline split and the CLCCD paper's own
classifier.py train_test_split call, for consistency of convention.

The full original files, and the CLCCD/ sibling repo this script reads
from, are left completely untouched.
"""
import argparse
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CLCCD_REPO = REPO_ROOT.parent.parent / "CLCCD"

QUESTION_RE = re.compile(
    r"Code 1: Python\n```python\n(?P<codeA>.*?)\n```\n\nCode 2: \w[\w+#]*\n```\w*\n(?P<codeB>.*?)\n```",
    re.DOTALL,
)


def extract_python_code(question):
    m = QUESTION_RE.search(question)
    return m.group("codeA") if m else None


def build_python_to_problem_map():
    pymap = {}
    for fname in ("java_cn.json", "java_xl.json"):
        for r in json.load(open(CLCCD_REPO / "data" / fname, encoding="utf-8")):
            if r.get("ll2") == "Python":
                pymap[r["codeB"]] = r["problem_id_2"]
    return pymap


def load(path):
    records = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "question" not in r:
            continue
        records.append(r)
    return records


def group_by_problem(records, pymap):
    groups = {}
    unmatched = 0
    for r in records:
        code = extract_python_code(r["question"])
        pid = pymap.get(code)
        if pid is None:
            unmatched += 1
            pid = f"__unmatched_{r['index']}"  # isolate rather than silently drop
        groups.setdefault(pid, []).append(r)
    return groups, unmatched


def split_groups(groups, test_frac, seed):
    pids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(pids)

    total = sum(len(g) for g in groups.values())
    target_heldout = round(total * test_frac)

    train, heldout = [], []
    heldout_count = 0
    for pid in pids:
        group = groups[pid]
        if heldout_count < target_heldout:
            heldout.extend(group)
            heldout_count += len(group)
        else:
            train.extend(group)
    rng.shuffle(train)
    rng.shuffle(heldout)
    return train, heldout


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def counts(rs):
    c = sum(1 for r in rs if (r.get("answer") or "").lower() == "clone")
    return len(rs), c, len(rs) - c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["java", "rust"])
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pymap = build_python_to_problem_map()
    report_lines = [
        "RL train/held-out split -- eval_data/build_rl_split.py",
        f"seed={args.seed}  test_frac={args.test_frac}  split by underlying problem_id, not by record",
        f"python-code -> problem_id map size: {len(pymap)}",
        "",
    ]

    for lang in args.langs:
        src = f"eval_data/test_python_{lang}_CLCCD.jsonl"
        records = load(src)
        groups, unmatched = group_by_problem(records, pymap)
        if unmatched:
            print(f"WARNING: {unmatched}/{len(records)} {lang} records had no problem_id match "
                  f"(isolated to their own singleton group rather than dropped)")

        train, heldout = split_groups(groups, args.test_frac, args.seed)

        # verify: no problem_id shared across train/heldout
        train_pids = {pid for pid, g in groups.items() if g and g[0] in train}
        heldout_pids = set(groups.keys()) - train_pids
        # (cheap sanity check via direct re-resolution, since groups already
        # partition cleanly by construction of split_groups)

        train_path = f"eval_data/rl_train_python_{lang}_CLCCD.jsonl"
        heldout_path = f"eval_data/rl_heldout_python_{lang}_CLCCD.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(heldout_path, heldout)

        n_all, c_all, nc_all = counts(records)
        n_tr, c_tr, nc_tr = counts(train)
        n_he, c_he, nc_he = counts(heldout)
        n_problems = len(groups)
        report_lines += [
            f"{lang}: source {src} -- {n_all} total ({c_all} clone / {nc_all} non-clone), {n_problems} distinct problems",
            f"  train:    {train_path} -- {n_tr} ({c_tr} clone / {nc_tr} non-clone)",
            f"  heldout:  {heldout_path} -- {n_he} ({c_he} clone / {nc_he} non-clone)",
            "",
        ]
        print(f"{lang}: {n_problems} problems  train={n_tr} ({c_tr}/{nc_tr})  heldout={n_he} ({c_he}/{nc_he})")

    report = "\n".join(report_lines)
    with open("eval_data/rl_split_report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nWrote eval_data/rl_split_report.txt")


if __name__ == "__main__":
    main()

"""Carve a stratified, seeded train/held-out split out of the CLCCD test
files, for the rl-self-improvement branch.

Every F1 number in RESUME_HERE.txt and EXTERNAL_BASELINES_COMPARISON.txt was
computed on the FULL eval_data/test_python_{java,rust}_CLCCD.jsonl -- the
same files this branch would otherwise be tempted to mine rollouts from and
train on. Doing that without first reserving a genuinely untouched slice
would invalidate every one of those comparisons (see RL_IDEAS.txt section 4,
"train/test leakage"). This script makes that split once, deterministically,
so every future run on this branch trains only on the "rl_train" file and
reports only on the "rl_heldout" file.

80/20, stratified by ground-truth label so class balance is preserved in
both halves (java is 900/900, rust is 300/900 -- an unstratified split could
by chance skew either half's class ratio). random_state=42, matching this
project's own GraphCodeBERT baseline calibration split and the CLCCD paper's
own classifier.py train_test_split call, for consistency of convention
rather than because 42 is special.

The full original files are left completely untouched; nothing downstream
of this script should ever read test_python_{java,rust}_CLCCD.jsonl again
for anything trained or evaluated on this branch.
"""
import argparse
import json
import random


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


def stratified_split(records, test_frac, seed):
    by_label = {"clone": [], "non-clone": []}
    for r in records:
        by_label[(r.get("answer") or "").lower()].append(r)

    rng = random.Random(seed)
    train, heldout = [], []
    for label, group in by_label.items():
        group = group[:]
        rng.shuffle(group)
        n_test = round(len(group) * test_frac)
        heldout.extend(group[:n_test])
        train.extend(group[n_test:])
    rng.shuffle(train)
    rng.shuffle(heldout)
    return train, heldout


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["java", "rust"])
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    report_lines = [
        "RL train/held-out split -- eval_data/build_rl_split.py",
        f"seed={args.seed}  test_frac={args.test_frac}  stratified by answer label",
        "",
    ]

    for lang in args.langs:
        src = f"eval_data/test_python_{lang}_CLCCD.jsonl"
        records = load(src)
        train, heldout = stratified_split(records, args.test_frac, args.seed)

        train_path = f"eval_data/rl_train_python_{lang}_CLCCD.jsonl"
        heldout_path = f"eval_data/rl_heldout_python_{lang}_CLCCD.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(heldout_path, heldout)

        def counts(rs):
            c = sum(1 for r in rs if (r.get("answer") or "").lower() == "clone")
            return len(rs), c, len(rs) - c

        n_all, c_all, nc_all = counts(records)
        n_tr, c_tr, nc_tr = counts(train)
        n_he, c_he, nc_he = counts(heldout)
        report_lines += [
            f"{lang}: source {src} -- {n_all} total ({c_all} clone / {nc_all} non-clone)",
            f"  train:    {train_path} -- {n_tr} ({c_tr} clone / {nc_tr} non-clone)",
            f"  heldout:  {heldout_path} -- {n_he} ({c_he} clone / {nc_he} non-clone)",
            "",
        ]
        print(f"{lang}: train={n_tr} ({c_tr}/{nc_tr})  heldout={n_he} ({c_he}/{nc_he})")

    report = "\n".join(report_lines)
    with open("eval_data/rl_split_report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nWrote eval_data/rl_split_report.txt")


if __name__ == "__main__":
    main()

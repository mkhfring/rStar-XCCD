"""Score mined MCTS rollouts (is_sampling=True mining runs on the
rl-self-improvement branch) for how many leaves are SAFE to imitate for
rejection-sampling SFT, and optionally emit those leaves as SFT-ready
training examples.

See SFT_DATA_PROCESS.txt section 4 for the full rationale. Short version:
a leaf is SAFE if its final_answer matches ground truth AND EITHER some
ancestor step genuinely executed code (exec_outcome_of()=="ran"), or no
ancestor ever attempted to run code at all (pure signature-level reasoning,
nothing to fabricate). A leaf is DANGEROUS -- excluded -- if some ancestor
attempted code but never got a confirmed "ran" (no_code/blocked/
assertion_failed/error), yet the leaf still concluded correctly anyway; that
shape is exactly what the mcts.py dropped-observation bug (RESUME_HERE.txt
section 3f) let happen, so a correct-looking leaf built on it is not
trustworthy to imitate even though the label happens to be right.

STRICT MODE (default since 2026-09-24, SFT_DATA_PROCESS.txt section 9):
section 4's rule let through leaves whose only code step printed NOTHING
(exec_outcome_of() maps an empty observation to None, i.e. "never tried
code", so they counted as pure reasoning) -- 69/168 leaves of the first
hard-codenet SFT sets. Strict mode additionally runs trace_checks.leaf_flags()
on every correct leaf and assigns a tier: A (verified by real execution
output), B (usable but weaker: empty code output, copied Code-2 summary, or
pure reasoning), X (dropped: section 4 DANGEROUS, answer claims tests that
never ran, clone concluded from an input on which the two programs actually
diverge, or a print-only "test" on a clone answer). --min_tier B keeps A+B,
--min_tier A keeps only A. --max_per_question caps near-duplicate leaves per
question (A before B). --legacy reproduces section 4's counts exactly.

Usage:
    python score_rl_rollouts.py FILE [FILE ...]
    python score_rl_rollouts.py FILE [FILE ...] --emit_sft OUT.jsonl
    python score_rl_rollouts.py FILE [FILE ...] --diff_test DIFF.jsonl \\
        --min_tier B --max_per_question 4 --emit_sft OUT.jsonl

--emit_sft writes one JSON object per safe leaf: {"question", "answer",
"source_file", "index", "leaf_tag", "target_text"}. target_text is the
concatenation (joined by "\n", matching step_delim in every config used so
far) of every node's "text" field from the root to that leaf, in order --
i.e. the full generated trace an SFT run would train the model to produce
for that question. Building the actual chat-formatted training example
(tokenizer.apply_chat_template with this as the assistant turn) is left to
the fine-tuning script, not done here.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_clone_results import normalize_label, node_sort_key, exec_outcome_of
from trace_checks import leaf_flags, exec_outcomes, leaf_tier

TIER_RANK = {"A": 0, "B": 1, "X": 2}


def ancestry(tag):
    parts = tag.split(".")
    for i in range(len(parts), 0, -1):
        yield ".".join(parts[:i])


def leaf_safety(tree, tag, numeric_nodes):
    outcomes_seen = set()
    for anc_tag in ancestry(tag):
        anc = numeric_nodes.get(anc_tag) or tree.get(anc_tag)
        if isinstance(anc, dict):
            oc = exec_outcome_of(anc)
            if oc is not None:
                outcomes_seen.add(oc)
    if "ran" in outcomes_seen:
        return "safe"
    if not outcomes_seen:
        return "safe"
    return "dangerous"


def build_target_text(tree, tag, numeric_nodes, step_delim="\n"):
    parts = tag.split(".")
    chunks = []
    for i in range(1, len(parts) + 1):
        anc_tag = ".".join(parts[:i])
        anc = numeric_nodes.get(anc_tag) or tree.get(anc_tag)
        if isinstance(anc, dict):
            chunks.append(anc.get("text") or "")
    return step_delim.join(chunks)


def score_file(path, sft_writer=None, legacy=False, diff_records=None, min_tier="B",
               max_per_question=None):
    n_questions = 0
    q_safe = 0
    total_safe = 0
    total_dangerous = 0
    flag_counts = {}

    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "question" not in r:
            continue
        n_questions += 1
        truth = (r.get("answer") or "").lower()
        tree = r["rstar"]
        numeric_nodes = {tag: node for tag, node in tree.items() if node_sort_key(tag) is not None}
        diff_record = (diff_records or {}).get(r.get("index"))

        kept = []
        for tag, node in numeric_nodes.items():
            fa = (node.get("final_answer") or "").strip()
            if not fa:
                continue
            label = normalize_label(fa)
            if label is None or label != truth:
                continue
            if legacy:
                if leaf_safety(tree, tag, numeric_nodes) == "safe":
                    kept.append((tag, "A", []))
                else:
                    total_dangerous += 1
                continue
            flags = leaf_flags(numeric_nodes, tag, label, diff_record)
            tier = leaf_tier(flags, exec_outcomes(numeric_nodes, tag))
            for f in flags:
                flag_counts[f] = flag_counts.get(f, 0) + 1
            if TIER_RANK[tier] <= TIER_RANK[min_tier]:
                kept.append((tag, tier, sorted(flags)))
            else:
                total_dangerous += 1

        kept.sort(key=lambda k: (TIER_RANK[k[1]], node_sort_key(k[0])))
        if max_per_question:
            kept = kept[:max_per_question]
        for tag, tier, flags in kept:
            if sft_writer is not None:
                rec = {
                    "question": r["question"],
                    "answer": r["answer"],
                    "source_file": str(path),
                    "index": r.get("index"),
                    "leaf_tag": tag,
                    "target_text": build_target_text(tree, tag, numeric_nodes),
                }
                if not legacy:
                    rec["tier"] = tier
                    rec["flags"] = flags
                sft_writer.write(json.dumps(rec) + "\n")

        total_safe += len(kept)
        if kept:
            q_safe += 1

    return dict(n_questions=n_questions, q_safe=q_safe, total_safe=total_safe,
                total_dangerous=total_dangerous, flag_counts=flag_counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--emit_sft", help="path to write safe-leaf training examples (JSONL)")
    ap.add_argument("--legacy", action="store_true", help="section 4 rule only (reproduces pre-2026-09-24 counts)")
    ap.add_argument("--diff_test", action="append", default=[],
                    help="diff_test_pairs.py output(s); enables the divergent_evidence check")
    ap.add_argument("--min_tier", choices=["A", "B"], default="B")
    ap.add_argument("--max_per_question", type=int, default=None)
    args = ap.parse_args()

    diff_records = {}
    for p in args.diff_test:
        for line in open(p, encoding="utf-8"):
            rec = json.loads(line)
            diff_records[rec["index"]] = rec

    sft_writer = open(args.emit_sft, "w", encoding="utf-8") if args.emit_sft else None
    try:
        grand = dict(n_questions=0, q_safe=0, total_safe=0, total_dangerous=0)
        grand_flags = {}
        for path in args.files:
            stats = score_file(path, sft_writer, legacy=args.legacy, diff_records=diff_records,
                               min_tier=args.min_tier, max_per_question=args.max_per_question)
            for k in grand:
                grand[k] += stats[k]
            for f, c in stats["flag_counts"].items():
                grand_flags[f] = grand_flags.get(f, 0) + c
            n = stats["n_questions"]
            print(f"{path}: n={n} safe_questions={stats['q_safe']}/{n} "
                  f"({stats['q_safe']/n:.1%})  safe_leaves={stats['total_safe']} "
                  f"dangerous_excluded={stats['total_dangerous']}")
        if len(args.files) > 1:
            n = grand["n_questions"]
            print(f"TOTAL: n={n} safe_questions={grand['q_safe']}/{n} ({grand['q_safe']/n:.1%})  "
                  f"safe_leaves={grand['total_safe']} dangerous_excluded={grand['total_dangerous']}")
        if grand_flags:
            print("flags on correct leaves:", dict(sorted(grand_flags.items())))
    finally:
        if sft_writer is not None:
            sft_writer.close()
            print(f"Wrote SFT examples to {args.emit_sft}")


if __name__ == "__main__":
    main()

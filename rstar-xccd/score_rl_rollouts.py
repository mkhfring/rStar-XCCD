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

Usage:
    python score_rl_rollouts.py FILE [FILE ...]
    python score_rl_rollouts.py FILE [FILE ...] --emit_sft OUT.jsonl

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


def score_file(path, sft_writer=None):
    n_questions = 0
    q_safe = 0
    total_safe = 0
    total_dangerous = 0

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

        n_safe_this_q = 0
        for tag, node in numeric_nodes.items():
            fa = (node.get("final_answer") or "").strip()
            if not fa:
                continue
            label = normalize_label(fa)
            if label is None or label != truth:
                continue
            safety = leaf_safety(tree, tag, numeric_nodes)
            if safety == "safe":
                n_safe_this_q += 1
                if sft_writer is not None:
                    sft_writer.write(json.dumps({
                        "question": r["question"],
                        "answer": r["answer"],
                        "source_file": str(path),
                        "index": r.get("index"),
                        "leaf_tag": tag,
                        "target_text": build_target_text(tree, tag, numeric_nodes),
                    }) + "\n")
            else:
                total_dangerous += 1

        total_safe += n_safe_this_q
        if n_safe_this_q > 0:
            q_safe += 1

    return dict(n_questions=n_questions, q_safe=q_safe, total_safe=total_safe, total_dangerous=total_dangerous)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--emit_sft", help="path to write safe-leaf training examples (JSONL)")
    args = ap.parse_args()

    sft_writer = open(args.emit_sft, "w", encoding="utf-8") if args.emit_sft else None
    try:
        grand = dict(n_questions=0, q_safe=0, total_safe=0, total_dangerous=0)
        for path in args.files:
            stats = score_file(path, sft_writer)
            for k in grand:
                grand[k] += stats[k]
            n = stats["n_questions"]
            print(f"{path}: n={n} safe_questions={stats['q_safe']}/{n} "
                  f"({stats['q_safe']/n:.1%})  safe_leaves={stats['total_safe']} "
                  f"dangerous_excluded={stats['total_dangerous']}")
        if len(args.files) > 1:
            n = grand["n_questions"]
            print(f"TOTAL: n={n} safe_questions={grand['q_safe']}/{n} ({grand['q_safe']/n:.1%})  "
                  f"safe_leaves={grand['total_safe']} dangerous_excluded={grand['total_dangerous']}")
    finally:
        if sft_writer is not None:
            sft_writer.close()
            print(f"Wrote SFT examples to {args.emit_sft}")


if __name__ == "__main__":
    main()

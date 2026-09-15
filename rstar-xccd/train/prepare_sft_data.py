"""Combine one or more score_rl_rollouts.py --emit_sft outputs (fields:
question, answer, source_file, index, leaf_tag, target_text) into a single
JSONL file in the {instruction, output} shape train_SFT.py's
SupervisedDataset expects. No other transformation is applied -- the
prompt wrapping (PROMPT_DICT's pot_suffix) happens inside train_SFT.py.

Usage:
    python train/prepare_sft_data.py FILE [FILE ...] --output_file OUT.jsonl
"""
import argparse
import json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="score_rl_rollouts.py --emit_sft output file(s)")
    p.add_argument("--output_file", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_in = 0
    with open(args.output_file, "w") as out:
        for path in args.files:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    n_in += 1
                    out.write(json.dumps({"instruction": r["question"], "output": r["target_text"]}) + "\n")
    print(f"{n_in} examples from {len(args.files)} file(s) -> {args.output_file}")

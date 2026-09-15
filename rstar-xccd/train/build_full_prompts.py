"""Rebuild score_rl_rollouts.py --emit_sft output into the {instruction,
output} shape train_SFT.py expects, with "instruction" set to the ACTUAL
full prompt the policy model saw when target_text was generated -- system
message, format rules, and the few-shot worked examples, run through the
model's real chat template (rstar_deepthink.agents.utils.rstar_prompt_wrap,
the exact function main.py's MCTS search itself calls) -- not just the bare
question text.

Why this matters: train_SFT.py's own PROMPT_DICT wraps "instruction" with a
generic "<|user|>:\\n{instruction}\\n<|assistant|>: Let's think step by
step and solve the problem with code." string that has nothing to do with
how these traces were actually produced (no system message, no few-shot
examples, not even the model's real chat-template markup). Training on that
mismatched prompt shape would teach the model to continue from an input it
will never see again at eval time. This script produces the real prompt
instead; train_SFT.py's PROMPT_DICT must be set to a pure passthrough
("{instruction}") to use it as-is rather than wrapping it again -- see the
launcher script for the one-line change.

Each source language uses a DIFFERENT few-shot/prompt config (java:
code2-exec-lang-fewshot-v2; rust: the v3 diverse-test-input prompt), so this
takes one input file PER LANGUAGE, each paired with the config that
actually mined it. num_few_shot exactly equals each pool's size (java 2/2,
rust 3/3) for every mining job run this branch, so random.sample's
selection was deterministic (always the full pool) -- safe to reproduce
here without needing to know which subset was drawn for any specific trace.

Usage:
    python train/build_full_prompts.py \\
      --input eval_data/sft_codenet_java_v1.jsonl --cfg config/my_test_mcts_qwen2.5_7b_depth_16_1gpu_sampling_rl_mining_java.yaml \\
      --input eval_data/sft_rl_train_java_v1.jsonl --cfg config/my_test_mcts_qwen2.5_7b_depth_16_1gpu_sampling_rl_mining_java.yaml \\
      --input eval_data/sft_codenet_rust_v1.jsonl --cfg config/my_test_mcts_qwen2.5_7b_depth_16_1gpu_sampling_rl_pilot_rust.yaml \\
      --input eval_data/sft_rl_train_rust_v1.jsonl --cfg config/my_test_mcts_qwen2.5_7b_depth_16_1gpu_sampling_rl_pilot_rust.yaml \\
      --model_dir models/Qwen2.5-Coder-7B-Instruct \\
      --output_file eval_data/sft_full_prompt_v1.jsonl

--input/--cfg are paired positionally: the Nth --input uses the Nth --cfg.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omegaconf import OmegaConf
from rstar_deepthink.config import BaseConfig
from rstar_deepthink.agents.utils import rstar_prompt_wrap


def load_config(cfg_path, model_dir):
    config = OmegaConf.structured(BaseConfig)
    custom_config = OmegaConf.load(cfg_path)
    config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    config.model_dir = model_dir
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", required=True, help="a score_rl_rollouts.py --emit_sft file; repeatable")
    ap.add_argument("--cfg", action="append", required=True, help="the mining config that produced the paired --input; repeatable, same order")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    if len(args.input) != len(args.cfg):
        ap.error(f"--input given {len(args.input)} times but --cfg given {len(args.cfg)} times; must pair 1:1 in order")

    n_out = 0
    with open(args.output_file, "w", encoding="utf-8") as out:
        for input_path, cfg_path in zip(args.input, args.cfg):
            config = load_config(cfg_path, args.model_dir)
            print(f"{input_path}: using {cfg_path} (few_shot_path={config.few_shot_path})")
            with open(input_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    full_prompt = rstar_prompt_wrap(r["question"], "", config)
                    out.write(json.dumps({"instruction": full_prompt, "output": r["target_text"]}) + "\n")
                    n_out += 1

    print(f"{n_out} examples -> {args.output_file}")


if __name__ == "__main__":
    main()

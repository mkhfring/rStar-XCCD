# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
import os
import json
import glob
import time
import torch
import traceback
import argparse
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from rstar_deepthink.agents import BS, MCTS
from rstar_deepthink.solver import Solver
from rstar_deepthink.config import BaseConfig
from rstar_deepthink.tools.python_tool import set_rust_crate_check, set_execute_code2, set_auto_code2
from evaluate_clone_results import evaluate as run_clone_evaluation, format_report as format_clone_eval_report

torch.set_num_threads(12)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_qaf(filename: str):
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
        if "example" in data:
            data = data["example"]
    elif filename.endswith(".jsonl"):
        data = []
        with open(filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    else:
        raise ValueError(f"Unrecognized file format: {filename}")
    return data

def batch(iterable, n=-1):
    l = len(iterable)
    if n <= 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="config/sft_eval_mcts.yaml")
    args.add_argument("--qaf", type=str, default="", help="quesuion and answer file")
    args.add_argument('--model_dir', type=str, default="") 
    args.add_argument('--reward_model_dir', type=str, default="") 
    args.add_argument('--save_in_model', type=str, default="")
    args.add_argument('--branch', type=str, default="",
                       help="Optional tag (e.g. the git branch these code changes came from) "
                            "folded into the output filename and its resume glob, so runs from "
                            "different branches/code versions never share or resume from each "
                            "other's output file, and stay easy to tell apart when comparing.")
    args.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                       help="Resume from the latest matching output file: skip questions already "
                            "present in it and append new results to it, instead of starting a "
                            "fresh output file and reprocessing everything. On by default; pass "
                            "--no-resume to always start over.")
    args = args.parse_args()
    return args


def find_latest_output_file(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def load_processed_indices(saved_jsonl_file: str) -> set:
    processed = set()
    with open(saved_jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "question" in record and "index" in record:
                processed.add(record["index"])
    return processed


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    if args.model_dir:
        config.model_dir = args.model_dir
    if args.reward_model_dir:
        config.reward_model_dir = args.reward_model_dir
    print(config)

    llm_version = os.path.basename(config.model_dir.rstrip("/"))

    # Before Solver(): its spawn-context worker pool inherits this via the
    # environment (see python_tool.set_rust_crate_check).
    set_rust_crate_check(config.rust_crate_check)
    set_execute_code2(config.execute_code2)
    set_auto_code2(config.auto_code2)

    data = load_qaf(args.qaf)
    solver = Solver(config=config)

    # init agent
    if config.mode == "mcts":
        agent = MCTS
    elif config.mode == "bs":
        agent = BS
    else:
        raise NotImplementedError
    if args.reward_model_dir:
        llm_version += "." + args.reward_model_dir.split("/")[-1]
        
    qaf_stem, qaf_ext = os.path.splitext(args.qaf)
    qaf_tag = f"{qaf_stem}_depth_{config.max_depth}{qaf_ext}"
    branch_tag = f".{args.branch}" if args.branch else ""
    # Slurm account (e.g. def-fard_gpu) folded into the name/resume glob so the same
    # job submitted under different accounts never shares or resumes the other's output.
    account = os.environ.get("SLURM_JOB_ACCOUNT", "")
    account_tag = f".{account}" if account else ""
    run_tag = f"{qaf_tag}.{config.mode}.{llm_version}{branch_tag}{account_tag}"
    saved_jsonl_file = f"{run_tag}.{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl"

    if args.save_in_model:
        saved_jsonl_file = args.save_in_model + '.jsonl'
        saved_jsonl_file_dir = os.path.dirname(saved_jsonl_file)
        os.makedirs(saved_jsonl_file_dir, exist_ok=True)

    total_data_len = len(data)
    if args.resume:
        # save_in_model already writes to a fixed, deterministic filename
        # every run; everything else is timestamped per run, so the
        # "latest output file" has to be found by globbing for prior runs
        # against this same question file/mode/model combo.
        if args.save_in_model:
            existing_file = saved_jsonl_file if os.path.exists(saved_jsonl_file) else None
        else:
            existing_file = find_latest_output_file(f"{run_tag}.*.jsonl")

        if existing_file:
            processed_indices = load_processed_indices(existing_file)
            data = [d for d in data if d["index"] not in processed_indices]
            saved_jsonl_file = existing_file
            print(f"Resuming from {existing_file}: {len(processed_indices)} already done, "
                  f"{len(data)} remaining.")

    with open(saved_jsonl_file, "a+", encoding='utf-8') as writer:
        for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
            agents = [agent(config=config, question=d["question"], ground_truth=str(d["answer"]))
                      for d in cur_data]
            try:
                jsonlines = solver.solve(agents, saved_jsonl_file, cur_data)
            except Exception as e:
                # A single pathological input (e.g. a prompt that overflows
                # max_model_len) would otherwise crash the whole batch job
                # and discard every already-processed sample. Log it and
                # write a placeholder so the run can continue past it.
                traceback.print_exc()
                print(f"Skipping {len(cur_data)} sample(s) after solve() failure: {e}")
                jsonlines = {d["question"]: {} for d in cur_data}
                for d in cur_data:
                    d["error"] = f"{type(e).__name__}: {e}"
            for d in cur_data:
                question = d["question"]
                d["rstar"] = jsonlines[question]
                writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                writer.flush()

        elapsed_minutes = round((time.time() - start_time) / 60, 2)
        writer.write(json.dumps({"index": total_data_len + 1, "time": elapsed_minutes}, ensure_ascii=False) + '\n')
        writer.flush()

    # rstar-xccd's clone-detection evaluator (precision/recall/F1/response
    # rate over question/answer/rstar records), same as pure_inference runs
    # at the end of their own inference call. Only meaningful for qaf files
    # carrying rstar-xccd's question/answer schema -- evaluate_clone_results
    # skips any line without a "question" field, so anything else just
    # yields zero counts rather than erroring.
    stats = run_clone_evaluation(saved_jsonl_file)
    report = format_clone_eval_report(stats, saved_jsonl_file)
    print(report)
    report_path = f"{saved_jsonl_file}_result"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Results written to: {report_path}")

    solver.close()

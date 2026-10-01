"""Run the CLCCD paper's (Moumoula et al., FSE 2025, arXiv 2408.04430) own
6 single-call prompts -- Simple (sp1), Improved Simple (sp2), Similarity
Score (sim), Reasoning (reas), Integrate (inte), Similar Lines (sl) -- on
this project's own Qwen models, over this project's own
test_python_{java,rust}_CLCCD.jsonl files.

Not included: "se" (Separate Explanation) and "sct" (Separate Code), the
paper's two other prompts, both multi-stage pipelines (sct needs two
description calls plus a comparison call; se aggregates 4 other prompts'
outputs into a 5th call) -- a meaningfully bigger job, left for later.

Prompt text and the system/user message shape are copied verbatim from the
CLCCD repo's prompt.py and gpt_inf.py's sp_inference() so the comparison is
to their exact prompts, not a paraphrase. codeA (Python) / codeB (other
language) are recovered from this project's own templated "question" field
via regex, since our eval_data files were themselves built by templating
codeA/codeB out of the same underlying CLCCD data (see
CLCCD_DATASET_CONSTRUCTION.md) -- so this round-trips cleanly.

temperature=0.3 matches gpt_inf.py's inference() call exactly.

Baseline revision (BASELINE_REVISION_PLAN_2026-09-28.txt):
  --split dev        run on the problem-disjoint CodeNet dev set
                     (eval_data/e16a/dev_python_{L}_codenet.jsonl) to CHOOSE the
                     best prompt per model/language without looking at test.
  --split hard       same-problem hard-negative set (eval_data/e16a/
                     hard_python_{L}_codenet.jsonl, build_hard_negatives.py);
                     use each model's dev-chosen prompt.
  --split hardeval   the LOCKED hard-negative sets (hardeval_python_{L}_codenet.jsonl,
                     split_training_pool.py); same prompt choice as for --split hard.
  --enable_thinking  Qwen3 native thinking. The answer is parsed only from the
                     text after </think>; a response whose thinking never closes
                     (hit max_tokens) counts as unanswered. Use a large
                     --max_tokens and Qwen's recommended thinking sampling
                     (--temperature 0.6 --top_p 0.95 --top_k 20).
  --seed             vLLM sampling seed (3 seeds for the final baselines).
  --prompt_format    how the query + code pair is wrapped. "chat" (default) is the
                     GPT/sp_inference() shape above, used for Qwen. The others copy
                     the CLCCD authors' own per-model scripts verbatim:
                       falcon     falcon_inf.py     >>QUESTION<<...>>ANSWER<<
                       starchat   starchat_inf.py   <|system|>/<|user|>/<|assistant|>, stop <|end|>
                       starcoder2 starcoder_inf.py  chat template + "Answer: " prefix, stop ###
                       llama2     llama2_inf.py     [INST] ... [/INST]
                     Pass those scripts' sampling settings via --temperature/--top_k/
                     --top_p/--repetition_penalty (see Qwen3/run_litprompts_1gpu.sh).
  --n_samples N      same-compute baseline (self-consistency): N samples per pair;
                     all raw outputs/predictions are stored ("raw_outputs",
                     "predictions") and "predicted" is their majority vote. Score
                     SC@k and other vote rules with
                     eval_data/e16a/score_self_consistency.py.
  Output names gain a variant tag, e.g. <prompt>.think.s1, so runs never collide.

Output: one JSONL per (model, dataset, prompt) at
eval_data/paper_prompt_replication/<dataset>.<model>.<prompt_tag>.jsonl,
one record per question: {index, question, answer, prompt_tag, raw_output,
predicted}. Also prints a confusion-matrix summary per (dataset, prompt) at
the end, in the same P/R/F1/response-rate shape used everywhere else in
this project.
"""
import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parent.parent

# Prompt text copied verbatim from CLCCD/prompt.py.
PROMPTS = {
    "sp1": "Analyze the following two code snippets and determine whether they are clones, regardless of the programming language. Respond with 'yes' if the code snippets are clones or 'no' if not.",
    "sp2": "Consider the overall structure and logic of the following two codes and determine if the two code snippets perform a similar task. Respond with 'yes' if the two codes perform similar tasks or 'no' otherwise.",
    "sim": "Assess the similarity of the following two code snippets and provide a similarity score between 0 and 10. A higher score indicates that the two codes are more similar. Output the similarity score.",
    "reas": "Provide a detailed reasoning process for detecting code clones in the following two code snippets, regardless of the programming language. Based on your analysis, respond with 'yes' if the code snippets are clones or 'no' if they are not.",
    "inte": "Analyze the following two code snippets to assess their similarity and determine if they are code clones, regardless of the programming language. Provide a similarity score between 0 and 10, where a higher score indicates more similarity. Additionally, presents a detailed reasoning process for detecting code clones. Conclude by 'yes' if they are clones or 'no' otherwise.",
    "sl": "Analyze the following two code snippets for code clone detection, regardless of the programming language. You should first report which lines of code are more similar. Then based on the report, please answer whether these two codes are a clone pair. The response should be 'yes' or 'no'",
}
SIM_ONLY = {"sim"}  # prompts whose output is a 0-10 score, not yes/no
SYSTEM_MSG = "You are an expert in code clone detection"

QUESTION_RE = re.compile(
    r"Code 1: Python\n```python\n(?P<codeA>.*?)\n```\n\nCode 2: \w[\w+#]*\n```\w*\n(?P<codeB>.*?)\n```",
    re.DOTALL,
)


def extract_codes(question):
    m = QUESTION_RE.search(question)
    if not m:
        return None, None
    return m.group("codeA"), m.group("codeB")


def build_user_message(query, codeA, codeB):
    # Verbatim format from CLCCD/gpt_inf.py sp_inference().
    return f"""{query} \n Code A: ``` {codeA} ``` \n Code B: ```{codeB}``` \n"""


YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


THINK_END = "</think>"


def strip_thinking(text, thinking):
    """With thinking on, keep only the final answer after </think>; None if the
    thinking block never closed (truncated at max_tokens)."""
    if not thinking:
        return text
    if THINK_END not in text:
        return None
    return text.split(THINK_END, 1)[1]


# Fallback for prose answers without the words yes/no (e.g. StarChat's "Both code snippets
# perform a similar task ..."). The CLCCD authors' released results carry a separate yes/no
# "output" label for such answers but not the code that produced it. Checked against those
# labels on their CodeNet+XLCoST sp1/sp2 outputs: agreement 0.80-1.00 per model/prompt
# (StarChat 0.06-0.25 -> 0.80-0.96, StarCoder2 0.84-0.96 -> 0.99, Falcon 0.90-0.94 -> 0.98-1.00).
NEG_RE = re.compile(
    r"\b(?:do|does|did)\s*n[o']t\s+(?:\w+\s+){0,2}(?:perform|have|share|do)\s+(?:a\s+|the\s+)?(?:similar|same)"
    r"|\bnot\s+(?:semantic(?:ally)?\s+|code\s+)?(?:clones?|similar|equivalent)"
    r"|\bdifferent\s+(?:tasks?|purposes?|functionalit(?:y|ies)|problems?)"
    r"|\bperform(?:s|ing)?\s+different", re.I)
POS_RE = re.compile(
    r"\bperform(?:s|ing)?\s+(?:a\s+|the\s+)?(?:very\s+)?(?:similar|same)\s+(?:tasks?|functions?|operations?)"
    r"|\b(?:similar|same)\s+(?:tasks?|functionalit(?:y|ies)|purposes?)"
    r"|\bare\s+(?:semantic\s+)?(?:code\s+)?clones\b", re.I)


def parse_output(prompt_tag, text):
    if text is None:
        return None
    if prompt_tag in SIM_ONLY:
        m = NUMBER_RE.search(text)
        if not m:
            return None
        score = float(m.group(1))
        return "clone" if score >= 5 else "non-clone"
    m = YES_NO_RE.search(text)
    if m:
        return "clone" if m.group(1).lower() == "yes" else "non-clone"
    neg, pos = NEG_RE.search(text), POS_RE.search(text)
    if neg and (not pos or neg.start() <= pos.start()):
        return "non-clone"
    if pos:
        return "clone"
    return None


def load_dataset(path):
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


def confusion(records, preds):
    tp = fp = tn = fn = nj = 0
    for r, pred in zip(records, preds):
        truth = (r.get("answer") or "").lower()
        if pred is None:
            nj += 1
        elif pred == "clone" and truth == "clone":
            tp += 1
        elif pred == "clone":
            fp += 1
        elif pred == "non-clone" and truth == "non-clone":
            tn += 1
        else:
            fn += 1
    n = len(records)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    resp = (n - nj) / n if n else 0
    return dict(n=n, tp=tp, fp=fp, tn=tn, fn=fn, nj=nj, prec=prec, rec=rec, f1=f1, resp=resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="e.g. Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--datasets", nargs="+", default=["java", "rust"], choices=["java", "rust"])
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()), choices=list(PROMPTS.keys()))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--split", choices=["test", "dev", "hard", "hardeval"], default="test")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--prompt_format", default="chat", choices=["chat", "falcon", "starchat", "starcoder2", "llama2"])
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    args = ap.parse_args()
    variant = ("think" if args.enable_thinking else "nothink") + (f".s{args.seed}" if args.seed is not None else "") \
        + (f".n{args.n_samples}" if args.n_samples > 1 else "")

    model_path = str(REPO_ROOT / "models" / args.model_dir)
    out_dir = REPO_ROOT / "eval_data" / "paper_prompt_replication"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Timestamped so a rerun (different prompt subset, different
    # max_tokens, a bugfix) never silently overwrites a prior run's output
    # -- matches every other result filename in this project.
    run_ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enforce_eager=True,
        distributed_executor_backend="ray" if args.tp > 1 else None,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(n=args.n_samples, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                     max_tokens=args.max_tokens, seed=args.seed,
                                     repetition_penalty=args.repetition_penalty,
                                     stop={"starchat": ["<|end|>"], "starcoder2": ["###"]}.get(args.prompt_format))
    print(f"variant={variant} split={args.split} sampling={sampling_params}")

    def to_model_text(query, codeA, codeB):
        pair = f"{query}\nCode A:\n{codeA}\nCode B:\n{codeB}\n"
        if args.prompt_format == "falcon":
            return f">>QUESTION<<{query}\nCode A:\n{codeA}\nCode B:\n{codeB}\n>>ANSWER<<"
        if args.prompt_format == "starchat":
            return f"<|system|>\n<|end|>\n<|user|>\n{pair}<|end|>\n<|assistant|>"
        if args.prompt_format == "starcoder2":
            return tokenizer.apply_chat_template([{"role": "user", "content": pair}], tokenize=False) + "Answer: "
        if args.prompt_format == "llama2":
            return f"[INST]\n{query}\n\nCode A:\n{codeA}\n\nCode B:\n{codeB}\n[/INST]"
        return to_chat_text(build_user_message(query, codeA, codeB))

    def to_chat_text(user_content):
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking,
        )

    for dataset in args.datasets:
        if args.split in ("dev", "hard", "hardeval"):
            src = REPO_ROOT / "eval_data" / "e16a" / f"{args.split}_python_{dataset}_codenet.jsonl"
        else:
            src = REPO_ROOT / "eval_data" / f"test_python_{dataset}_CLCCD.jsonl"
        records = load_dataset(src)
        codes = [extract_codes(r["question"]) for r in records]
        n_bad = sum(1 for a, b in codes if a is None or b is None)
        if n_bad:
            print(f"WARNING: {n_bad}/{len(records)} records in {dataset} failed code extraction", file=sys.stderr)

        for prompt_tag in args.prompts:
            query = PROMPTS[prompt_tag]
            chat_prompts = []
            keep_idx = []
            for i, (r, (codeA, codeB)) in enumerate(zip(records, codes)):
                if codeA is None or codeB is None:
                    continue
                text = to_model_text(query, codeA, codeB)
                if len(tokenizer(text).input_ids) >= args.max_model_len - args.max_tokens:
                    continue
                chat_prompts.append(text)
                keep_idx.append(i)

            print(f"[{args.model_dir}] dataset={dataset} prompt={prompt_tag}: generating {len(chat_prompts)}/{len(records)}")
            outputs = llm.generate(chat_prompts, sampling_params)
            raw_by_kept = [[c.text for c in o.outputs] for o in outputs]

            preds = [None] * len(records)
            raws = [None] * len(records)
            all_raws = [None] * len(records)
            all_preds = [None] * len(records)
            for pos, i in enumerate(keep_idx):
                samples = raw_by_kept[pos]
                all_raws[i] = samples
                all_preds[i] = [parse_output(prompt_tag, strip_thinking(t, args.enable_thinking)) for t in samples]
                raws[i] = samples[0]
                if args.n_samples == 1:
                    preds[i] = all_preds[i][0]
                else:  # majority of the answered samples; ties -> non-clone
                    votes = [v for v in all_preds[i] if v]
                    preds[i] = None if not votes else (
                        "clone" if votes.count("clone") > votes.count("non-clone") else "non-clone")

            stem = src.name if args.split in ("dev", "hard", "hardeval") else f"test_python_{dataset}_CLCCD.jsonl"
            out_path = out_dir / f"{stem}.{args.model_dir}.{prompt_tag}.{variant}.{run_ts}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for r, raw, pred, ar, ap_ in zip(records, raws, preds, all_raws, all_preds):
                    extra = {"raw_outputs": ar, "predictions": ap_} if args.n_samples > 1 else {}
                    f.write(json.dumps({**extra, 
                        "index": r["index"], "question": r["question"], "answer": r["answer"],
                        "prompt_tag": prompt_tag, "variant": variant, "split": args.split,
                        "raw_output": raw, "predicted": pred,
                    }) + "\n")

            stats = confusion(records, preds)
            print(f"  -> {out_path.name}: n={stats['n']} P={stats['prec']:.4f} R={stats['rec']:.4f} "
                  f"F1={stats['f1']:.4f} resp={stats['resp']:.4f}")


if __name__ == "__main__":
    main()

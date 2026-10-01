"""Cost summary for the 100-pair cost runs (cost100_python_{L}_CLCCD.jsonl), 2026-10-01.

Per system and language (same 100 pairs, Narval A100-SXM4-40GB, one GPU unless noted):
  wall_s        WALLCLOCK_SECONDS of the job (includes start-up)
  load_s        vLLM "Model loading took ... seconds" + "init engine ... took ... seconds"
  steady_s/pair (wall_s - load_s) / 100        -> GPU-seconds per pair = steady_s/pair x GPUs
  weights_GiB   vLLM "Model loading took X GiB" (per GPU; vLLM pre-allocates the rest of the
                memory for the KV cache, so nvidia-smi "used memory" would be misleading)
  gen_tok/pair  generated tokens per pair, counted with the model's own tokenizer:
                literature prompts: the saved raw output(s);
                SCB / extension: every node text of the search tree with the harness <output>
                blocks removed (= all tokens the model generated over all trajectories)
  steps/pair    SCB / extension only: number of generated tree nodes (one vLLM call each)

Usage (../venv-qwen3): python eval_data/e16a/cost_summary.py
"""
import glob
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

Q = "Qwen3"
E = "eval_data/e16a"
P = "eval_data/paper_prompt_replication"
OUT_RE = re.compile(r"<output>.*?<end_of_output>", re.S)
SYSTEMS = [  # name, log glob, model dir, output glob, kind, gpus
    ("SCB (python-only, tested)", "cost-scb-{L}_*.log.out", "Qwen3-4B", f"{E}/cost100_python_{{L}}_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.cost-scb.*.jsonl", "tree", 1),
    ("Extension (auto-code2)", "cost-ext-{L}_*.log.out", "Qwen3-4B", f"{E}/cost100_python_{{L}}_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.cost-ext.*.jsonl", "tree", 1),
    ("Qwen3-4B sp2 think", "cost-q3-4b-think-{L}_*.log.out", "Qwen3-4B", f"{P}/cost100_python_{{L}}_CLCCD.jsonl.Qwen3-4B.sp2.think.*.jsonl", "lit", 1),
    ("Qwen3-8B sp2 think", "cost-q3-8b-think-{L}_*.log.out", "Qwen3-8B", f"{P}/cost100_python_{{L}}_CLCCD.jsonl.Qwen3-8B.sp2.think.*.jsonl", "lit", 1),
    ("phi-4 sp2 (2 GPU)", "cost-phi4-{L}_*.log.out", "phi-4", f"{P}/cost100_python_{{L}}_CLCCD.jsonl.phi-4.sp2.nothink.*.jsonl", "lit", 2),
]


def log_numbers(path):
    txt = open(path, errors="replace").read()
    wall = int(re.search(r"WALLCLOCK_SECONDS: (\d+)", txt).group(1))
    m = re.search(r"Model loading took ([\d.]+) GiB and ([\d.]+) seconds", txt)
    init = re.search(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) seconds", txt)
    gpu = re.search(r"^(NVIDIA [^,\n]+)", txt, re.M)
    load = (float(m.group(2)) if m else 0.0) + (float(init.group(1)) if init else 0.0)
    return wall, load, float(m.group(1)) if m else float("nan"), gpu.group(1) if gpu else "?"


def tokens(path, kind, tok):
    rows = [json.loads(l) for l in open(path)]
    n_tok = n_steps = 0
    for r in rows:
        if kind == "lit":
            outs = r.get("raw_outputs") or [r.get("raw_output") or ""]
            n_tok += sum(len(tok(o or "").input_ids) for o in outs)
        else:
            t = r.get("rstar") or {}
            for k, v in t.items():
                if isinstance(v, dict) and k != "0":
                    n_steps += 1
                    n_tok += len(tok(OUT_RE.sub("", v.get("text") or "")).input_ids)
    return n_tok / len(rows), n_steps / len(rows), len(rows)


def main():
    toks = {}
    print(f"{'lang':5s} {'system':28s} {'GPUs':>4s} {'wall_s':>7s} {'load_s':>7s} {'s/pair':>7s} "
          f"{'GPU-s/pair':>10s} {'weights GiB':>11s} {'gen tok/pair':>12s} {'steps/pair':>10s}  GPU")
    for L in ("java", "rust"):
        for name, lg, model, og, kind, gpus in SYSTEMS:
            logs = sorted(glob.glob(f"{Q}/{lg.format(L=L)}"))
            outs = sorted(glob.glob(og.format(L=L)))
            if not logs or not outs:
                print(f"{L:5s} {name:28s} missing ({len(logs)} logs, {len(outs)} outputs)")
                continue
            wall, load, gib, gpu = log_numbers(logs[-1])
            if model not in toks:
                toks[model] = AutoTokenizer.from_pretrained(f"models/{model}")
            tok_pp, steps_pp, n = tokens(outs[-1], kind, toks[model])
            spp = (wall - load) / n
            print(f"{L:5s} {name:28s} {gpus:4d} {wall:7d} {load:7.0f} {spp:7.1f} {spp * gpus:10.1f} "
                  f"{gib:11.2f} {tok_pp:12.0f} {steps_pp if kind == 'tree' else float('nan'):10.1f}  {gpu}")


if __name__ == "__main__":
    main()

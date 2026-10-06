"""Phase 0.4 round-trip test: SFT examples must match what the model sees at inference.

For every node of every root->leaf path in real code-step trees (DEV hard pilot, java + rust):
  R1 string:  rstar_prompt_wrap(question, partial solution before node k) == (prompt + completion)[:start_k]
  R2 tokens:  tokens of that inference prompt are an exact prefix of the training tokens
  R3 loss:    every model-written character is covered by a loss token; no loss token starts
              in harness text (tokens running from the model's stop string into the harness keep loss)
  R4 vocab:   no added special tokens; "<code>" etc. are ordinary multi-token text
Also prints the token-length distribution (for --model_max_length).

Usage (from rstar-xccd/; imports vLLM via rstar_deepthink -> takes a few minutes on login nodes):
    ../venv-qwen3/bin/python tests/test_codestep_sft_roundtrip.py [--trees_per_lang 20]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from transformers import AutoTokenizer  # noqa: E402
from codestep_sft_data import (IGNORE_INDEX, build_examples, leaf_paths, load_config,  # noqa: E402
                               question_of, serialise_path, tokenize_example)
from rstar_deepthink.agents.utils import rstar_prompt_wrap  # noqa: E402

CS = "/lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD-codestep/rstar-xccd/eval_data/e16a/codestep_hard"
RUNS = {"java": ("codestep-v3-hard", "config/qwen3_4b_codestep_java_v3_1gpu.yaml"),
        "rust": ("codestep-v3-hard", "config/qwen3_4b_codestep_rust_v3_1gpu.yaml")}
MODEL = str(ROOT / "models/Qwen3-4B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees_per_lang", type=int, default=20)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    base = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    check("R4 tokenizer has no tokens added by us", len(tok) == len(base) and
          all(m not in tok.get_vocab() for m in ("<code>", "<end_of_step>", "<output>", "<end_of_output>", "<answer>")),
          f"(vocab {len(tok)})")
    ids = tok("<code>", add_special_tokens=False)["input_ids"]
    check("R4 '<code>' is ordinary multi-token text", len(ids) > 1, str(ids))

    lengths, n_paths, n_nodes = [], 0, 0
    r1_bad = r2_bad = r3_bad = boundary = 0
    for lang, (branch, cfgp) in RUNS.items():
        cfg = load_config(str(ROOT / cfgp), MODEL)
        f = sorted(glob.glob(f"{CS}/hard_python_{lang}_codenet_depth_16.jsonl.mcts.Qwen3-4B.{branch}.*.jsonl"))[0]
        recs = [json.loads(l) for l in open(f)]
        recs = [r for r in recs if "rstar" in r][: a.trees_per_lang]
        for r in recs:
            tree = r["rstar"]
            q = question_of(tree)
            exs = build_examples(r, cfg)
            for ex, (tag, states) in zip(exs, leaf_paths(tree)):
                n_paths += 1
                full = ex["prompt"] + ex["completion"]
                full_ids = tok(full, add_special_tokens=False)["input_ids"]
                lengths.append(len(full_ids))
                _, _, starts = serialise_path(states, cfg.step_delim)
                texts = [s.get("text") or "" for s in states]
                for k in range(len(states)):
                    if not texts[k]:
                        continue
                    n_nodes += 1
                    inf = rstar_prompt_wrap(q, "".join(texts[:k]), cfg)       # what inference feeds before node k
                    if inf != full[: len(ex["prompt"]) + starts[k]]:
                        r1_bad += 1
                        continue
                    inf_ids = tok(inf, add_special_tokens=False)["input_ids"]
                    if full_ids[: len(inf_ids)] != inf_ids:
                        r2_bad += 1
                t = tokenize_example(tok, ex)
                boundary += t["n_lost"]
                # every model-written character is covered by a loss token, and no loss token
                # starts in harness text
                enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
                pl = len(ex["prompt"])
                harness = [(pl + s0, pl + e0) for s0, e0 in ex["mask_spans"]]
                in_h = lambda x: x < pl or any(s0 <= x < e0 for s0, e0 in harness)
                covered = set()
                bad = False
                for (a0, b0), lab in zip(enc["offset_mapping"], t["labels"]):
                    if lab != IGNORE_INDEX:
                        bad |= in_h(a0)
                        covered.update(range(a0, b0))
                model_chars = {x for x in range(pl, len(full)) if not in_h(x)}
                if bad or not model_chars <= covered:
                    r3_bad += 1
    check("R1 inference prompt == training text prefix at every node", r1_bad == 0, f"({r1_bad} of {n_nodes} nodes, {n_paths} paths)")
    check("R2 inference tokens are an exact prefix of training tokens", r2_bad == 0, f"({r2_bad} of {n_nodes} nodes)")
    check("R3 loss covers every model-written char and starts only in model text", r3_bad == 0 and boundary == 0,
          f"({r3_bad} bad paths; {boundary} tokens start in harness text but run into model text)")
    lengths.sort()
    q = lambda p: lengths[min(len(lengths) - 1, int(p * len(lengths)))]
    print(f"     tokens per example: median {q(.5)}  p90 {q(.9)}  p99 {q(.99)}  max {lengths[-1]}  "
          f"(> 8192: {sum(l > 8192 for l in lengths)} of {len(lengths)})")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

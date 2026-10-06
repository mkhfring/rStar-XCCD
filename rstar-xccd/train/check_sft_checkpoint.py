"""Check a saved SFT checkpoint against the base model (Phase 0.5 smoke tests (d), (e)).

  * same tensor names (tied lm_head allowed to be absent), shapes and dtypes; vocab not resized
  * tokenizer: same vocabulary, no added tokens
  * --identical: every tensor bit-identical (learning-rate-0 run: proves load -> FSDP -> save is lossless)
  * otherwise: reports the relative change per tensor group (trained run: changed, not exploded)

Usage: python train/check_sft_checkpoint.py --base models/Qwen3-4B --ckpt <dir> [--identical]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import torch
from safetensors import safe_open


def tensors(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        h = safe_open(f, "pt")
        for k in h.keys():
            out[k] = (f, k)
    return out


def load(ref):
    f, k = ref
    return safe_open(f, "pt").get_tensor(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--identical", action="store_true")
    a = ap.parse_args()
    fails = []
    cb, cc = (json.load(open(os.path.join(d, "config.json"))) for d in (a.base, a.ckpt))
    if cb["vocab_size"] != cc["vocab_size"]:
        fails.append(f"vocab_size {cb['vocab_size']} -> {cc['vocab_size']}")
    tb, tc = tensors(a.base), tensors(a.ckpt)
    tied = cb.get("tie_word_embeddings", False)
    for k in set(tb) ^ set(tc):
        if not (tied and k == "lm_head.weight"):
            fails.append(f"tensor only in {'base' if k in tb else 'ckpt'}: {k}")
    groups = defaultdict(lambda: [0.0, 0.0])
    n_diff = 0
    for k in sorted(set(tb) & set(tc)):
        x, y = load(tb[k]), load(tc[k])
        if x.shape != y.shape or x.dtype != y.dtype:
            fails.append(f"{k}: {tuple(x.shape)} {x.dtype} -> {tuple(y.shape)} {y.dtype}")
            continue
        if a.identical:
            n_diff += not torch.equal(x, y)
        else:
            g = "embed" if "embed" in k else "lm_head" if "lm_head" in k else "norm" if "norm" in k else "attn" if "attn" in k else "mlp" if "mlp" in k else "other"
            groups[g][0] += (y.float() - x.float()).norm().item() ** 2
            groups[g][1] += x.float().norm().item() ** 2
    if a.identical and n_diff:
        fails.append(f"{n_diff} tensors differ (expected bit-identical)")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        pb, pc = os.path.join(a.base, name), os.path.join(a.ckpt, name)
        if os.path.exists(pb) and os.path.exists(pc):
            vb = json.load(open(pb)).get("added_tokens", json.load(open(pb)).get("added_tokens_decoder"))
            vc = json.load(open(pc)).get("added_tokens", json.load(open(pc)).get("added_tokens_decoder"))
            if json.dumps(vb, sort_keys=True) != json.dumps(vc, sort_keys=True):
                fails.append(f"{name}: added tokens differ")
        elif os.path.exists(pb):
            fails.append(f"{name} missing in checkpoint")
    if not a.identical:
        for g, (d2, x2) in sorted(groups.items()):
            print(f"  relative change {g:8s} {((d2 / x2) ** .5 if x2 else 0):.2e}")
    print(f"{len(set(tb) & set(tc))} common tensors checked")
    for f in fails:
        print("FAIL", f)
    print("CHECKPOINT OK" if not fails else "CHECKPOINT BAD")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

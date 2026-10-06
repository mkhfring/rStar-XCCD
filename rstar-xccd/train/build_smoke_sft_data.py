"""Smoke-test data for Phase 0.5 (pipeline check only; the smoke model is discarded, no result
is drawn from it).

  * SFT data: up to 100 correct-label paths per language from the DEV hard-pilot code-step trees
    (codestep-v3-hard), serialised with the zero-shot inference prompts.
  * Format-check set: 20 pairs (10 clone / 10 non-clone, seed 0) from the CLCCD-style DEV set
    dev_python_java_codenet.jsonl -- a different set from the SFT data.

Usage (from rstar-xccd/; imports vLLM via rstar_deepthink -> a few minutes on a login node):
    ../venv-qwen3/bin/python train/build_smoke_sft_data.py
"""
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
from codestep_sft_data import build_examples, load_config  # noqa: E402
from evaluate_clone_results import normalize_label  # noqa: E402

CS = "/lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD-codestep/rstar-xccd/eval_data/e16a/codestep_hard"
MAIN = "/lustre06/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd/eval_data/e16a"
CFG = {"java": "config/qwen3_4b_codestep_java_v3_zeroshot_1gpu.yaml",
       "rust": "config/qwen3_4b_codestep_rust_v2_zeroshot_1gpu.yaml"}
OUT = ROOT / "eval_data/train"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    exs = []
    for lang, cfgp in CFG.items():
        cfg = load_config(str(ROOT / cfgp), str(ROOT / "models/Qwen3-4B"))
        f = sorted(glob.glob(f"{CS}/hard_python_{lang}_codenet_depth_16.jsonl.mcts.Qwen3-4B.codestep-v3-hard.*.jsonl"))[0]
        good = []
        for line in open(f):
            r = json.loads(line)
            if "rstar" not in r:
                continue
            for ex in build_examples(r, cfg):
                if normalize_label(ex["meta"]["label"]) == r["answer"]:
                    good.append(ex)
        rng.shuffle(good)
        exs += good[:100]
        print(f"{lang}: {len(good)} correct-label paths, kept {min(100, len(good))}")
    rng.shuffle(exs)
    with open(OUT / "smoke_sft_200.jsonl", "w") as fo:
        for ex in exs:
            fo.write(json.dumps(ex) + "\n")
    dev = [json.loads(l) for l in open(f"{MAIN}/dev_python_java_codenet.jsonl")]
    pick = (rng.sample([r for r in dev if r["answer"] == "clone"], 10)
            + rng.sample([r for r in dev if r["answer"] != "clone"], 10))
    with open(OUT / "smoke_dev20_python_java_codenet.jsonl", "w") as fo:
        for r in sorted(pick, key=lambda r: r["index"]):
            fo.write(json.dumps(r) + "\n")
    print(f"wrote {len(exs)} SFT examples and 20 format-check pairs to {OUT}")


if __name__ == "__main__":
    main()

"""SFT for code-step trajectories (TRAINING_PLAN_CODESTEP_2026-10-05.txt, Phase 0.5 / Phase 1 3.4).

Replaces train_SFT.py for code-step. Data = jsonl from train/codestep_sft_data.py
({prompt, completion, mask_spans, meta}); tokenisation = codestep_sft_data.tokenize_example
(the round-trip-tested one: loss only on model-written text, harness observations masked).

Built-in checks (each one is a past failure; the run STOPS if any fails):
  * no tokens are added and embeddings are never resized (special-token bug, 2026-09-25)
  * max length is explicit (default 8192) and the run fails if more than --max_cut_frac of the
    examples are longer (truncation bug: 2048 cut most targets)
  * fails if any example has no loss token, or loses loss on model text at a boundary
  * saves the FINAL weights only (full state dict) + tokenizer + train_manifest.json;
    train/check_sft_checkpoint.py then compares tensor names/shapes with the base model

Launch (4 GPUs, FSDP): see Qwen3/run_codestep_sft_4gpu.sh.
"""
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import transformers
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codestep_sft_data import IGNORE_INDEX, PROVENANCE, tokenize_example  # noqa: E402

TAGS = ("<code>", "<end_of_step>", "<end_of_code>", "<output>", "<end_of_output>", "<answer>", "<end_of_answer>")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/Qwen3-4B")
    attn_impl: str = field(default="sdpa")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "jsonl from codestep_sft_data.py"})
    max_cut_frac: float = field(default=0.01, metadata={"help": "fail if a larger share is over model_max_length"})
    provenance: str = field(default="self,claude_written,claude_edited",
                            metadata={"help": "comma list of provenance tags to train on"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=8192)
    optim: str = field(default="adamw_torch")


class CodestepDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, max_cut_frac, provenance):
        keep_prov = set(provenance.split(","))
        assert keep_prov <= set(PROVENANCE), keep_prov
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r["meta"].get("provenance", "self") in keep_prov]
        self.items, cut, stats = [], 0, Counter()
        for r in rows:
            t = tokenize_example(tokenizer, r, max_len)
            if t is None:
                cut += 1
                continue
            if t["n_loss"] == 0:
                raise ValueError(f"example without loss tokens: {r['meta']}")
            if t["n_lost"]:
                raise ValueError(f"loss lost on model text at a boundary: {r['meta']}")
            self.items.append(t)
            stats["tokens"] += len(t["input_ids"])
            stats["loss_tokens"] += t["n_loss"]
            stats["prov_" + r["meta"].get("provenance", "self")] += 1
            stats["gold_" + str(r["meta"].get("gold"))] += 1
        if not rows:
            raise ValueError("no examples after the provenance filter")
        if cut / len(rows) > max_cut_frac:
            raise ValueError(f"{cut}/{len(rows)} examples exceed model_max_length={max_len} (> {max_cut_frac:.0%})")
        self.stats = dict(stats, examples=len(self.items), cut=cut, read=len(rows))
        logging.warning(f"dataset: {self.stats}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@dataclass
class Collator:
    pad_id: int

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), IGNORE_INDEX, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return dict(input_ids=ids, labels=lab, attention_mask=att)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_bf16(out_dir):
    """FSDP saves its fp32 master weights; store the checkpoint in bf16 like the base model
    (the E16a checkpoint was fp32: 2x disk and vLLM may then run in fp32)."""
    from safetensors.torch import load_file, save_file
    import glob
    total = 0
    for f in sorted(glob.glob(os.path.join(out_dir, "*.safetensors"))):
        sd = load_file(f)
        sd = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v).contiguous() for k, v in sd.items()}
        total += sum(v.numel() * v.element_size() for v in sd.values())
        save_file(sd, f, metadata={"format": "pt"})
    idx = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        d = json.load(open(idx))
        d.setdefault("metadata", {})["total_size"] = total
        json.dump(d, open(idx, "w"), indent=2)
    cfg = os.path.join(out_dir, "config.json")
    c = json.load(open(cfg))
    c["torch_dtype"] = "bfloat16"
    json.dump(c, open(cfg, "w"), indent=2)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    margs, dargs, targs = parser.parse_args_into_dataclasses()
    transformers.set_seed(targs.seed)

    tok = transformers.AutoTokenizer.from_pretrained(margs.model_name_or_path, trust_remote_code=True,
                                                     model_max_length=targs.model_max_length, padding_side="right")
    n_vocab = len(tok)
    if tok.pad_token_id is None:
        raise ValueError("tokenizer has no pad token; refusing to add one (would resize embeddings)")
    for t in TAGS:
        if t in tok.get_vocab():
            raise ValueError(f"{t} is a vocabulary token; code-step tags must be plain text")

    ds = CodestepDataset(dargs.data_path, tok, targs.model_max_length, dargs.max_cut_frac, dargs.provenance)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        margs.model_name_or_path, trust_remote_code=True, attn_implementation=margs.attn_impl,
        torch_dtype=torch.bfloat16 if targs.bf16 else torch.float32, use_cache=False)

    trainer = transformers.Trainer(model=model, args=targs, train_dataset=ds,
                                   data_collator=Collator(tok.pad_token_id), processing_class=tok)
    trainer.train()

    assert len(tok) == n_vocab, "tokenizer changed during training"
    trainer.save_model(targs.output_dir)            # full state dict (fsdp_config state_dict_type)
    if trainer.args.should_save:
        tok.save_pretrained(targs.output_dir)
        if targs.bf16:
            to_bf16(targs.output_dir)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit = "unknown"
        manifest = dict(data_path=dargs.data_path, data_sha256=sha256(dargs.data_path), dataset=ds.stats,
                        base_model=margs.model_name_or_path, git_commit=commit,
                        args={k: str(v) for k, v in targs.to_dict().items()
                              if k in ("learning_rate", "num_train_epochs", "per_device_train_batch_size",
                                       "gradient_accumulation_steps", "lr_scheduler_type", "warmup_ratio",
                                       "weight_decay", "model_max_length", "seed", "bf16", "fsdp")},
                        final_log=trainer.state.log_history[-1] if trainer.state.log_history else None)
        json.dump(manifest, open(os.path.join(targs.output_dir, "train_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

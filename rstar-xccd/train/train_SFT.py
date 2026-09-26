# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/meta-math/MetaMath
import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import argparse
import json
import random;random.seed(42)
import numpy as np

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(seed)
    torch.use_deterministic_algorithms(True)

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    # Pure passthrough: "instruction" is expected to already be the complete,
    # real generation-time prompt (system message + few-shot examples + the
    # model's actual chat template, built by train/build_full_prompts.py via
    # rstar_deepthink.agents.utils.rstar_prompt_wrap -- the exact function
    # the MCTS search itself calls), not a bare question. The original
    # rStar-Math version of this dict wrapped "instruction" in a generic
    # "<|user|>:...<|assistant|>: Let's think step by step and solve the
    # problem with code." template that has no relationship to how this
    # project's traces were actually produced; re-wrapping an already-
    # complete prompt here would double the chat markup and reintroduce
    # exactly the train/inference prompt-shape mismatch build_full_prompts.py
    # exists to avoid.
    "prompt_input": "{instruction}",
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    attn_impl: Optional[str] = field(default="eager") # flash_attention_2 \ sdpa \ eager


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=True)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        data_path = data_args.data_path
        try:
            data_path = data_path
        except:
            # data_path = data_path
            pass
        try:
            list_data_dict = jload(data_path)
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]

        list_data_dict = random.sample(list_data_dict,  len(list_data_dict))
        # print('actual length', len(list_data_dict))
        list_data_dict = list_data_dict[:data_args.data_length]
        # print('use length', len(list_data_dict))

        # logging.warning("Formatting inputs...")
        prompt_input = PROMPT_DICT["prompt_input"]
        # print(list_data_dict[0])
        if 'instruction' in list_data_dict[0]:
            pass
        else:
            def get_input(query):
                if query.find('\n') == -1:
                    return ''
                return '\n'.join(query.split('\n')[1:])
            list_data_dict = [{'instruction':data['query'], 'output':data['response']} for data in list_data_dict]
        # import ipdb; ipdb.set_trace()
        sources = [
            prompt_input.format_map(example)
            for example in list_data_dict 
        ]
        sources = []
        for example in list_data_dict:
            if example['instruction'] == '':
                sources.append('')
            else:
                sources.append(prompt_input.format_map(example))
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        # UPDATE 2026-09-25: drop over-length examples instead of letting
        # _tokenize_fn truncate them. Truncation is from the right, so it cuts
        # the TARGET first; with the old 2048 default (no launcher passed
        # --model_max_length) 116/130 trackA and 168/168 hard-codenet
        # java-rust examples had their target removed entirely (loss 0.0,
        # grad_norm 0.0) and every example was at least partly cut.
        max_len = tokenizer.model_max_length
        keep = [len(tokenizer(s + t)["input_ids"]) <= max_len for s, t in zip(sources, targets)]
        dropped = keep.count(False)
        logging.warning(f"model_max_length={max_len}: keeping {len(keep) - dropped}/{len(keep)} examples, "
                        f"dropping {dropped} longer than the limit")
        if dropped == len(keep):
            raise ValueError(f"every example exceeds model_max_length={max_len}")
        sources = [s for s, k in zip(sources, keep) if k]
        targets = [t for t, k in zip(targets, keep) if k]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    data_args.data_length = int(remaining_args[1])
    print(training_args.run_name)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    training_args.seed = seed

    # https://github.com/huggingface/transformers/issues/31787  attn_implementation=flash_attention_2 \ sdpa \ eager
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        attn_implementation=model_args.attn_impl,
        # UPDATE 2026-09-15: was `torch.bfloat16 if attn_impl=="flash_attention_2"
        # else torch.float32` -- with the default eager attn_impl (this
        # launcher never passes --attn_impl), this loaded the FULL model in
        # fp32 on every one of the 4 distributed ranks BEFORE FSDP could
        # shard it, i.e. ~4x the fp32 footprint of a single 7B model
        # simultaneously in host RAM. That is what caused job 3121560's
        # OOM kill during checkpoint-shard loading (MaxRSS hit the 128G
        # limit exactly). training_args.bf16=True already asks for a bf16
        # training run; the initial load should match that regardless of
        # attn_impl, not silently fall back to fp32.
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        use_cache = False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left" if "mistral" in model_args.model_name_or_path.lower() else "right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path.lower() and 'llama-3' not in model_args.model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    # UPDATE 2026-09-25: removed the rStar-Math block that registered the
    # step tags ('<code>', '<end_of_step>', '<end_of_code>', '<analysis>', ...
    # 17 in all) as additional special tokens and then called
    # resize_token_embeddings(len(tokenizer)) with no initialisation.
    # Qwen3-4B's embedding already has 151936 rows (151669 real tokens + unused
    # padding), so that "resize" SHRANK it to 151686 and the 17 new ids reused
    # padding rows that are all the same vector (pairwise cosine 1.0). With
    # tied embeddings the model could not tell <code> from <end_of_step> from
    # <analysis> on input or output, and small SFT runs never separated them:
    # every checkpoint trained with that block writes 0 code steps at MCTS
    # eval (base model: 100%). The base model -- and the MCTS prompt, few-shots
    # and vLLM stop strings -- treat these tags as plain text, so SFT now does
    # the same: no added tokens, no resize, identical tokenization at train
    # and inference time.
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()  # resume
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
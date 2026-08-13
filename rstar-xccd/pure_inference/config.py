# Adapted from rstar_deepthink/config.py to follow the same
# dataclass + OmegaConf pattern used for the rstar-xccd YAML configs.
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class BaseConfig:

    datafile: Optional[str] = field(
        default=None, metadata={"help": "path to the JSONL input dataset, relative to pure_inference/"}
    )
    model: str = field(
        default="qwen23b", metadata={"help": "key into local_models, or a model dir name directly"}
    )
    finetuned: bool = field(
        default=False, metadata={"help": "whether to load a LoRA fine-tuned adapter on top of the base model"}
    )
    lora_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "key into checkpoint_versions for the LoRA adapter to load"}
    )
    # vllm args
    num_gpus: int = field(
        default=1, metadata={"help": "tensor-parallel size / number of GPUs for vLLM"}
    )
    gpu_memory_utilization: float = field(
        default=0.90, metadata={"help": "fraction of GPU memory vLLM may reserve"}
    )
    max_model_len: int = field(
        default=16384, metadata={"help": "maximum model context length"}
    )
    max_tokens: int = field(
        default=3000, metadata={"help": "maximum number of tokens to generate per response"}
    )
    temperature: float = field(
        default=0.0, metadata={"help": "sampling temperature"}
    )

# ──────────────────────────────────────────────────────────────────────────
# async_chatgpt.py
# ──────────────────────────────────────────────────────────────────────────
import os, sys, json, time, pathlib, asyncio, aiohttp, tiktoken, torch
import argparse
from typing import List, Tuple, Any

from omegaconf import OmegaConf
from config import BaseConfig

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ---------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------
current_location = pathlib.Path(__file__).parent.resolve()
upper_level_path = current_location.parent

# rstar-xccd's own clone-detection evaluator (precision/recall/F1 over
# question/answer/rstar records), reused as-is instead of the old
# analyse_reasoning.py / analyse_reasoning_new.py Analyser classes.
sys.path.insert(0, str(upper_level_path))
from evaluate_clone_results import evaluate as run_clone_evaluation, format_report as format_clone_eval_report

# ---------------------------------------------------------------------
# Prompt: adapted from rstar_deepthink/few_shots/mcts_prompt.json's
# pot_format_instructions. Same task framing and the same requirement to put
# the final clone/non-clone decision in \boxed{...} (so evaluate_clone_results
# can read it), but with the <analysis>/<code>/<output>/<answer> step tags and
# the code-execution instructions stripped out -- this is a single generate()
# call, not a step-by-step agent loop, so there is no step to tag and no
# interpreter to hand code off to.
system_prompt = (
    "You are a powerful code analysis agent with broad software engineering "
    "knowledge and strong programming skills. Your task is to determine "
    "whether two code snippets written in different programming languages "
    "are semantic code clones.\n\n"
    "A semantic code clone means that two code snippets implement the same "
    "or highly similar functionality, even if they are written in different "
    "programming languages, use different syntax, use different variable "
    "names, or follow different implementation styles.\n\n"
    "!!! Remember:\n"
    "1. Focus on semantic behavior, not surface-level syntax.\n"
    "2. Consider input type, output type, main logic, edge cases, and "
    "whether both snippets produce equivalent outputs for equivalent valid "
    "inputs.\n"
    "3. Avoid relying only on variable names, formatting, comments, "
    "programming language syntax, or superficial structural similarity.\n"
    "4. The final decision must be either clone or non-clone. Put the final "
    "decision inside \\boxed{}.\n\n"
    "Use this format:\n"
    "Analysis: a brief comparison of the two snippets' behavior.\n"
    "Final decision: \\boxed{clone} or \\boxed{non-clone}"
)

def extract_boxed_final_answer(text: str) -> str:
    """Pull the content of the last \\boxed{...} out of a model response.

    Lightweight equivalent of rstar_deepthink.agents.utils.extract_math_answer
    / extract_boxed_answer, without pulling in that module's much heavier
    import chain (transformers/sympy/math_evaluation/etc.) just for brace
    matching. Falls back to the full stripped text when there's no \\boxed{},
    which evaluate_clone_results.normalize_label() can still substring-match
    against ("clone"/"non-clone"/paraphrases).
    """
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return text.strip()
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip()


# ---------------------------------------------------------------------
# Loading tiktoken ofline for phi3-small
# ---------------------------------------------------------------------
tiktoken_cache_dir = upper_level_path /"tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))



local_models = {
    "mini-f2-allpairs": "mini-v2-fine-tuned-allpairs-merged",
    "mini-f2-pj": "mini-v2-fine-tuned-python-java-merged",
    "mini-f2": "phi3-mini-v2-merged", 
    "mini-f3":"phi3-mini-v3-merged",
    "mini":"Phi-3-mini-128k-instruct",
    "qwen23b":"Qwen2.5-Coder-3B-Instruct",
    "qwen-f2": "qwen-coder-pyjava-merged",
    "qwen3-4b": "Qwen3-4B",

    # >5B parameter local models
    "phi3-small": "Phi-3-small-128k-instruct",
    "qwen25-7b": "Qwen2.5-Coder-7B-Instruct",
    "qwen3-8b": "Qwen3-8B",
    "deepseek-6.7b": "deepseek-coder-6.7b-instruct",
    "deepseek-7b-v1.5": "deepseek-coder-7b-instruct-v1.5",
}


class OfflineRequest:
    def __init__(
        self,
        model="Phi-3-medium-128k-instruct",
        stream=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        max_tokens=3000,
        temperature=0.0,
        enable_lora=False,
    ):
        self.model_path = str(upper_level_path / "models" / model)
        self.max_model_len = max_model_len
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            enforce_eager=True,
            distributed_executor_backend="ray" if tensor_parallel_size > 1 else None,
            enable_lora=enable_lora,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Set by FineTunedModelInference; left None here so vLLM just uses
        # the base model.
        self.lora_request = None
        self.stream = stream

        self.results = []

    def _build_prompt_text(self, prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        # enable_thinking=False keeps Qwen3 from emitting a <think> block
        # instead of a direct answer; harmless extra template kwarg for
        # every other model's chat template (mirrors rstar_prompt_wrap).
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def process_prompts(self, prompts, requested_samples_file, output_file):
        if not prompts:
            return
        # vLLM batches all prompts across the tensor-parallel GPUs in one
        # call instead of the previous one-request-at-a-time HF .generate()
        # loop. vLLM rejects the whole batch with a hard ValueError if even
        # one prompt's token count exceeds max_model_len (seen on Qwen3
        # runs, where a couple of eval_data rows tokenize far longer than
        # under Qwen2.5's tokenizer -- one row even exceeds Qwen3's own
        # absolute context limit). Filter those out up front, budgeting room
        # for the response too, so one oversized row doesn't take down the
        # other ~1000 legitimate ones; they're still written to the output
        # file with an empty response so they show up as "no judgment"
        # rather than silently vanishing from the totals.
        prompt_budget = self.max_model_len - self.sampling_params.max_tokens
        fittable, oversized = [], []
        for prompt_id, prompt, row in prompts:
            prompt_text = self._build_prompt_text(prompt)
            token_len = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            if token_len > prompt_budget:
                print(
                    f"[{prompt_id}] SKIPPED: prompt is {token_len} tokens, "
                    f"exceeds the {prompt_budget}-token budget "
                    f"(max_model_len={self.max_model_len} - max_tokens={self.sampling_params.max_tokens})"
                )
                oversized.append((prompt_id, prompt, row))
            else:
                fittable.append((prompt_id, prompt, row, prompt_text))

        prompt_texts = [f[3] for f in fittable]
        outputs = self.llm.generate(
            prompt_texts,
            self.sampling_params,
            lora_request=self.lora_request,
        ) if prompt_texts else []

        with open(output_file, "a", encoding="utf-8") as fout:
            for (prompt_id, prompt, row, _), output in zip(fittable, outputs):
                response = output.outputs[0].text
                print(f"[{prompt_id}] {response}\n")
                entry = self._build_output_entry(prompt_id, prompt, response, row)
                json.dump(entry, fout, ensure_ascii=False)
                fout.write("\n")
            for prompt_id, prompt, row in oversized:
                entry = self._build_output_entry(prompt_id, prompt, "", row)
                json.dump(entry, fout, ensure_ascii=False)
                fout.write("\n")

    def _build_output_entry(self, prompt_id, prompt, response, row: dict) -> dict:
        _, conversation = self._post_process_response((prompt_id, response), prompt)
        entry = {"idx": prompt_id, "text": conversation}
        # Carry the ground-truth fields through when the source row has them
        # (rstar-xccd's eval_data/*.jsonl schema), so the output file is a
        # standalone input to evaluate_clone_results.py.
        if "question" in row:
            entry["question"] = row["question"]
        if "answer" in row:
            entry["answer"] = row["answer"]
        # evaluate_clone_results.py expects a "rstar" tree of {tag: {final_answer}}
        # nodes (majority-voted across an MCTS search); pure inference only ever
        # produces one candidate, so a single node "1" is that whole tree.
        entry["rstar"] = {"1": {"final_answer": extract_boxed_final_answer(response)}}
        return entry

    @staticmethod
    def _post_process_response(result: dict, prompt: str) -> str:
        return result[0], (
            f"|user|\n{prompt}\n|assistant|\n"
            f"{result[1].strip()}"
        )


class QwenOfflineRequest(OfflineRequest):
    # vLLM applies Qwen's own chat template and EOS handling automatically,
    # so no per-model override is needed here (unlike the old HF-generate
    # path, which had to hand-roll padding/EOS logic per model).
    pass


class FineTunedModelInference(OfflineRequest):
    def __init__(
        self,
        lora_path,
        model="Phi-3-medium-128k-instruct",
        stream=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        max_tokens=3000,
        temperature=0.0,
    ):
        super().__init__(
            model=model,
            stream=stream,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_lora=True,
        )
        # vLLM loads and applies the LoRA adapter directly at generate time
        # instead of the old peft merge_and_unload() + save_pretrained()
        # round trip through a "merged_model" dir.
        self.peft_path = str(upper_level_path / "pure_inference_results" / lora_path)
        self.lora_request = LoRARequest("adapter", 1, self.peft_path)


class CodeCloneDetection:
    def __init__(
        self,
        data_file: str,
        temperature: float = 0.0,
        model: str = "Phi-3-medium-128k-instruct",
        output_file: str = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
        max_tokens: int = 3000,
    ):
        self.model = model
        self.temperature = temperature
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.data_file = data_file
        self.data = self._read_data(data_file)
        self.output_file = output_file 
        requested_indices = None
        # write a code to check if self.output_file exists and the extension is .jsonl. If the condition satisties, make a list of all all indexes
        if os.path.exists(self.output_file) and self.output_file.endswith(".jsonl"):
            output = self._read_data(self.output_file)
            # The trailing {"index": ..., "time": ...} timing record (see
            # main()) has no "idx" -- skip it rather than KeyError on resume.
            requested_indices = [d["idx"] for d in output if "idx" in d]

            
        if requested_indices is None:
            self.prompts = [self._make_prompt_from_row(d) for d in self.data]
            assert 1==1
        else:
            self.prompts = [
                self._make_prompt_from_row(d)
                for d in self.data if d["index"] not in requested_indices
            ]
        
        self.gpt = self._create_gpt()

    def _create_gpt(self):
        return OfflineRequest(
            model=self.model,
            stream=False,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


#     # ---------- data helpers -----------------------------------------
    @staticmethod
    def _read_data(data_file: str) -> List[dict]:
        with open(data_file, "r") as f:
            return [json.loads(line) for line in f]

    def _make_prompt_from_row(self, d: dict) -> Tuple[int, str, dict]:
        # rstar-xccd's eval_data/*.jsonl rows are pre-formatted into a single
        # "question" string (both snippets already embedded as fenced code
        # blocks) rather than separate code1/code2 fields; splice that
        # straight into the code section of the prompt below instead of
        # rebuilding it from code1/code2.
        if "code1" in d and "code2" in d:
            code_section = f"Code1:\n{d['code1']}\n\nCode2:\n{d['code2']}"
        else:
            code_section = d["question"]
        sample_id, prompt = self._make_prompt(d["index"], code_section)
        return sample_id, prompt, d

    @staticmethod
    def _make_prompt(sample_id: int, code_section: str) -> Tuple[int, str]:
        # All the task instructions (criteria, format, the \boxed{} decision
        # requirement) now live in the system prompt, mirroring
        # rstar_prompt_wrap's pot_format_instructions as the system turn. The
        # user turn here is just the question itself, matching pot_suffix's
        # "Question: {input}".
        return sample_id, f"Question: {code_section}"

#     # ---------- main entry -------------------------------------------
#     def run_processing(self, requested_file: str, output_file: str):
#         asyncio.run(
#             self.gpt.process_prompts_async(self.prompts, requested_file, output_file)
#         )
    def run_processing(self, requested_samples_file, output_file):
        self.output_file = output_file
        self.gpt.process_prompts(self.prompts, requested_samples_file, output_file)
        return self


class CodeCloneDetectionFineTuned(CodeCloneDetection):
    def __init__(
        self,
        lora_path,
        data_file: str,
        temperature: float = 0.0,
        model: str = "Phi-3-medium-128k-instruct",
        output_file: str = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
        max_tokens: int = 3000,
    ):
        self.lora_path = lora_path
        super().__init__(
            data_file, temperature, model, output_file,
            tensor_parallel_size, gpu_memory_utilization, max_model_len, max_tokens,
        )

    def _create_gpt(self):
        return FineTunedModelInference(
            self.lora_path,
            model=self.model,
            stream=False,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


class CodeCloneDetectionQwenCoder(CodeCloneDetection):
    def __init__(
        self,
        data_file: str,
        temperature: float = 0.0,
        model: str = "qwen2.5-Coder-3B-Instruct",
        output_file: str = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
        max_tokens: int = 3000,
    ):
        super().__init__(
            data_file, temperature, model, output_file,
            tensor_parallel_size, gpu_memory_utilization, max_model_len, max_tokens,
        )

    def _create_gpt(self):
        return QwenOfflineRequest(
            model=self.model,
            stream=False,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


# # ---------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline clone-detection inference, configured via a YAML file (see pure_inference/config/)."
    )
    parser.add_argument(
        "--custom_cfg", type=str, default=None,
        help="Path to a YAML config file (e.g. config/qwen25_3b.yaml), merged over the BaseConfig defaults"
    )
    # The following all override whatever the config file (or its defaults) set.
    parser.add_argument("-d", "--datafile", type=str, default=None, help="Path to the JSONL input dataset")
    parser.add_argument("-m", "--model", type=str, default=None, help="Model to be loaded")
    parser.add_argument("-f", "--finetuned", action="store_true", default=None, help="Use the fine-tuned/LoRA model")
    parser.add_argument("-g", "--num_gpus", type=int, default=None, help="Number of GPUs for vLLM tensor-parallel inference")
    return parser.parse_args()


def load_config(args) -> "OmegaConf":
    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))

    if args.datafile is not None:
        config.datafile = args.datafile
    if args.model is not None:
        config.model = args.model
    if args.finetuned is not None:
        config.finetuned = args.finetuned
    if args.num_gpus is not None:
        config.num_gpus = args.num_gpus
    return config


def main():
    start_time = time.time()

    checkpoint_versions = {
        "v2" : "mini-v2-fine-tuned/checkpoint-3340"
    }

    args = parse_args()
    config = load_config(args)
    print(config)

    if not config.datafile:
        raise ValueError("datafile must be set via --custom_cfg or -d/--datafile")

    # All run outputs (raw generations, evaluation metrics, missing/correct
    # index lists) live under this one directory instead of being scattered
    # across eval_data/, extended-experiments/test_files/, etc.
    results_dir = os.path.join(current_location, "offline_results")
    os.makedirs(results_dir, exist_ok=True)

    data_file_path = os.path.join(current_location, config.datafile)
    datafile_basename = os.path.basename(config.datafile)
    datafile_stem = os.path.splitext(datafile_basename)[0]
    output = os.path.join(results_dir, f"{datafile_basename}_{config.model}_inference_result.jsonl")
    ids_filename = f"{datafile_stem}_ids.txt"

    print(f"code input infor: the output path:{output} \n data_path:{data_file_path} \n ids_file:{ids_filename}")

    gpt_kwargs = dict(
        data_file=data_file_path,
        output_file=output,
        tensor_parallel_size=config.num_gpus,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    if config.finetuned:
        print("Running fine-tuned version")
        ccd = CodeCloneDetectionFineTuned(
            lora_path=checkpoint_versions[config.lora_checkpoint],
            model="Phi-3-mini-128k-instruct",
            **gpt_kwargs,
        )
    elif "qwen" in config.model:
        print("Running Qwen Coder version")
        ccd = CodeCloneDetectionQwenCoder(
            model=local_models[config.model],
            **gpt_kwargs,
        )
    else:
        print("Running original version")
        ccd = CodeCloneDetection(
            model=local_models[config.model],
            **gpt_kwargs,
        )
    

    ccd.run_processing(
        requested_samples_file=os.path.join(results_dir, ids_filename),
        output_file= output
    )

    # Same trailing footer record main.py writes: {"index": len(data)+1,
    # "time": elapsed_minutes} as the last line of the output file.
    # evaluate_clone_results.evaluate() already skips any line without a
    # "question" field, so this line is transparent to it (that's literally
    # called out in its own docstring as the expected trailing footer).
    elapsed_minutes = round((time.time() - start_time) / 60, 2)
    with open(output, "a", encoding="utf-8") as f:
        f.write(json.dumps({"index": len(ccd.data) + 1, "time": elapsed_minutes}, ensure_ascii=False) + "\n")

    # rstar-xccd's own evaluator (precision/recall/F1/response-rate over
    # question/answer/rstar records) instead of analyse_reasoning.py /
    # analyse_reasoning_new.py. Only meaningful for rows carrying rstar-xccd's
    # question/answer schema -- evaluate_clone_results skips any line without
    # a "question" field, so a code1/code2-schema run just yields zero counts.
    stats = run_clone_evaluation(output)
    report = format_clone_eval_report(stats, output)
    print(report)
    report_path = os.path.join(results_dir, f"{datafile_basename}_{config.model}_evaluate_result.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Results written to: {report_path}")


if __name__ == "__main__":
    main()

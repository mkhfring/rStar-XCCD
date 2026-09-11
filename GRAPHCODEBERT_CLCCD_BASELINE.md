# GraphCodeBERT zero-shot baseline on CLCCD

## Motivation

All CLCCD (see `CLCCD_DATASET_CONSTRUCTION.md`) results so far in this
project come from prompted generative LLMs (Qwen2.5-Coder, Qwen3, Phi-3,
DeepSeek-Coder), evaluated via `main.py`'s MCTS search or
`pure_inference/offline_inference_fine_tuned_model_extended_experiments.py`'s
single-pass generation. A literature search (2025-09-05/06) found no
published CLCCD result, or result on any comparable cross-language
clone-detection benchmark, for any encoder-only code model (GraphCodeBERT,
CodeBERT, UniXcoder) or for any of the specific LLMs used in this repo. The
closest published baselines are the CLCCD paper's own evaluation of 5
LLMs (GPT-3.5-Turbo, LLAMA2-Chat-7B, StarChat-β, StarCoder2-Instruct,
Falcon-7B-Instruct) plus an OpenAI `text-embedding-3-large` + SVM classifier,
none of which overlap with this repo's models.

To get a non-LLM reference point without training a new model from scratch,
this experiment evaluates `microsoft/graphcodebert-base` -- a well-known,
open-weight, non-generative encoder -- on the same CLCCD python-java and
python-rust test sets used everywhere else in this repo
(`eval_data/test_python_java_CLCCD.jsonl`, `eval_data/test_python_rust_CLCCD.jsonl`).

## Why this model, and why not the "ready-made" clone-detection checkpoints

A few already-fine-tuned GraphCodeBERT clone-detection checkpoints exist on
Hugging Face (e.g. `jiekeshi/GraphCodeBERT-50MB-Clone-Detection`,
`aliceinbordernone/graphcodebert-finetuned-python-clone-detection`). All of
them were rejected for this experiment:

- They ship a raw PyTorch state dict (`.pt`/`.bin`) with no `config.json`,
  meaning the classifier head's architecture isn't declared anywhere
  loadable via `transformers.AutoModel` -- using them requires either an
  external, unaudited GitHub repo's model-definition code, or reverse
  engineering a personal research notebook. `torch.load` on an untrusted
  pickle from an unfamiliar account is also a real arbitrary-code-execution
  risk, not just an inconvenience.
- They were fine-tuned on BigCloneBench (Java-Java, same-language) or a
  Python-only dataset -- not cross-language -- so they wouldn't be
  "no fine-tuning" for *this* task even if the loading concerns didn't
  apply.

`microsoft/graphcodebert-base` (the official Microsoft release: config.json
+ tokenizer + `pytorch_model.bin`, MIT-style license, 83k+ downloads) avoids
all of that: standard `transformers` loading, no external code, and it is
genuinely untouched by any clone-detection fine-tuning, cross-language or
otherwise. Downloaded to `models/graphcodebert-base/` in this repo (~476MB,
config/tokenizer/`pytorch_model.bin` only -- the TF/Flax weight files were
skipped).

## Method

Implementation: `pure_inference/graphcodebert_clone_baseline.py`.

GraphCodeBERT's release does not include a trained pooler or classification
head -- `AutoModel.from_pretrained` reports `roberta.pooler.dense.{weight,bias}`
as randomly initialized, so `pooler_output` cannot be used. Instead, each of
the two code snippets in a CLCCD question (extracted from the prompt's two
fenced code blocks via regex) is encoded independently:

1. Tokenize the snippet (truncate to 512 tokens).
2. Run it through the encoder to get `last_hidden_state`.
3. Mean-pool over real (non-padding) tokens using the attention mask.
4. L2-normalize the resulting vector.

The two snippets' normalized embeddings are compared by cosine similarity.
A similarity `>= threshold` is classified `"clone"`, else `"non-clone"`.
Output is written in this repo's standard
`{"idx", "question", "answer", "similarity", "rstar": {"1": {"final_answer": ...}}}`
schema (plus the raw `"similarity"` score, added specifically for this
experiment) so `evaluate_clone_results.py` scores it exactly like every
other model's output.

Run via `pure_inference/run_graphcodebert_clccd_slurm.sh`: a single SLURM job
(1 A100 GPU, 30 min, `def-khajezad_gpu`) covering both datasets -- much
lighter than the vLLM generation jobs elsewhere in this repo, since this is
one forward pass per snippet with no autoregressive decoding, so there's no
need for the redundant both-accounts submission pattern used for the
long-running MCTS/vLLM jobs.

## Attempt 1: fixed threshold (0.5), fully zero-shot

First run used `threshold=0.5`, chosen *a priori* with no data fit at all --
the strictest possible reading of "no fine-tuning."

| Dataset | Precision | Recall | F1 |
|---|---|---|---|
| python-java | 0.5000 | 1.0000 | 0.6667 |
| python-rust | 0.2500 | 1.0000 | 0.4000 |

These numbers are **misleading if read at face value**. The confusion
matrix for both datasets has zero true negatives and zero false negatives --
the model predicted `"clone"` for every single instance in both datasets.
Precision exactly equals each dataset's clone base rate (900/1800=50% for
java, 300/1200=25% for rust); recall is trivially 1.0. This is not
discrimination, it's a degenerate "always answer clone" classifier.

**Why**: raw, untrained sentence embeddings from encoder-only transformer
models are known to be *anisotropic* -- without contrastive fine-tuning,
mean-pooled embeddings of essentially any two inputs end up with uniformly
high cosine similarity, regardless of actual semantic relatedness. A fixed,
un-fit threshold of 0.5 sits entirely below where GraphCodeBERT's
pairwise-similarity distribution actually lives, so every pair clears it.
This is exactly why the CLCCD paper's own embedding baseline
(`text-embedding-3-large`) needed a *trained SVM classifier* on top of the
embeddings rather than a raw cosine threshold -- a raw un-fit threshold on
top of an untrained encoder is not a meaningful baseline.

## Attempt 2: dev-calibrated threshold

To get a meaningful number without fine-tuning the encoder itself, only the
scalar similarity *threshold* is calibrated -- via
`pure_inference/calibrate_graphcodebert_threshold.py`:

1. Each dataset's judged rows are split 80/20 (test/dev), **stratified by
   ground-truth label** (fixed seed 42) so both splits preserve the
   dataset's original clone/non-clone ratio -- important for python-rust,
   which is 300 clone / 900 non-clone (25%), not balanced.
2. On the dev split only, sweep every candidate threshold (midpoints
   between consecutive distinct similarity values -- the only points where
   the confusion matrix can change) and pick the one maximizing dev F1.
3. Apply that single threshold to the held-out test split once, and report
   test-split precision/recall/F1.

The GraphCodeBERT encoder is never touched by this process -- no gradients,
no backprop, no weight updates. Only one scalar (the decision threshold) is
fit, and only on a disjoint dev split from what's reported.

| Dataset | Threshold | Dev F1 | Dev split (clone/non-clone) | Test F1 (reported) | Test split |
|---|---|---|---|---|---|
| python-java | 0.9302 | 0.7339 | 360 rows (180/180) | **0.7257** | 1440 rows (720/720) |
| python-rust | 0.9039 | 0.4112 | 240 rows (60/180) | **0.3731** | 960 rows (240/720) |

Full test-split confusion matrices:

- **python-java**: TP=496, FP=151, TN=569, FN=224 -- Precision=0.7666, Recall=0.6889, F1=0.7257
- **python-rust**: TP=158, FP=449, TN=271, FN=82 -- Precision=0.2603, Recall=0.6583, F1=0.3731

Dev and test F1 are close for both datasets (java: 0.7339 vs 0.7257; rust:
0.4112 vs 0.3731), suggesting the calibrated threshold isn't badly
overfit to the small dev split.

## Comparison against the prompted LLMs (full-dataset F1, for context)

The GraphCodeBERT numbers above are scored on an 80%-of-data held-out test
split (since 20% was spent on threshold calibration), while every LLM number
below is scored on the *full* dataset (no split needed, since nothing was
fit on the data). They aren't perfectly apples-to-apples for that reason,
but the gap is large enough that it's still informative:

| Model | python-java F1 | python-rust F1 |
|---|---|---|
| Qwen3-8B | 0.9559 | 0.9088 |
| deepseek-coder-7b-instruct-v1.5 | 0.9506 | 0.7878 |
| Qwen2.5-Coder-7B-Instruct | 0.9492 | 0.8893 |
| Qwen2.5-Coder-3B-Instruct | 0.9457 | 0.6255 |
| Qwen3-4B | 0.8767 | 0.4806 |
| **GraphCodeBERT-base (calibrated threshold, test split)** | **0.7257** | **0.3731** |
| Phi-3-mini-128k-instruct | (java pending) | 0.4349 |

Every prompted LLM in this repo, even the smallest (Qwen3-4B), outperforms
the calibrated zero-shot GraphCodeBERT baseline on both language pairs.
GraphCodeBERT's python-rust score in particular is the weakest in the
entire table.

## Caveats / what "no fine-tuning" does and doesn't mean here

- The encoder's weights are exactly as released by Microsoft -- zero
  gradient updates, zero training.
- The *threshold* is fit on data (the dev split). This is a much lighter
  touch than fine-tuning a classifier head or the encoder itself, but it is
  a supervised step, and it means the reported test-split F1 is not a
  literal zero-shot number the way the LLM numbers are (those need no
  fitting step at all, since a prompt with no ground-truth labels shown to
  the model is used directly on the full dataset).
- Mean-pooling over the whole snippet is a simple, un-tuned pooling choice;
  other choices (e.g. the `<s>` token only, or attention-weighted pooling)
  were not explored.
- `max_length=512` truncates any code snippet longer than that; no snippets
  were logged as truncated in this run, but this wasn't explicitly checked.

## Files

- `pure_inference/graphcodebert_clone_baseline.py` -- embedding + similarity scoring
- `pure_inference/calibrate_graphcodebert_threshold.py` -- dev/test split + threshold sweep
- `pure_inference/run_graphcodebert_clccd_slurm.sh` -- SLURM job (both datasets)
- `models/graphcodebert-base/` -- downloaded checkpoint (config/tokenizer/pytorch_model.bin only)
- `pure_inference/offline_results/test_python_java_CLCCD.jsonl_graphcodebert-base_inference_result.jsonl` -- raw per-row output (includes `"similarity"`)
- `pure_inference/offline_results/test_python_rust_CLCCD.jsonl_graphcodebert-base_inference_result.jsonl` -- same, for python-rust

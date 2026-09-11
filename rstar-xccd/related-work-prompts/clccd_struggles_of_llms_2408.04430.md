# The Struggles of LLMs in Cross-Lingual Code Clone Detection

- Moumoula, M. B., Kaboré, A. K., Klein, J., & Bissyandé, T. F. (2025).
  "The Struggles of LLMs in Cross-Lingual Code Clone Detection."
  Proceedings of the ACM on Software Engineering, 2(FSE), Article FSE047.
  **FSE 2025.**
- arXiv preprint (Aug 2024, ahead of the FSE 2025 publication):
  https://arxiv.org/abs/2408.04430
- ACM: https://dl.acm.org/doi/10.1145/3715764
- This is the paper the CLCCD dataset (TruX-DTF/CLCCD) used elsewhere in this
  repo (`eval_data/test_python_java_CLCCD.jsonl`,
  `eval_data/test_python_rust_CLCCD.jsonl`; see `CLCCD_DATASET_CONSTRUCTION.md`)
  comes from.
- Evaluated 5 LLMs (GPT-3.5-Turbo, LLAMA2-Chat-7B, StarChat-β,
  StarCoder2-15b-Instruct, Falcon-7B-Instruct) across these 8 prompts, on
  XLCoST and CodeNet.
- Extracted from the paper's Table 1 via an automated HTML fetch/summary; the
  two multi-step prompts ("Separate Code", "Separate Explanation") render as
  a table in the source and may not be reproduced 100% verbatim below --
  check the PDF directly before quoting these two in anything citable.

## Simple Prompt (zero-shot)

> Analyze the following two code snippets and determine whether they are
> clones, regardless of the programming language. Respond with 'yes' if the
> code snippets are clones or 'no' if not.

## Improved Simple Prompt (zero-shot)

> Consider the overall structure and logic of the following two codes and
> determine if the two code snippets perform a similar task. Respond with
> 'yes' if the two codes perform similar tasks or 'no' otherwise.

## Similar Line (chain-of-thought)

> Analyze the following two code snippets for code clone detection,
> regardless of the programming language. You should first report which
> lines of code are more similar. Then based on the report, please answer
> whether these two codes are a clone pair. The response should be 'yes' or
> 'no'.

## Reasoning (chain-of-thought)

> Provide a detailed reasoning process for detecting code clones in the
> following two code snippets, regardless of the programming language. Based
> on your analysis, respond with 'yes' if the code snippets are clones or
> 'no' if they are not.

## Integrate (chain-of-thought)

> Analyze the following two code snippets to assess their similarity and
> determine if they are code clones, regardless of the programming language.
> Provide a similarity score between 0 and 10, where a higher score
> indicates more similarity. Additionally, presents a detailed reasoning
> process for detecting code clones. Conclude by 'yes' if they are clones or
> 'no' otherwise.

## Separate Code (chain-of-thought, two-step) -- approximate, verify against PDF

> Step 1: Analyze the following code snippet and explain the function of the
> snippet. Step 2: Analyze the following functions of two code snippets and
> determine if they are code clones, regardless of the programming language.
> The function of the first code snippet is {step 1 output} and the function
> of the second is {step 1 output}. Please answer 'yes' if the code
> snippets are clones, regardless of the programming language, or 'no' if
> they are not.

## Separate Explanation (chain-of-thought, two-step) -- approximate, verify against PDF

> Step 1: Similarity/Reasoning/Difference/Integrated process without the
> cloning conclusion. Step 2: Analyze the following two code snippets and
> determine if they are code clones. The Clone Similarity/Reasoning/
> Difference Integrated information of the first and the second code is:
> Please respond with 'yes' if the code snippets are clones or 'no' if they
> are not.

## Code Similarity (score-only, no clone/non-clone label)

> Assess the similarity of the following two code snippets and provide a
> similarity score between 0 and 10. A higher score indicates that the two
> codes are more similar. Output the similarity score.

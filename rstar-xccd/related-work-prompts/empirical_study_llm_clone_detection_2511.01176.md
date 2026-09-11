# An Empirical Study of LLM-Based Code Clone Detection

- arXiv: https://arxiv.org/abs/2511.01176 (Nov 2025)
- Authors: Wenqing Zhu, Norihiro Yoshida, Eunjong Choi, Yutaka Matsubara, Hiroaki Takada
- Evaluated 5 LLMs (incl. o3-mini, best F1 = 0.943 on CodeNet-derived
  datasets) across these 4 prompts, on 7 datasets sampled from CodeNet and
  BigCloneBench via Levenshtein-ratio sampling.
- P1-P3 are close paraphrases of three of the CLCCD paper's prompts
  ("Integrate", "Reasoning", "Similar Line" respectively --
  see `clccd_struggles_of_llms_2408.04430.md`); P0 is this paper's own
  addition, a minimal-output zero-shot variant.
- Extracted from the paper's Table 2 via an automated HTML fetch; verify
  against the PDF before quoting.

## P0 (zero-shot, minimal output)

> Do code 1 and code 2 solve identical problems with the same inputs and
> outputs? Answer with yes or no and no explanation.

## P1 (similarity score + clone type + reasoning)

> Please analyze the following two code snippets to assess their similarity
> and determine if they are code clones. Provide a similarity score between
> 0 and 10, where a higher score indicates more similarity. Additionally,
> identify the type of code clone they represent and present a detailed
> reasoning process for detecting code clones. The response should be 'yes'
> or 'no'.

## P2 (chain-of-thought reasoning)

> Please provide a detailed reasoning process for detecting code clones in
> the following two code snippets. Based on your analysis, respond with
> 'yes' if the code snippets are clones or 'no' if they are not.

## P3 (line-level comparison first, then answer)

> Please analyze the following two code snippets for code clone detection.
> You should first report which lines of code are more similar. Then based
> on the report, please answer whether these two codes are a clone pair.
> The response should be 'yes' or 'no'.

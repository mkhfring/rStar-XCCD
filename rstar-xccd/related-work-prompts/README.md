# Related-work prompts for cross-language code clone detection

Prompt templates collected from recent papers on (cross-language) code
clone detection with LLMs, for reference when designing prompts in this
repo. Each file below covers one paper: full citation, a short note on the
paper's scope/dataset, and the prompt(s) it uses, quoted as closely to
verbatim as an automated extraction from the paper's HTML/PDF allows (a few
individual prompts are flagged inline as approximate -- worth checking
against the source PDF before quoting in anything citable).

| File | Paper | Prompts |
|---|---|---|
| `clccd_struggles_of_llms_2408.04430.md` | Moumoula et al., "The Struggles of LLMs in Cross-Lingual Code Clone Detection," FSE 2025 (arXiv 2408.04430) | 8 prompts (Simple, Improved Simple, Similar Line, Reasoning, Integrate, Separate Code, Separate Explanation, Code Similarity) |
| `empirical_study_llm_clone_detection_2511.01176.md` | An Empirical Study of LLM-Based Code Clone Detection (arXiv 2511.01176) | 4 prompts (P0-P3; P1-P3 are close paraphrases of 3 of the CLCCD paper's prompts) |

Note: `clccd_struggles_of_llms_2408.04430.md` is the paper the CLCCD dataset
used elsewhere in this repo (`eval_data/test_python_java_CLCCD.jsonl`,
`eval_data/test_python_rust_CLCCD.jsonl`) comes from --
see `../CLCCD_DATASET_CONSTRUCTION.md`. Neither of these papers'
prompts have been run against this repo's models yet; they're collected
here purely as reference/inspiration for prompt design, not as something
already integrated into `main.py`/`pure_inference`.

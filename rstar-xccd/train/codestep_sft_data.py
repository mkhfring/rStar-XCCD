"""Code-step trajectories -> SFT examples (TRAINING_PLAN_CODESTEP_2026-10-05.txt, Phase 0.4).

One example = one root->leaf path of a saved search tree:
    prompt      = the INFERENCE prompt of the trained model (rstar_prompt_wrap with the code-step
                  instructions and num_few_shot = 0), i.e. chat template + generation prompt
    completion  = "".join(node texts root->leaf) -- exactly the text the model continues at
                  inference (collect_partial_solution)
    mask_spans  = [start, end) character ranges of the completion written by the HARNESS
                  ("\\n<output>{observation}<end_of_output>" after every code step); no loss there.
Only the model's own steps get loss (rule R1: never train the model to state outputs).

The tokenisation here (tokenize_example) is the one train_SFT.py must use: one pass over
prompt + completion with offsets; a token gets loss only if it lies entirely inside model text.

    python train/codestep_sft_data.py --trees <run.jsonl> --config <yaml> --model_dir <dir> --out <sft.jsonl>
(the selection of WHICH paths to keep is the verifier's job, Phase 1 3.2; this module only
serialises the paths it is given.)
"""
import argparse
import json
import sys
from pathlib import Path

OUTPUT, OUTPUT_END = "<output>", "<end_of_output>"
IGNORE_INDEX = -100
PROVENANCE = ("self", "claude_written", "claude_edited")


def _key(tag):
    return tuple(int(x) for x in tag.split("."))


def question_of(tree):
    info = tree["0"].get("extra_info", "")
    return info[len("question: "):] if info.startswith("question: ") else info


def leaf_paths(tree):
    """[(leaf_tag, [state root..leaf])] for every node with a final answer."""
    out = []
    for tag, s in tree.items():
        if not isinstance(s, dict) or not s.get("final_answer"):
            continue
        k = _key(tag)
        chain = [tree[".".join(map(str, k[:i]))] for i in range(1, len(k) + 1)]
        out.append((tag, chain))
    return sorted(out, key=lambda x: _key(x[0]))


def serialise_path(states, step_delim="\n"):
    """completion text, harness mask spans, and the completion offset at which each node starts.
    Raises ValueError if a code node's text is not step + delim + wrapped observation."""
    completion, spans, starts = "", [], []
    for s in states:
        text = s.get("text") or ""
        starts.append(len(completion))
        if s.get("action") == "python_interpreter":
            suffix = f"{step_delim}{OUTPUT}{s.get('observation', '')}{OUTPUT_END}"
            if not text.endswith(suffix):
                raise ValueError("code node text does not end with its wrapped observation")
            spans.append([len(completion) + len(text) - len(suffix), len(completion) + len(text)])
        completion += text
    return completion, spans, starts


def make_prompt(question, config):
    """The trained model's inference prompt (no partial solution)."""
    from rstar_deepthink.agents.utils import rstar_prompt_wrap
    return rstar_prompt_wrap(question, "", config)


def build_examples(record, config, provenance="self", leaves=None):
    """record: one line of a run's output jsonl ({"index", "answer", "rstar": tree, ...}).
    leaves: optional set of leaf tags to keep (the verifier's choice); default all."""
    assert provenance in PROVENANCE
    tree = record["rstar"]
    q = question_of(tree)
    prompt = make_prompt(q, config)
    exs = []
    for tag, states in leaf_paths(tree):
        if leaves is not None and tag not in leaves:
            continue
        completion, spans, _ = serialise_path(states, config.step_delim)
        exs.append(dict(prompt=prompt, completion=completion, mask_spans=spans,
                        meta=dict(index=record.get("index"), gold=record.get("answer"), leaf=tag,
                                  label=states[-1].get("final_answer"), n_code_steps=len(spans),
                                  provenance=provenance)))
    return exs


def tokenize_example(tokenizer, ex, max_len=None):
    """input_ids + labels for prompt + completion in ONE tokenisation pass (no special tokens
    added: the chat template already contains them).
    Loss rule: a token gets loss iff it STARTS in model-written text. A token that starts there
    and runs into the harness text (e.g. '>' of the model's '<end_of_step>' merged with the
    harness's '\n' into '>\n') keeps its loss -- otherwise the model never learns the last
    character of its stop string; at inference vLLM cuts the output at the stop string anyway.
    Tokens starting in the prompt or in harness text get no loss; if such a token runs into model
    text it is counted in n_lost (those model characters get no loss; the round-trip test
    requires 0).
    Returns dict(input_ids, labels, n_loss, n_lost) or None if longer than max_len."""
    full = ex["prompt"] + ex["completion"]
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if max_len is not None and len(ids) > max_len:
        return None
    p = len(ex["prompt"])
    masked = [(p + a, p + b) for a, b in ex["mask_spans"]]
    in_harness = lambda x: x < p or any(ms <= x < me for ms, me in masked)
    labels, n_lost = [], 0
    for tid, (a, b) in zip(ids, offs):
        if in_harness(a):
            labels.append(IGNORE_INDEX)
            n_lost += any(not in_harness(x) for x in range(a, b))
        else:
            labels.append(tid)
    return dict(input_ids=ids, labels=labels, n_loss=sum(l != IGNORE_INDEX for l in labels), n_lost=n_lost)


def load_config(path, model_dir, num_few_shot=0):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    cfg.model_dir = model_dir
    cfg.num_few_shot = num_few_shot
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", default="self", choices=PROVENANCE)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    cfg = load_config(a.config, a.model_dir)
    n = 0
    with open(a.out, "w") as f:
        for line in open(a.trees):
            r = json.loads(line)
            if "rstar" not in r:
                continue
            for ex in build_examples(r, cfg, a.provenance):
                f.write(json.dumps(ex) + "\n")
                n += 1
    print(f"wrote {n} examples to {a.out}")


if __name__ == "__main__":
    main()

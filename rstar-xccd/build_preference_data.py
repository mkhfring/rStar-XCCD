"""Preference data from mined MCTS trees + reviewed / teacher traces
(SFT_DATA_PROCESS.txt section 9). Two outputs:

--ppm_out  STEP-level pairs in train/train_RM.py's format ({"prompt", "pos",
           "neg", "pos_count", "neg_count", "step_margin"}), i.e. what rStar's
           process preference model trains on. At every tree node with >= 2
           children, each child is scored by the mean process_reward.
           leaf_reward() of the leaves below it; the best and worst children
           form a pair when their scores differ by >= --min_margin. "prompt"
           is the full policy prompt (rstar_prompt_wrap, as in
           build_full_prompts.py) plus the shared prefix of steps.

--dpo_out  TRAJECTORY-level pairs {"prompt", "chosen", "rejected", "source"}
           for DPO, three kinds:
             tree     -- best-rewarded vs worst-rewarded complete trace of the
                         same question (chosen reward > 0 > rejected reward);
             teacher  -- a teacher trace (build_teacher_traces.py) vs the
                         model's own most-visited wrong-label leaf;
             repair   -- a repaired leaf vs the same leaf before repair (the
                         only difference is a real test instead of an empty
                         code step).

Review verdicts (--review leaf_review_*.jsonl) override the automatic tier:
a leaf marked "drop" is never chosen and scores -0.5.

Usage:
    python build_preference_data.py --trees T1.jsonl [T2.jsonl ...] \
        --cfg config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_rl_mining.yaml \
        --model_dir models/Qwen3-4B [--diff_test D.jsonl ...] [--review R.jsonl ...] \
        [--teacher TEACH.jsonl ...] --ppm_out PPM.json --dpo_out DPO.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_clone_results import node_sort_key, normalize_label
from process_reward import leaf_reward
from score_rl_rollouts import build_target_text


def children_map(numeric):
    kids = defaultdict(list)
    for tag in numeric:
        if "." in tag:
            kids[tag.rsplit(".", 1)[0]].append(tag)
    return kids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", nargs="+", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--diff_test", action="append", default=[])
    ap.add_argument("--review", action="append", default=[])
    ap.add_argument("--teacher", action="append", default=[])
    ap.add_argument("--min_margin", type=float, default=0.5)
    ap.add_argument("--only_indices_from", action="append", default=[],
                    help="question file(s); only tree records whose index is in them are used "
                         "(e.g. rl_train_*.jsonl, to mine the full-CLCCD baseline trees without "
                         "touching held-out questions)")
    ap.add_argument("--ppm_out")
    ap.add_argument("--dpo_out")
    args = ap.parse_args()

    from train.build_full_prompts import load_config
    from rstar_deepthink.agents.utils import rstar_prompt_wrap
    config = load_config(args.cfg, args.model_dir)
    wrap = lambda q: rstar_prompt_wrap(q, "", config)

    diffs = {}
    for p in args.diff_test:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            diffs[r["index"]] = r
    verdicts = {}
    for p in args.review:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            verdicts[(r["source_file"], r["index"], r["leaf_tag"])] = r["verdict"]
    teacher = defaultdict(list)
    for p in args.teacher:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            teacher[r["index"]].append(r)

    allowed = None
    for p in args.only_indices_from:
        allowed = (allowed or set()) | {json.loads(l)["index"] for l in open(p, encoding="utf-8")}

    ppm, dpo = [], []
    wrong_leaf_by_q = {}
    for path in args.trees:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if "rstar" not in r or (allowed is not None and r.get("index") not in allowed):
                continue
            idx, truth, tree = r["index"], (r.get("answer") or "").lower(), r["rstar"]
            numeric = {t: n for t, n in tree.items() if node_sort_key(t) is not None}
            leaves = [t for t, n in numeric.items() if (n.get("final_answer") or "").strip()]
            if not leaves:
                continue
            reward = {t: leaf_reward(numeric, t, truth, diffs.get(idx), verdicts.get((path, idx, t)))
                      for t in leaves}
            kids = children_map(numeric)

            def subtree_score(tag):
                ls = [l for l in leaves if l == tag or l.startswith(tag + ".")]
                return (sum(reward[l] for l in ls) / len(ls), len(ls)) if ls else (None, 0)

            prompt = wrap(r["question"])
            # step-level pairs
            for parent, cs in kids.items():
                scored = [(subtree_score(c), c) for c in cs]
                scored = [(s, n, c) for (s, n), c in scored if s is not None]
                if len(scored) < 2:
                    continue
                scored.sort(key=lambda x: x[0])
                (lo, nlo, c_lo), (hi, nhi, c_hi) = scored[0], scored[-1]
                if hi - lo < args.min_margin:
                    continue
                prefix = build_target_text(tree, parent, numeric) if parent != "0" else ""
                ppm.append({"prompt": prompt + prefix, "pos": "\n" + (numeric[c_hi].get("text") or ""),
                            "neg": "\n" + (numeric[c_lo].get("text") or ""), "pos_count": nhi,
                            "neg_count": nlo, "step_margin": hi - lo, "index": idx, "source_file": path})
            # trajectory-level pairs
            best = max(leaves, key=lambda t: (reward[t], -node_sort_key(t)[0] if node_sort_key(t) else 0))
            worst = min(leaves, key=lambda t: reward[t])
            if reward[best] > 0 > reward[worst]:
                dpo.append({"prompt": prompt, "chosen": build_target_text(tree, best, numeric),
                            "rejected": build_target_text(tree, worst, numeric), "source": "tree",
                            "index": idx, "chosen_reward": reward[best], "rejected_reward": reward[worst]})
            wrong = [t for t in leaves if reward[t] == -1.0]
            if wrong:
                top = max(wrong, key=lambda t: int(numeric[t].get("visit_count") or 0))
                wrong_leaf_by_q.setdefault(idx, (prompt, build_target_text(tree, top, numeric)))

    for idx, recs in teacher.items():
        for t in recs:
            if t.get("origin") == "repair":
                src = t["source_file"]
                tree = next(json.loads(l)["rstar"] for l in open(src) if f'"index": {idx},' in l and '"rstar"' in l)
                numeric = {k: v for k, v in tree.items() if node_sort_key(k) is not None}
                dpo.append({"prompt": wrap(t["question"]), "chosen": t["target_text"],
                            "rejected": build_target_text(tree, t["leaf_tag"], numeric), "source": "repair",
                            "index": idx})
            elif idx in wrong_leaf_by_q:
                prompt, rejected = wrong_leaf_by_q[idx]
                dpo.append({"prompt": prompt, "chosen": t["target_text"], "rejected": rejected,
                            "source": "teacher", "index": idx})

    if args.ppm_out:
        json.dump(ppm, open(args.ppm_out, "w"), indent=1)
        print(f"{len(ppm)} step-level pairs -> {args.ppm_out}")
    if args.dpo_out:
        with open(args.dpo_out, "w", encoding="utf-8") as w:
            for d in dpo:
                w.write(json.dumps(d) + "\n")
        by = defaultdict(int)
        for d in dpo:
            by[d["source"]] += 1
        print(f"{len(dpo)} trajectory pairs {dict(by)} -> {args.dpo_out}")


if __name__ == "__main__":
    main()

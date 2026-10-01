"""One table of every literature-prompt baseline vs SCB (Qwen3-4B, python-only, tested rule).

For each (model, variant) found in eval_data/paper_prompt_replication/:
  1. choose the prompt on the dev set (dev_python_{L}_codenet.jsonl.*) by F1 over ALL pairs
     (an unanswered pair counts as an error, so a prompt cannot win by abstaining);
  2. report that prompt on the CLCCD test set: P / R / F1 over answered pairs, response rate,
     F1 over all pairs;
  3. paired bootstrap (10,000 resamples) of SCB minus baseline, F1 over all pairs.
Variant = nothink | think (+ .nK for self-consistency files, which are skipped here; see
score_self_consistency.py). Test files from 2026-09-12 (no variant tag) count as nothink.
A (model, variant) with no dev run is chosen on test and flagged "TEST-CHOSEN".

Usage: python eval_data/e16a/summarize_baselines.py [--lang java rust] [--boot 10000]
"""
import argparse
import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rescore_runs import predict_tested  # noqa: E402

# Every saved raw output is RE-PARSED with the current parser of the runner (yes/no, then the
# prose fallback validated against the CLCCD authors' labels), so all models share one rule.
_src = open("pure_inference/run_clccd_paper_prompts.py").read().replace("from vllm import LLM, SamplingParams", "")
_ns = {"__file__": "pure_inference/run_clccd_paper_prompts.py", "__name__": "runner"}
exec(compile(_src, "run_clccd_paper_prompts.py", "exec"), _ns)
parse_output, strip_thinking = _ns["parse_output"], _ns["strip_thinking"]


def load_preds(path, prompt, variant):
    out = {}
    for r in map(json.loads, open(path)):
        raw = r.get("raw_output")
        out[int(r["index"])] = (r["answer"].lower(),
                                None if raw is None else parse_output(prompt, strip_thinking(raw, variant == "think")))
    return out

D = "eval_data/paper_prompt_replication"
NAME = re.compile(r"^(?P<stem>(?:dev|test)_python_(?P<L>java|rust)_(?:codenet|CLCCD))\.jsonl\.(?P<model>.+?)\."
                  r"(?P<prompt>sp1|sp2|sim|reas|inte|sl)\.(?:(?P<variant>think|nothink)(?:\.s\d+)?(?P<n>\.n\d+)?\.)?\d{14}\.jsonl$")


def stats(gold, pred):
    tp = sum(g == "clone" and p == "clone" for g, p in zip(gold, pred))
    fp = sum(g != "clone" and p == "clone" for g, p in zip(gold, pred))
    fn_ans = sum(g == "clone" and p == "non-clone" for g, p in zip(gold, pred))
    fn_all = sum(g == "clone" and p != "clone" for g, p in zip(gold, pred))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn_ans) if tp + fn_ans else 0.0
    f1_ans = 2 * P * R / (P + R) if P + R else 0.0
    f1_all = 2 * tp / (2 * tp + fp + fn_all) if tp else 0.0
    resp = sum(p is not None for p in pred) / len(pred)
    return dict(P=P, R=R, f1=f1_ans, f1_all=f1_all, resp=resp)


def f1_all(gold, pred):
    tp = sum(g == "clone" and p == "clone" for g, p in zip(gold, pred))
    fp = sum(g != "clone" and p == "clone" for g, p in zip(gold, pred))
    fn = sum(g == "clone" and p != "clone" for g, p in zip(gold, pred))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", nargs="+", default=["java", "rust"])
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    runs = defaultdict(lambda: defaultdict(dict))  # (L, model, variant) -> split -> prompt -> latest file
    for f in sorted(glob.glob(f"{D}/*.jsonl")):
        m = NAME.match(Path(f).name)
        if not m or m["n"]:
            continue
        split = "dev" if m["stem"].startswith("dev") else "test"
        runs[(m["L"], m["model"], m["variant"] or "nothink")][split][m["prompt"]] = f  # sorted -> latest wins

    for L in args.lang:
        scb_file = glob.glob(f"eval_data/test_python_{L}_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.ablation-pyonly.*.jsonl")[0]
        scb = {int(r["index"]): (r["answer"], predict_tested(r["rstar"]))
               for r in map(json.loads, open(scb_file)) if "rstar" in r}
        keys = sorted(scb)
        gold = [scb[i][0] for i in keys]
        a = [scb[i][1] for i in keys]
        s = stats(gold, a)
        print(f"\n=== {L}  SCB: P {s['P']:.4f} R {s['R']:.4f} F1 {s['f1']:.4f} resp {s['resp']:.2%}")
        print(f"{'model':34s} {'var':8s} {'prompt':6s} {'devF1':>6s} | {'P':>6s} {'R':>6s} {'F1':>6s} {'resp':>7s} "
              f"{'F1all':>6s} | SCB-base [95% CI]")
        for (LL, model, var), sp in sorted(runs.items()):
            if LL != L or "test" not in sp:
                continue
            flag = ""
            if "dev" in sp:
                devf = {}
                for p, f in sp["dev"].items():
                    rs = list(load_preds(f, p, var).values())
                    devf[p] = f1_all([g for g, _ in rs], [q for _, q in rs])
                prompt = max(devf, key=devf.get)
                devtxt = f"{devf[prompt]:.3f}"
            else:
                flag, devtxt = " TEST-CHOSEN", "  -  "
                tf = {}
                for p, f in sp["test"].items():
                    rs = list(load_preds(f, p, var).values())
                    tf[p] = f1_all([g for g, _ in rs], [q for _, q in rs])
                prompt = max(tf, key=tf.get)
            if prompt not in sp["test"]:
                print(f"{model:34s} {var:8s} {prompt:6s} {devtxt:>6s} | (no test run for the dev-chosen prompt)")
                continue
            bp = load_preds(sp["test"][prompt], prompt, var)
            b = [bp[i][1] if i in bp else None for i in keys]
            t = stats(gold, b)
            random.seed(0)
            ds = []
            for _ in range(args.boot):
                ix = [random.randrange(len(keys)) for _ in keys]
                G = [gold[i] for i in ix]
                ds.append(f1_all(G, [a[i] for i in ix]) - f1_all(G, [b[i] for i in ix]))
            ds.sort()
            lo, hi = ds[int(.025 * args.boot)], ds[int(.975 * args.boot) - 1]
            print(f"{model:34s} {var:8s} {prompt:6s} {devtxt:>6s} | {t['P']:.4f} {t['R']:.4f} {t['f1']:.4f} "
                  f"{t['resp']:7.2%} {t['f1_all']:.4f} | {f1_all(gold, a) - t['f1_all']:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag}")


if __name__ == "__main__":
    main()

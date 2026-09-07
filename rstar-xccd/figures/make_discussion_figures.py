"""Build the three discussion figures for the CLCCD MCTS paper.

Each figure carries one finding from the rollout audit that is far more
legible as a picture than as a table:

  fig1  what the old scorer actually paid for -- the reward mass by
        observation class, showing that a content-free class (the model
        printing its own verdict) is the majority of every run.

  fig2  why self-consistency signals mislead -- agreement decomposed into
        its four transitions, showing a 0.00-accuracy bucket and a
        base-rate bucket averaged into one number.

  fig3  the ceiling -- pure inference vs MCTS vs an oracle over the answers
        already in the tree, per model x dataset.

Usage:  python figures/make_discussion_figures.py
Writes PDF (for the paper) and PNG (for preview) into figures/.

Palette is the dataviz reference instance, validated for light mode:
categorical slots 1-6 pass the adjacent-pair gates; slots 1-3 pass
all-pairs (used by the dot plot in fig3). Three slots sit under 3:1
contrast on the light surface, so every mark carries a visible direct
label -- that is the required relief, not decoration.
"""
import collections
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_clone_results import MAJORITY, normalize_label, node_sort_key, predict_label

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Pure-inference results live in the *other* checkout: those SLURM scripts
# cd to rStar-XCCD/rstar-xccd while the MCTS scripts cd to
# rstar-xccd-remote/rStar-XCCD/rstar-xccd. Datasets are byte-identical
# across the two (verified by md5), so the comparison is sound.
PURE = "/lustre06/project/6104180/khajezad/rStar-XCCD/rstar-xccd/pure_inference/offline_results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#87867f"
GRID = "#e2e1dc"
NEUTRAL = "#b9b8b2"

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

RUNS = {
    "Qwen3-4B\npython↔java": "eval_data/test_python_java_CLCCD_depth_16.jsonl.mcts."
                             "Qwen3-4B.assert-consistency-score.20260904205237.jsonl",
    "Qwen3-4B\npython↔rust": "eval_data/test_python_rust_CLCCD_depth_16.jsonl.mcts."
                             "Qwen3-4B.assert-consistency-score.20260904210649.jsonl",
    "Qwen2.5-3B\npython↔java": "eval_data/test_python_java_CLCCD_depth_16.jsonl.mcts."
                               "Qwen2.5-Coder-3B-Instruct.assert-consistency-score.20260905233058.jsonl",
    "Qwen2.5-3B\npython↔rust": "eval_data/test_python_rust_CLCCD_depth_16.jsonl.mcts."
                               "Qwen2.5-Coder-3B-Instruct.assert-consistency-score.20260905233058.jsonl",
}
PURE_FILES = {
    "Qwen3-4B\npython↔java": "test_python_java_CLCCD.jsonl_qwen3-4b_inference_result.jsonl",
    "Qwen3-4B\npython↔rust": "test_python_rust_CLCCD.jsonl_qwen3-4b_inference_result.jsonl",
    "Qwen2.5-3B\npython↔java": "test_python_java_CLCCD.jsonl_qwen23b-khajezad_inference_result.jsonl",
    "Qwen2.5-3B\npython↔rust": "test_python_rust_CLCCD.jsonl_qwen23b-khajezad_inference_result.jsonl",
}

VERDICT_ECHO_MAX_LEN = 64


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    })


def instances(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "question" not in record:
                continue
            yield (record.get("answer") or "").strip().lower(), record.get("rstar", {})


def answer_leaves(tree):
    numbered = [(node_sort_key(t), t, n) for t, n in tree.items()
                if isinstance(n, dict) and node_sort_key(t) is not None]
    for _k, tag, node in sorted(numbered):
        label = normalize_label((node.get("final_answer") or "").strip())
        if label is not None:
            yield tag, node, label


def f1(pairs):
    tp = fp = fn = 0
    for truth, pred in pairs:
        if pred == "clone" and truth == "clone":
            tp += 1
        elif pred == "clone":
            fp += 1
        elif truth == "clone":
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


# --------------------------------------------------------------------------
# fig 1 -- what the scorer paid for
# --------------------------------------------------------------------------
CLASSES = [
    "model prints its verdict",
    "no runnable code produced",
    "AssertionError",
    "real computed output",
    "genuine crash",
    "other language blocked",
]


def classify(observation):
    text = observation.strip()
    if text.startswith("AssertionError"):
        return "AssertionError"
    if text.startswith("No valid Python code found"):
        return "no runnable code produced"
    if text.startswith("Java execution is not supported"):
        return "other language blocked"
    if text and len(text) < VERDICT_ECHO_MAX_LEN and normalize_label(text) is not None:
        return "model prints its verdict"
    if "error" in text.lower():
        return "genuine crash"
    return "real computed output"


def fig1():
    shares, totals = {}, {}
    for name, path in RUNS.items():
        counts = collections.Counter()
        for _truth, tree in instances(os.path.join(ROOT, path)):
            for node in tree.values():
                if isinstance(node, dict) and (node.get("observation") or ""):
                    counts[classify(node["observation"])] += 1
        total = sum(counts.values())
        totals[name] = total
        shares[name] = {c: counts[c] / total for c in CLASSES}

    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    names = list(RUNS)
    ypos = range(len(names))
    left = [0.0] * len(names)
    for ci, cls in enumerate(CLASSES):
        widths = [shares[n][cls] for n in names]
        ax.barh(list(ypos), widths, left=left, height=0.62, color=CAT[ci],
                edgecolor=SURFACE, linewidth=1.6, zorder=3)
        for yi, (w, l) in enumerate(zip(widths, left)):
            if w >= 0.045:
                ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center",
                        fontsize=8, color="#ffffff" if ci in (0, 5) else INK,
                        fontweight="bold", zorder=4)
        left = [l + w for l, w in zip(left, widths)]

    ax.set_yticks(list(ypos))
    ax.set_yticklabels([f"{n}\n{totals[n]:,} executions" for n in names], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of code-execution steps in the search", fontsize=8.5)
    ax.spines["left"].set_visible(False)
    ax.set_title("What the search was rewarded for",
                 fontsize=11.5, fontweight="bold", loc="left", pad=26, color=INK)
    ax.text(0, 1.05, "Under the previous scoring the first two classes — no computation of any kind — "
                      "earned the same positive reward as a real verification.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    ax.legend(handles=[Patch(facecolor=CAT[i], label=c) for i, c in enumerate(CLASSES)],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False,
              fontsize=8.5, handlelength=1.1, handleheight=1.1, columnspacing=1.4)
    fig.tight_layout()
    save(fig, "fig1_reward_allocation")


# --------------------------------------------------------------------------
# fig 2 -- agreement decomposed into transitions
# --------------------------------------------------------------------------
import re
ASSERT_RE = re.compile(r"\bassert\b")
TRANSITIONS = ["clone → clone", "non-clone → clone",
               "non-clone → non-clone", "clone → non-clone"]


def pre_code_verdict(tree, tag):
    parts = tag.split(".")
    verdict = None
    for prefix in [".".join(parts[:i]) for i in range(1, len(parts) + 1)]:
        node = tree.get(prefix)
        if not isinstance(node, dict):
            continue
        obs = (node.get("observation") or "").strip()
        if obs and len(obs) < VERDICT_ECHO_MAX_LEN and normalize_label(obs):
            verdict = normalize_label(obs)
        elif ASSERT_RE.search(node.get("action_input") or ""):
            verdict = "clone"
    return verdict


def fig2():
    data = {}
    for name, path in RUNS.items():
        counts = collections.Counter()
        for truth, tree in instances(os.path.join(ROOT, path)):
            for tag, _node, label in answer_leaves(tree):
                v = pre_code_verdict(tree, tag)
                if v is None:
                    continue
                counts[(f"{v} → {label}", label == truth)] += 1
        data[name] = counts

    fig, axes = plt.subplots(1, 4, figsize=(11.4, 2.9), sharex=True)
    for ax, (name, counts) in zip(axes, data.items()):
        overall_ok = sum(v for (k, ok), v in counts.items() if ok)
        overall_n = sum(counts.values())
        ax.axvline(overall_ok / overall_n, color=NEUTRAL, lw=1.2, ls=(0, (3, 3)), zorder=2)
        for yi, tr in enumerate(TRANSITIONS):
            ok, wrong = counts[(tr, True)], counts[(tr, False)]
            n = ok + wrong
            if n == 0:
                ax.text(0.02, yi, "not observed", fontsize=7.5, color=INK_MUTED, va="center")
                continue
            acc = ok / n
            highlight = tr == "clone → non-clone"
            colour = CAT[1] if highlight else NEUTRAL
            ax.barh(yi, acc, height=0.6, zorder=3, color=colour)
            # A 0.00 bar has no width, and on this figure that bar *is* the
            # finding. Anchor every highlighted row with a dot at its value
            # so a true zero stays visible without being drawn as non-zero.
            if highlight:
                ax.scatter(acc, yi, s=46, color=colour, zorder=5,
                           edgecolor=SURFACE, linewidth=1.4, clip_on=False)
            ax.text(acc + 0.05, yi, f"{acc:.2f}  (n={n})", va="center",
                    fontsize=7.5, color=INK if highlight else INK_2,
                    fontweight="bold" if highlight else "normal")
        ax.set_yticks(range(len(TRANSITIONS)))
        ax.set_yticklabels(TRANSITIONS if ax is axes[0] else [], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.42)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(name.replace("\n", "  "), fontsize=8.5, color=INK, pad=6)
        ax.spines["left"].set_visible(False)

    fig.suptitle("Agreement between a branch's early leaning and its conclusion, decomposed",
                 fontsize=11.5, fontweight="bold", x=0.006, ha="left", y=0.985, color=INK)
    fig.text(0.006, 0.925,
             "Rows are pre-code verdict → conclusion. Orange: the “give-up flip”, right 1 time in 155. The largest bucket "
             "(non-clone → non-clone) merely restates the class prior.",
             fontsize=8.5, color=INK_2, ha="left", va="top")
    fig.supxlabel("accuracy of the branch's conclusion   ·   dashed line = the run's overall leaf accuracy",
                  fontsize=8.5, color=INK_2, x=0.006, ha="left", y=0.03)
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    save(fig, "fig2_transition_decomposition")


# --------------------------------------------------------------------------
# fig 3 -- the ceiling
# --------------------------------------------------------------------------
def fig3():
    rows = []
    for name, path in RUNS.items():
        cur, oracle = [], []
        for truth, tree in instances(os.path.join(ROOT, path)):
            labels = [lab for _t, _n, lab in answer_leaves(tree)]
            if not labels:
                continue
            # Pinned to leaf-vote-only scoring: these figures were published
            # against it, and predict_label() defaults the exec-signature
            # override on since 2026-09-07.
            majority = predict_label(tree, aggregation=MAJORITY, exec_signature=frozenset())
            cur.append((truth, majority))
            oracle.append((truth, truth if truth in labels else majority))
        pure = []
        with open(os.path.join(PURE, PURE_FILES[name]), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "question" not in rec:
                    continue
                pure.append(((rec.get("answer") or "").strip().lower(),
                             predict_label(rec.get("rstar", {}),
                                           aggregation=MAJORITY,
                                           exec_signature=frozenset())))
        rows.append((name, f1(pure), f1(cur), f1(oracle)))

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    labels = ["pure inference (1 call)", "MCTS (current)", "oracle over MCTS leaves"]
    for yi, (name, a, b, c) in enumerate(rows):
        lo, hi = min(a, b, c), max(a, b, c)
        ax.plot([lo, hi], [yi, yi], color=GRID, lw=2.4, zorder=2, solid_capstyle="round")
        for val, col in zip((a, b, c), CAT[:3]):
            ax.scatter(val, yi, s=110, color=col, zorder=4,
                       edgecolor=SURFACE, linewidth=1.8)
        # Label the range endpoints only. Labelling all three collides
        # wherever the methods are close -- which is most rows, and is
        # itself the point the figure is making.
        ax.text(lo - 0.012, yi, f"{lo:.3f}", ha="right", va="center",
                fontsize=7.5, color=INK_2)
        ax.text(hi + 0.012, yi, f"{hi:.3f}", ha="left", va="center",
                fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, *_ in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0.40, 1.03)
    ax.set_xlabel("F1", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("Search gains, against a single model call and against the tree's own ceiling",
                 fontsize=11.5, fontweight="bold", loc="left", pad=34, color=INK)
    ax.text(0, 1.045,
            "The oracle picks the best answer already present in the tree, so no reward function can beat it. "
            "On Qwen2.5-3B/java a single call beats both.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", markersize=8,
                                  markerfacecolor=CAT[i], markeredgecolor=SURFACE, label=l)
                       for i, l in enumerate(labels)],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig3_ceiling")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


if __name__ == "__main__":
    style()
    fig1()
    fig2()
    fig3()

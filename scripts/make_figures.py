"""Figures for the slides. Every number is read from the reports, never typed in.

fig1  micro and macro F1 for the three approaches
fig2  per-type F1, grouped, so the rare types are visible
fig3  the error breakdown: correct / mis-cut / mis-typed / missed / spurious
fig4  the confusion matrix of the best approach as a heatmap
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PROC = ROOT / "data" / "processed"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
YELLOW = "#eda100"

SERIES = {"gazetteer": BLUE, "crf": ORANGE, "spacy": AQUA}
NICE = {"gazetteer": "Gazetteer", "crf": "CRF", "spacy": "spaCy"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK,
})


def strip(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
        spine.set_color(GRID)
    ax.tick_params(length=0)


def load() -> dict:
    path = PROC / "comparison.json"
    if not path.exists():
        raise SystemExit("run scripts/evaluate.py first")
    return json.loads(path.read_text())


def fig_headline(data: dict) -> None:
    names = list(data)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(names))
    width = 0.36
    micro = [data[n]["micro"]["f1"] for n in names]
    macro = [data[n]["macro"]["f1"] for n in names]
    ax.bar(x - width / 2, micro, width, color=BLUE, label="micro F1", zorder=3)
    ax.bar(x + width / 2, macro, width, color=ORANGE, label="macro F1", zorder=3)
    for xi, (a, b) in enumerate(zip(micro, macro)):
        ax.text(xi - width / 2, a + 0.008, f"{a:.3f}", ha="center", fontsize=9,
                color=INK2)
        ax.text(xi + width / 2, b + 0.008, f"{b:.3f}", ha="center", fontsize=9,
                color=INK2)
    ax.set_xticks(x, [NICE[n] for n in names])
    ax.set_ylabel("entity-level F1, exact span match")
    ax.set_ylim(0, max(micro + macro) * 1.25)
    ax.set_title("Training on the corpus beats both shortcuts",
                 fontsize=12, loc="left", pad=30)
    ax.text(0, 1.015, "micro weights every entity equally; macro weights every "
                      "type equally, so the gap shows who ignores rare types",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_headline_f1.png", dpi=200)
    plt.close(fig)


def fig_per_type(data: dict) -> None:
    names = list(data)
    types = sorted(data[names[0]]["support"],
                   key=lambda t: -data[names[0]]["support"][t])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(types))
    width = 0.26
    for i, name in enumerate(names):
        vals = [data[name]["per_type"][t]["f1"] for t in types]
        ax.bar(x + (i - 1) * width, vals, width, color=SERIES[name],
               label=NICE[name], zorder=3)
    labels = [f"{t}\n{data[names[0]]['support'][t]:,}" for t in types]
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("entity-level F1")
    ax.set_title("The CRF wins every type; the other two swap places\ndepending on which type you ask about",
                 fontsize=12, loc="left", pad=30)
    ax.text(0, 1.015, "test entity count under each type; spaCy scores zero on "
                      "building because no OntoNotes label maps to it",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip(ax)
    ax.legend(frameon=False, labelcolor=INK2, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_per_type_f1.png", dpi=200)
    plt.close(fig)


def fig_errors(data: dict) -> None:
    names = list(data)
    order = ["correct", "boundary error", "type error", "type + boundary",
             "missed"]
    colors = [AQUA, YELLOW, ORANGE, RED, "#c3c2b7"]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    y = np.arange(len(names))
    left = np.zeros(len(names))
    totals = [sum(data[n]["errors"].get(c, 0) for c in order) for n in names]
    for category, color in zip(order, colors):
        vals = np.array([data[n]["errors"].get(category, 0) / t
                         for n, t in zip(names, totals)])
        ax.barh(y, vals, 0.55, left=left, color=color, label=category,
                zorder=3, edgecolor=SURFACE, linewidth=2)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 0.045:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color="#ffffff" if category != "missed"
                        else INK2)
        left += vals
    ax.set_yticks(y, [NICE[n] for n in names])
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the 96,842 gold test entities")
    ax.set_title("Most mistakes are entities the model DID find,\nbut cut or labelled wrongly",
                 fontsize=12, loc="left", pad=30)
    ax.text(0, 1.015, "spurious predictions are not shown - they are extra, not "
                      "a share of the gold entities",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    strip(ax, keep=("bottom",))
    ax.legend(frameon=False, labelcolor=INK2, ncol=5, fontsize=9,
              loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_error_breakdown.png", dpi=200)
    plt.close(fig)


def fig_confusion(data: dict) -> None:
    best = max(data, key=lambda n: data[n]["micro"]["f1"])
    path = PROC / f"confusion_{best}.csv"
    if not path.exists():
        return
    cm = pd.read_csv(path, index_col=0)
    # row-normalise: each row is "of the gold entities of this type, where did
    # they go", which is the question a reader actually has
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(norm.columns)), norm.columns, rotation=45,
                  ha="right", fontsize=9)
    ax.set_yticks(range(len(norm.index)), norm.index, fontsize=9)
    ax.tick_params(colors=INK2)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            value = norm.values[i, j]
            if value >= 0.01:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center",
                        fontsize=8,
                        color="#ffffff" if value > 0.5 else INK2)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true type")
    ax.set_title(f"Where each true type ends up - {NICE[best]}",
                 fontsize=12, loc="left", pad=30)
    ax.text(0, 1.015, "each row sums to 100%; MISSED = nothing predicted there",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).outline.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_confusion.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    data = load()
    fig_headline(data)
    fig_per_type(data)
    fig_errors(data)
    fig_confusion(data)
    for path in sorted(FIG.glob("*.png")):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

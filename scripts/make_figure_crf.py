"""One extra figure for the model-training part of the talk.

fig5  what the CRF actually sees when it labels a single token: the +-2 context
      window, the features extracted at that position, and the BIO output.

This is a diagram, not a chart - there is no data to plot. It exists because
"the CRF uses context features" is a sentence nobody pictures, and one slide
showing the window over a real sentence replaces a paragraph of explanation.

The feature values are computed by the real token_features() function, not
typed in, so the slide cannot drift away from the code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

import approach_crf as crf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
PALE = "#cde2fb"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

SENTENCE = "Barack Obama visited New York in 1961 .".split()
GOLD = ["B-person", "I-person", "O", "B-location", "I-location", "O", "O", "O"]
FOCUS = 3          # the token we are labelling: "New"
WINDOW = crf.WINDOW


def main() -> None:
    pos = crf.pos_tags(SENTENCE)
    features = crf.token_features(SENTENCE, pos, FOCUS)

    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.7)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    # --- the context window band ----------------------------------------
    box_w, gap = 1.15, 0.12
    x0 = 0.45
    lo, hi = FOCUS - WINDOW, FOCUS + WINDOW
    ax.add_patch(Rectangle((x0 + lo * (box_w + gap) - 0.10, 4.28),
                           (hi - lo + 1) * (box_w + gap) - gap + 0.20, 0.92,
                           facecolor=PALE, edgecolor="none", zorder=1))
    ax.text(x0 + lo * (box_w + gap) - 0.10, 5.52,
            f"the +-{WINDOW} context window the model may look at",
            fontsize=9.5, color=BLUE, va="bottom")

    # --- tokens, POS, gold labels ---------------------------------------
    for i, token in enumerate(SENTENCE):
        x = x0 + i * (box_w + gap)
        focused = i == FOCUS
        ax.add_patch(FancyBboxPatch(
            (x, 4.45), box_w, 0.58,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=BLUE if focused else SURFACE,
            edgecolor=BLUE if focused else GRID,
            linewidth=1.6 if focused else 1.0, zorder=3))
        ax.text(x + box_w / 2, 4.74, token, ha="center", va="center",
                fontsize=11, color="#ffffff" if focused else INK,
                fontweight="bold" if focused else "normal", zorder=4)
        ax.text(x + box_w / 2, 4.20, pos[i], ha="center", va="center",
                fontsize=8.5, color=MUTED)
        label = GOLD[i]
        ax.text(x + box_w / 2, 3.82, label, ha="center", va="center",
                fontsize=8.5,
                color=AQUA if label != "O" else MUTED,
                fontweight="bold" if label != "O" else "normal")
        if i in (FOCUS - WINDOW, FOCUS + WINDOW) or i == FOCUS:
            offset = i - FOCUS
            ax.text(x + box_w / 2, 5.10,
                    "labelling this one" if offset == 0 else f"{offset:+d}",
                    ha="center", va="bottom", fontsize=8.5,
                    color=BLUE if offset == 0 else MUTED)

    ax.text(x0 - 0.18, 4.74, "tokens", ha="right", va="center",
            fontsize=9.5, color=INK2)
    ax.text(x0 - 0.18, 4.20, "POS tag", ha="right", va="center",
            fontsize=9, color=MUTED)
    ax.text(x0 - 0.18, 3.82, "gold BIO", ha="right", va="center",
            fontsize=9, color=INK2)

    # --- the features -----------------------------------------------------
    own = [(k, v) for k, v in features.items()
           if not k[0] in "+-" and k not in {"bias"}]
    ctx = [(k, v) for k, v in features.items() if k[0] in "+-"]

    ax.text(0.45, 3.22, f'Features for "{SENTENCE[FOCUS]}" - '
                        f"{len(features)} of them, all computed from the text",
            fontsize=11, color=INK, fontweight="bold")

    ax.text(0.45, 2.88, "from the token itself", fontsize=9.5, color=BLUE)
    for row, (key, value) in enumerate(own[:9]):
        ax.text(0.45, 2.58 - row * 0.235, f"{key}", fontsize=9, color=INK2,
                family="monospace")
        ax.text(2.75, 2.58 - row * 0.235, f"{value}", fontsize=9, color=INK,
                family="monospace")

    ax.text(4.55, 2.88, f"from the {len(ctx)} context features "
                        f"(2 tokens each side)", fontsize=9.5, color=BLUE)
    show = [(k, v) for k, v in ctx
            if k.endswith(":word") or k.endswith(":pos")
            or k.endswith(":word.istitle")][:9]
    for row, (key, value) in enumerate(show):
        ax.text(4.55, 2.58 - row * 0.235, f"{key}", fontsize=9, color=INK2,
                family="monospace")
        ax.text(7.35, 2.58 - row * 0.235, f"{value}", fontsize=9, color=INK,
                family="monospace")

    ax.text(0.45, 0.10,
            'The word "New" alone is ambiguous. What tells the model it starts '
            'a location is the context:\n"visited" before it and "York" after '
            'it - and those are features -1:word and +1:word.',
            fontsize=10, color=INK2)

    ax.text(0.45, 6.40, "What the CRF sees when it labels one token",
            fontsize=13.5, color=INK, fontweight="bold")
    ax.text(0.45, 6.10, "every feature below is produced by token_features() in "
                        "scripts/approach_crf.py",
            fontsize=9.5, color=MUTED)

    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig5_crf_features.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {(FIG / 'fig5_crf_features.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

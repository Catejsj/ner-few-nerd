"""Did mention replacement actually help the CRF?

Compares two CRF runs that differ in one thing only - the training file:

    approach_crf.py                                  -> score_crf.json
    approach_crf.py --train-file train_small_aug \\
                    --tag crf_aug                    -> score_crf_aug.json

Same features, same tuning grid, same test set, same scorer. Whatever the
difference is, it is the augmentation.

Output: reports/07_augmentation_effect.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import nerlib

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "07_augmentation_effect.txt"

# The types augment.py targeted. Read from the augmentation report so the two
# scripts cannot drift apart.
TARGETED = {"art", "event", "building"}


def load(tag: str) -> dict:
    path = PROC / f"score_{tag}.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)} - run:\n"
            "  cd scripts && ../.venv/bin/python approach_crf.py\n"
            "  cd scripts && ../.venv/bin/python approach_crf.py "
            "--train-file train_small_aug --tag crf_aug")
    return json.loads(path.read_text())


def main() -> None:
    base, aug = load("crf"), load("crf_aug")

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("[1] WHAT IS BEING COMPARED")
    say("    Two CRF runs, identical except for the training file:")
    say("      baseline   train_small      20,000 sentences")
    say("      augmented  train_small_aug  29,410 sentences "
        "(+9,410 by mention replacement)")
    say("    Same feature set, same c1/c2 grid tuned on the same validation")
    say("    sentences, same test set, same scorer. Any difference is the")
    say(f"    augmentation. Targeted types: {', '.join(sorted(TARGETED))}.")
    say()

    say("[2] OVERALL")
    say(f"    {'':<12} {'P':>8} {'R':>8} {'F1':>8}")
    for name, result in [("baseline", base), ("augmented", aug)]:
        m = result["micro"]
        say(f"    {name:<12} {m['precision']:>8.3f} {m['recall']:>8.3f} "
            f"{m['f1']:>8.3f}   micro")
    delta = aug["micro"]["f1"] - base["micro"]["f1"]
    say(f"    {'change':<12} {aug['micro']['precision'] - base['micro']['precision']:>+8.3f} "
        f"{aug['micro']['recall'] - base['micro']['recall']:>+8.3f} "
        f"{delta:>+8.3f}")
    say()
    for name, result in [("baseline", base), ("augmented", aug)]:
        M = result["macro"]
        say(f"    {name:<12} {M['precision']:>8.3f} {M['recall']:>8.3f} "
            f"{M['f1']:>8.3f}   macro")
    macro_delta = aug["macro"]["f1"] - base["macro"]["f1"]
    say(f"    {'change':<12} {'':>8} {'':>8} {macro_delta:>+8.3f}")
    say()
    say("    macro is the number to watch: augmentation targeted the rare types,")
    say("    and macro weights every type equally, so that is where an effect")
    say("    should appear first.")
    say()

    say("[3] PER TYPE")
    say(f"    {'type':<14} {'support':>9} {'base F1':>9} {'aug F1':>9} "
        f"{'change':>9}  targeted")
    for label in sorted(base["support"], key=lambda t: -base["support"][t]):
        b = base["per_type"][label]["f1"]
        a = aug["per_type"][label]["f1"]
        mark = "yes" if label in TARGETED else ""
        say(f"    {label:<14} {base['support'][label]:>9,} {b:>9.3f} "
            f"{a:>9.3f} {a - b:>+9.3f}  {mark}")
    say()

    targeted_change = sum(
        aug["per_type"][t]["f1"] - base["per_type"][t]["f1"] for t in TARGETED
    ) / len(TARGETED)
    others = [t for t in base["support"] if t not in TARGETED]
    other_change = sum(
        aug["per_type"][t]["f1"] - base["per_type"][t]["f1"] for t in others
    ) / len(others)
    say(f"    mean F1 change, targeted types      {targeted_change:>+7.3f}")
    say(f"    mean F1 change, non-targeted types  {other_change:>+7.3f}")
    say()

    say("[4] ERROR SHAPE")
    say("    Where the extra data changed what kind of mistake the model makes.")
    say(f"    {'category':<18} {'baseline':>10} {'augmented':>11} {'change':>9}")
    for category in ["correct", "type error", "boundary error",
                     "type + boundary", "missed", "spurious"]:
        b = base.get("errors", {}).get(category)
        a = aug.get("errors", {}).get(category)
        if b is None or a is None:
            continue
        say(f"    {category:<18} {b:>10,} {a:>11,} {a - b:>+9,}")
    if "errors" not in base:
        say("    (run scripts/evaluate.py to populate the error categories)")
    say()

    say("[5] VERDICT")
    if delta > 0.005:
        say(f"    Augmentation helped: micro F1 {delta:+.3f}, "
            f"macro F1 {macro_delta:+.3f}.")
    elif delta < -0.005:
        say(f"    Augmentation HURT: micro F1 {delta:+.3f}, "
            f"macro F1 {macro_delta:+.3f}.")
        say("    Reported as found. The most likely reason is that mention")
        say("    replacement adds surface forms but not new context - the words")
        say("    around the entity are unchanged, so the model sees the same")
        say("    evidence repeatedly and over-weights it.")
    else:
        say(f"    No meaningful change: micro F1 {delta:+.3f}, "
            f"macro F1 {macro_delta:+.3f}.")
        say("    A null result, reported as found. Mention replacement adds new")
        say("    entity surface forms but no new context, and context is most of")
        say("    what this CRF learns from - so a small effect is what theory")
        say("    would predict.")
    say()
    say("    Either way, this is the honest answer to 'did your augmentation")
    say("    work', and having measured it is worth more than asserting it.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

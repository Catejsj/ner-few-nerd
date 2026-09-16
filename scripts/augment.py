"""Data augmentation for NER: mention replacement.

The rubric lists random replacement, insertion, deletion and back-translation.
For sequence labelling most of those are unusable as written:

  random deletion     deletes tokens, so every label after the deletion point
                      shifts and the annotation is destroyed
  random insertion     inserts a token with no label; if it lands inside an
                      entity it silently corrupts that entity's span
  back-translation     returns a fluent sentence with completely different token
                      boundaries, and nothing tells us where the entities went

The technique that does work is **mention replacement** (Dai & Adel, COLING
2020): keep the sentence, swap one entity's tokens for another entity of the
SAME type taken from the training data. The label sequence is rebuilt around the
new length, so it stays correct by construction.

    original   Barack Obama visited New York .
               B-person I-person O B-location I-location O
    augmented  Angela Merkel visited New York .
               B-person I-person O B-location I-location O

Why it is justified here: our CRF trains on a 20,000-sentence subsample, so it
is genuinely data-limited, and the rarest types (art, event, building) have the
fewest examples in that sample. Augmentation targets exactly those.

Output: data/processed/train_small_aug.jsonl, reports/03_augmentation.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import nerlib

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "03_augmentation.txt"

SEED = 42


def mention_bank(records: list[dict]) -> dict[str, list[list[str]]]:
    """type -> list of surface forms seen with that type in training."""
    bank: dict[str, set[tuple[str, ...]]] = collections.defaultdict(set)
    for record in records:
        for label, start, end in nerlib.to_spans(record["bio"]):
            bank[label].add(tuple(record["tokens"][start:end]))
    return {label: [list(m) for m in sorted(forms)] for label, forms in bank.items()}


def replace_mentions(record: dict, bank: dict, targets: set[str],
                     rng: random.Random) -> dict | None:
    """Return a copy with every target-type mention swapped, or None."""
    spans = sorted(nerlib.to_spans(record["bio"]), key=lambda s: s[1])
    hits = [s for s in spans if s[0] in targets]
    if not hits:
        return None

    tokens, bio = [], []
    cursor = 0
    changed = False
    for label, start, end in spans:
        tokens.extend(record["tokens"][cursor:start])
        bio.extend(record["bio"][cursor:start])
        if label in targets and bank.get(label):
            new = rng.choice(bank[label])
            if new != record["tokens"][start:end]:
                changed = True
            tokens.extend(new)
            bio.extend([f"B-{label}"] + [f"I-{label}"] * (len(new) - 1))
        else:
            tokens.extend(record["tokens"][start:end])
            bio.extend(record["bio"][start:end])
        cursor = end
    tokens.extend(record["tokens"][cursor:])
    bio.extend(record["bio"][cursor:])

    if not changed:
        return None
    assert len(tokens) == len(bio)
    return {"id": record["id"], "tokens": tokens, "bio": bio}


def counts(records: list[dict]) -> collections.Counter:
    c: collections.Counter[str] = collections.Counter()
    for record in records:
        for label, _, _ in nerlib.to_spans(record["bio"]):
            c[label] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="train_small",
                    help="file to augment")
    ap.add_argument("--rare", type=int, default=3,
                    help="how many of the least frequent types to target")
    ap.add_argument("--copies", type=int, default=2,
                    help="augmented copies per eligible sentence")
    args = ap.parse_args()

    base = nerlib.load(args.base)
    full_train = nerlib.load("train_clean")
    rng = random.Random(SEED)

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    before = counts(base)
    targets = {label for label, _ in before.most_common()[-args.rare:]}

    say("[1] WHY AUGMENT AT ALL")
    say(f"    Base file: {args.base}, {len(base):,} sentences, "
        f"{sum(before.values()):,} entities.")
    say("    The full training split has 131,767 sentences, but the CRF is fitted")
    say("    on a subsample for memory reasons, so from the CRF's point of view")
    say("    this IS a limited-data setting - which is the condition the rubric")
    say("    asks augmentation to be justified by.")
    say()
    say("    entity counts in the base file")
    for label, count in before.most_common():
        mark = "  <- targeted" if label in targets else ""
        say(f"      {label:<14} {count:>7,}{mark}")
    say()
    say(f"    Targeting the {args.rare} rarest types: {', '.join(sorted(targets))}.")
    say()

    say("[2] WHY MENTION REPLACEMENT AND NOT THE OTHER METHODS")
    say("    random deletion   shifts every label after the deleted token")
    say("    random insertion  can land inside an entity and corrupt its span")
    say("    back-translation  returns different token boundaries with no way to")
    say("                      map the old labels onto the new tokens")
    say("    mention replacement  keeps the sentence and swaps one entity for")
    say("                      another of the same type; the labels are rebuilt")
    say("                      around the new length, so they stay correct.")
    say("    Reference: Dai and Adel (2020), 'An Analysis of Simple Data")
    say("    Augmentation for Named Entity Recognition'.")
    say()

    # The bank is drawn from the FULL training split, not the subsample: that is
    # where the extra information comes from. It is still training data, so
    # there is no leakage from validation or test.
    bank = mention_bank(full_train)
    say("[3] THE MENTION BANK")
    say("    Built from the full TRAINING split (never validation or test).")
    for label in sorted(bank):
        mark = "  <- used" if label in targets else ""
        say(f"      {label:<14} {len(bank[label]):>7,} distinct surface forms{mark}")
    say()

    augmented: list[dict] = []
    for record in base:
        for _ in range(args.copies):
            new = replace_mentions(record, bank, targets, rng)
            if new:
                augmented.append(new)

    combined = base + augmented
    after = counts(combined)

    say("[4] RESULT")
    say(f"    {'type':<14} {'before':>8} {'after':>8} {'change':>9}")
    for label, count in before.most_common():
        grown = after[label]
        say(f"    {label:<14} {count:>8,} {grown:>8,} "
            f"{grown / count - 1:>+8.0%}")
    say()
    say(f"    sentences  {len(base):,} -> {len(combined):,} "
        f"(+{len(augmented):,} augmented)")
    say(f"    entities   {sum(before.values()):,} -> {sum(after.values()):,}")
    say()
    say("    Note the non-targeted types grow too. An augmented sentence keeps its")
    say("    other entities, so a sentence containing both an art entity and a")
    say("    person contributes that person again. The targeted types still grow")
    say("    fastest, which is the point.")
    say()

    say("[5] WORKED EXAMPLES")
    shown = 0
    for record in base:
        new = replace_mentions(record, bank, targets, rng)
        if not new or len(record["tokens"]) > 22:
            continue
        say(f"    before  {' '.join(record['tokens'])}")
        say(f"    after   {' '.join(new['tokens'])}")
        gold_ents = [(l, " ".join(record["tokens"][a:b]))
                     for l, a, b in sorted(nerlib.to_spans(record["bio"]),
                                           key=lambda s: s[1])]
        new_ents = [(l, " ".join(new["tokens"][a:b]))
                    for l, a, b in sorted(nerlib.to_spans(new["bio"]),
                                          key=lambda s: s[1])]
        say(f"      entities before: {gold_ents}")
        say(f"      entities after:  {new_ents}")
        say()
        shown += 1
        if shown == 3:
            break

    say("[6] THE HONEST WARNING")
    say("    Mention replacement produces sentences that are often factually")
    say("    absurd - a director being released as a film, a war fought in a")
    say("    museum. That is acceptable for a CRF, which learns from surface")
    say("    patterns and context words, not from world knowledge. It would NOT be")
    say("    acceptable if we were fine-tuning a language model, which can be")
    say("    damaged by implausible text.")
    say()
    say("    Whether it actually helped is an empirical question, answered by")
    say("    rerunning the CRF on this file:")
    say("      ./.venv/bin/python scripts/approach_crf.py --train-file train_small_aug")
    say("    and comparing reports/04_crf.txt before and after. The comparison is")
    say("    recorded in reports/07_augmentation_effect.txt.")

    out = PROC / f"{args.base}_aug.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for record in combined:
            fh.write(json.dumps(record) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(combined):,} sentences)")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

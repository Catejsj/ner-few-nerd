"""Approach 1 of 3: the gazetteer (dictionary lookup).

The oldest NER method there is. Build a list of known entity names and their
types, then scan each sentence for the longest matching phrase.

It has no model and learns no weights, so it is the baseline the other two have
to beat. It is also genuinely useful: for closed classes (countries, currencies)
a dictionary is hard to improve on.

The gazetteer is built from the TRAINING split only. Building it from all the
data - including test - would be the most direct form of cheating available in
this project, because the gazetteer would then contain the answers.

Three design choices worth defending in the presentation:

  longest match    "New York Times" is matched before "New York", otherwise
                   every long name is chopped into its first two tokens.
  ambiguity        a surface form can appear with several types in training
                   ("Washington" is a person and a location). We take the most
                   frequent type and report how often that is wrong.
  case             matching is case-sensitive by default. --ignore-case is
                   available and measured: it raises recall and destroys
                   precision, which is the trade this approach is made of.

Output: data/processed/pred_gazetteer.jsonl, models/gazetteer.json,
        reports/03_gazetteer.txt
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import nerlib

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORT = ROOT / "reports" / "03_gazetteer.txt"

MAX_SPAN = 8  # longest entity we will try to match, in tokens


def build(records: list[dict], min_count: int) -> tuple[dict, dict]:
    """surface form -> most frequent entity type, plus the ambiguity record."""
    seen: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    for record in records:
        for label, start, end in nerlib.to_spans(record["bio"]):
            surface = " ".join(record["tokens"][start:end])
            seen[surface][label] += 1

    gazetteer, ambiguous = {}, {}
    for surface, counter in seen.items():
        total = sum(counter.values())
        if total < min_count:
            continue
        label, count = counter.most_common(1)[0]
        gazetteer[surface] = label
        if len(counter) > 1:
            ambiguous[surface] = dict(counter)
    return gazetteer, ambiguous


def tag_sentence(tokens: list[str], gazetteer: dict, ignore_case: bool) -> list[str]:
    """Longest-match left to right, no overlaps."""
    n = len(tokens)
    bio = ["O"] * n
    i = 0
    while i < n:
        for length in range(min(MAX_SPAN, n - i), 0, -1):
            phrase = " ".join(tokens[i:i + length])
            key = phrase.lower() if ignore_case else phrase
            label = gazetteer.get(key)
            if label:
                bio[i] = f"B-{label}"
                for j in range(i + 1, i + length):
                    bio[j] = f"I-{label}"
                i += length
                break
        else:
            i += 1
    return bio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=None,
                    help="how many times a name must appear in training to enter "
                         "the gazetteer; default: chosen on validation")
    ap.add_argument("--ignore-case", action="store_true")
    args = ap.parse_args()

    train = nerlib.load("train_clean")
    val = nerlib.load("validation_clean")
    test = nerlib.load("test_clean")

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    # --- choose min_count on validation ----------------------------------
    say("[0] CHOOSING THE ONE PARAMETER, ON VALIDATION")
    say("    min_count is how many times a name must appear in training before it")
    say("    is trusted. At 1, every one-off name enters the list, including words")
    say("    that happen to be song titles ('Today', 'Hello'), and they then match")
    say("    everywhere. Raising it trades recall for precision.")
    say()
    say("    Test is not touched here. The value is chosen on validation and used")
    say("    once on test, which is the only honest order.")
    say()
    say(f"    {'min_count':>10} {'entries':>10} {'P':>7} {'R':>7} {'F1':>7}")
    val_gold = [r["bio"] for r in val]
    grid = [args.min_count] if args.min_count else [1, 2, 3, 5, 10]
    best, best_f1 = grid[0], -1.0
    for candidate in grid:
        g, _ = build(train, candidate)
        if args.ignore_case:
            g = {k.lower(): v for k, v in g.items()}
        preds = [tag_sentence(r["tokens"], g, args.ignore_case) for r in val]
        res = nerlib.score(val_gold, preds)
        say(f"    {candidate:>10} {len(g):>10,} {res['micro']['precision']:>7.3f} "
            f"{res['micro']['recall']:>7.3f} {res['micro']['f1']:>7.3f}")
        if res["micro"]["f1"] > best_f1:
            best, best_f1 = candidate, res["micro"]["f1"]
    say()
    say(f"    chosen min_count = {best}  (validation micro F1 {best_f1:.3f})")
    say()

    gazetteer, ambiguous = build(train, best)
    if args.ignore_case:
        gazetteer = {k.lower(): v for k, v in gazetteer.items()}
    args.min_count = best

    say("[1] THE GAZETTEER")
    say(f"    built from       the {len(train):,} TRAINING sentences only")
    say(f"    entries          {len(gazetteer):,} distinct surface forms")
    say(f"    min count        {args.min_count} "
        f"(a name must appear this often to be included)")
    say(f"    matching         longest first, up to {MAX_SPAN} tokens, no overlaps")
    say(f"    case             {'ignored' if args.ignore_case else 'respected'}")
    say()
    by_type = collections.Counter(gazetteer.values())
    say("    entries per type")
    for label, count in by_type.most_common():
        say(f"      {label:<14} {count:>8,}")
    say()

    say("[2] AMBIGUITY INSIDE THE GAZETTEER")
    say(f"    {len(ambiguous):,} surface forms appear with more than one type in")
    say(f"    training ({len(ambiguous) / len(gazetteer):.1%} of entries). We keep "
        "the most frequent type,")
    say("    so every one of these is a guaranteed error whenever the other")
    say("    reading is the correct one.")
    say()
    worst = sorted(ambiguous.items(), key=lambda kv: -sum(kv[1].values()))[:10]
    for surface, counter in worst:
        spread = ", ".join(f"{k} {v}" for k, v in
                           sorted(counter.items(), key=lambda kv: -kv[1]))
        say(f"      {surface:<26} {spread}")
    say()
    say("    This is the structural weakness of the approach: a dictionary has no")
    say("    context, so it cannot tell which reading is meant.")
    say()

    # --- coverage: how much of test could a perfect dictionary even see? --
    test_surfaces = collections.Counter()
    for record in test:
        for label, start, end in nerlib.to_spans(record["bio"]):
            test_surfaces[" ".join(record["tokens"][start:end])] += 1
    lookup = {k.lower() for k in gazetteer} if args.ignore_case else set(gazetteer)
    seen_count = sum(c for s, c in test_surfaces.items()
                     if (s.lower() if args.ignore_case else s) in lookup)
    total_count = sum(test_surfaces.values())

    say("[3] CEILING")
    say(f"    {seen_count:,} of {total_count:,} test entity mentions "
        f"({seen_count / total_count:.1%}) have their")
    say("    exact surface form in the gazetteer. The rest are names the training")
    say("    data never contained, and no dictionary can find them.")
    say(f"    So {1 - seen_count / total_count:.1%} recall is lost before matching "
        "even starts. This is the")
    say("    number that explains the result below.")
    say()

    predictions = [tag_sentence(r["tokens"], gazetteer, args.ignore_case)
                   for r in test]
    gold = [r["bio"] for r in test]
    result = nerlib.score(gold, predictions)

    say("[4] RESULT ON TEST  (entity level, exact span match)")
    nerlib.report_table(result, say)
    say()
    say(f"    micro F1 {result['micro']['f1']:.3f}, "
        f"macro F1 {result['macro']['f1']:.3f}")
    say()
    micro = result["micro"]
    say(f"    The model fires {micro['tp'] + micro['fp']:,} times and is right "
        f"{micro['tp']:,} of those.")
    if micro["precision"] < micro["recall"]:
        say("    Precision is BELOW recall, which is not what a dictionary is")
        say("    supposed to do, and it is the most interesting thing in this")
        say("    report. The cause is in [2]: a gazetteer built from real text")
        say("    contains ordinary words that happen to be names of songs, films")
        say("    and books. In ours, 'Today', 'Why' and 'Hello' are all entered")
        say("    as art, so they now match every ordinary use of those words.")
        worst = max(result["per_type"].items(), key=lambda kv: kv[1]["fp"])
        say(f"    Worst offender: {worst[0]} with {worst[1]['fp']:,} false "
            f"positives against {worst[1]['tp']:,} correct.")
    else:
        say("    Precision is above recall, the classic dictionary behaviour: what")
        say("    it knows it gets right, what it has not seen it cannot find.")

    MODELS.mkdir(exist_ok=True)
    (MODELS / "gazetteer.json").write_text(
        json.dumps({"entries": gazetteer, "ambiguous": ambiguous}), encoding="utf-8")
    nerlib.save_predictions("gazetteer", predictions)
    (ROOT / "data" / "processed" / "score_gazetteer.json").write_text(
        json.dumps(result, indent=1))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

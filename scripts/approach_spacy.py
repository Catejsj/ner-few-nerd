"""Approach 3 of 3: spaCy's pre-trained English NER model.

This model was never trained on Few-NERD. It was trained on OntoNotes 5 and
predicts 18 OntoNotes entity types. Two problems follow, and both have to be
solved before a fair comparison is possible.

1. TOKENIZATION. spaCy normally runs its own tokenizer, which can split a token
   Few-NERD kept whole. The predicted spans would then not line up with the gold
   spans and the score would measure tokenizer disagreement rather than NER
   quality. We build each Doc with Doc(nlp.vocab, words=gold_tokens), which
   bypasses the tokenizer entirely and guarantees alignment. The mismatch we
   avoid by doing this is measured in [2].

2. LABEL MISMATCH. OntoNotes PERSON obviously corresponds to Few-NERD person,
   but NORP (nationalities, religious and political groups) has no obvious home,
   and Few-NERD's "other" is a bag of languages, laws, awards and diseases. So
   the mapping is not written from intuition - it is LEARNED FROM VALIDATION:
   for every OntoNotes label we find which Few-NERD type its spans actually
   coincide with, and map it there. Test is never used for this.

Output: data/processed/pred_spacy.jsonl, reports/05_spacy.txt
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import spacy
from spacy.tokens import Doc

import nerlib

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "05_spacy.txt"

MODEL = "en_core_web_sm"

# OntoNotes labels that describe numbers, dates and quantities. Few-NERD has no
# equivalent type, so a prediction of one of these is dropped rather than
# force-fitted into a category it does not belong to.
NUMERIC = {"CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"}


def predict_raw(nlp, records: list[dict]) -> list[list[tuple[str, int, int]]]:
    """Run spaCy over gold tokens; return OntoNotes spans per sentence."""
    docs = (Doc(nlp.vocab, words=r["tokens"]) for r in records)
    out = []
    for doc in nlp.pipe(docs, batch_size=256):
        out.append([(e.label_, e.start, e.end) for e in doc.ents])
    return out


def learn_mapping(records: list[dict], raw: list[list[tuple[str, int, int]]],
                  say) -> dict[str, str]:
    """For each OntoNotes label, the Few-NERD type its spans line up with."""
    agree: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    fired: collections.Counter[str] = collections.Counter()
    for record, ents in zip(records, raw):
        gold = {(a, b): label for label, a, b in nerlib.to_spans(record["bio"])}
        for label, a, b in ents:
            fired[label] += 1
            if (a, b) in gold:
                agree[label][gold[(a, b)]] += 1

    say(f"    {'OntoNotes':<14} {'fired':>8} {'exact hits':>11} {'-> Few-NERD':<16} "
        f"{'purity':>8}")
    mapping: dict[str, str] = {}
    for label in sorted(fired, key=lambda x: -fired[x]):
        counter = agree[label]
        if label in NUMERIC:
            say(f"    {label:<14} {fired[label]:>8,} {sum(counter.values()):>11,} "
                f"{'(dropped)':<16}")
            continue
        if not counter:
            say(f"    {label:<14} {fired[label]:>8,} {0:>11,} "
                f"{'(no overlap)':<16}")
            continue
        target, hits = counter.most_common(1)[0]
        purity = hits / sum(counter.values())
        mapping[label] = target
        say(f"    {label:<14} {fired[label]:>8,} {sum(counter.values()):>11,} "
            f"{target:<16} {purity:>7.1%}")
    return mapping


def apply_mapping(records, raw, mapping) -> list[list[str]]:
    predictions = []
    for record, ents in zip(records, raw):
        bio = ["O"] * len(record["tokens"])
        for label, a, b in ents:
            target = mapping.get(label)
            if not target:
                continue
            bio[a] = f"B-{target}"
            for j in range(a + 1, b):
                bio[j] = f"I-{target}"
        predictions.append(bio)
    return predictions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    try:
        nlp = spacy.load(args.model, exclude=["lemmatizer"])
    except OSError:
        raise SystemExit(f"run: ./.venv/bin/python -m spacy download {args.model}")

    val = nerlib.load("validation_clean")
    test = nerlib.load("test_clean")

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("[1] MODEL")
    say(f"    model            {args.model} v{nlp.meta['version']}")
    say(f"    trained on       {nlp.meta.get('sources', [{'name': 'OntoNotes 5'}])[0].get('name', 'OntoNotes 5')}")
    say(f"    licence          {nlp.meta.get('license', 'MIT')}")
    say(f"    entity types     {len(nlp.get_pipe('ner').labels)} OntoNotes types")
    say("    training data    OntoNotes 5 - news, telephone speech, weblogs.")
    say("                     Few-NERD is Wikipedia. This is a domain shift, and")
    say("                     it is a fair part of what we are measuring: how far")
    say("                     an off-the-shelf model transfers.")
    say("    NOT fine-tuned on Few-NERD. Zero training sentences were used.")
    say()

    say("[2] TOKENIZATION ALIGNMENT")
    mismatch = 0
    checked = 0
    for record in val[:3000]:
        checked += 1
        own = [t.text for t in nlp.tokenizer(" ".join(record["tokens"]))]
        if own != record["tokens"]:
            mismatch += 1
    say(f"    If we let spaCy tokenize the raw sentence text, it produces a")
    say(f"    different token sequence from Few-NERD's in {mismatch:,} of "
        f"{checked:,} sentences ({mismatch / checked:.1%}).")
    say("    Every one of those would misalign the labels.")
    say("    We therefore pass the gold tokens in directly with Doc(words=...),")
    say("    which turns the tokenizer off. Alignment is then exact by")
    say("    construction, and the comparison measures NER, not tokenization.")
    say()

    raw_val = predict_raw(nlp, val)
    say("[3] LEARNING THE LABEL MAPPING  (on validation only)")
    say("    'fired' is how many spans spaCy predicted with that label.")
    say("    'exact hits' is how many of those exactly match a gold Few-NERD span.")
    say("    'purity' is how consistently those hits point at one Few-NERD type -")
    say("    low purity means the mapping is a compromise, not a translation.")
    say()
    mapping = learn_mapping(val, raw_val, say)
    say()
    say(f"    Final mapping: " + ", ".join(f"{k}->{v}" for k, v in
                                           sorted(mapping.items())))
    say()
    say("    Numeric types (" + ", ".join(sorted(NUMERIC)) + ") are dropped:")
    say("    Few-NERD has no type for dates, money or counts, so predicting them")
    say("    can only produce false positives. Dropping them is the mapping")
    say("    decision that most helps spaCy's score, and we state it openly.")
    say()

    raw_test = predict_raw(nlp, test)
    predictions = apply_mapping(test, raw_test, mapping)
    gold = [r["bio"] for r in test]
    result = nerlib.score(gold, predictions)

    say("[4] RESULT ON TEST  (entity level, exact span match)")
    nerlib.report_table(result, say)
    say()
    say(f"    micro F1 {result['micro']['f1']:.3f}, "
        f"macro F1 {result['macro']['f1']:.3f}")
    say()
    say("    Read this as a transfer result, not as spaCy's quality. On OntoNotes")
    say("    text this model scores about 0.85 F1. Here it is being asked to")
    say("    predict a different label scheme on a different domain with no")
    say("    training, so the gap is the cost of the mismatch.")
    say()

    say("[5] WHAT IT CANNOT DO BY DESIGN")
    unmapped = sorted(set(nlp.get_pipe("ner").labels) - set(mapping) - NUMERIC)
    if unmapped:
        say(f"    OntoNotes labels with no Few-NERD equivalent found: "
            f"{', '.join(unmapped)}")
    missing = [t for t in nerlib.TYPES if t not in set(mapping.values())]
    say(f"    Few-NERD types no OntoNotes label maps onto: "
        f"{', '.join(missing) if missing else 'none'}")
    if missing:
        support = sum(result["support"][t] for t in missing)
        say(f"    Those types are {support:,} test entities "
            f"({support / sum(result['support'].values()):.1%}) that spaCy cannot")
        say("    score above zero on, no matter how good it is.")

    nerlib.save_predictions("spacy", predictions)
    (ROOT / "data" / "processed" / "score_spacy.json").write_text(
        json.dumps(result, indent=1))
    (ROOT / "models" / "spacy_label_map.json").write_text(
        json.dumps(mapping, indent=1))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

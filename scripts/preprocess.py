"""Preprocessing for NER - including the steps we deliberately do not apply.

NER is sequence labelling: every token carries a label, and the answer is a span
of the original text. That changes what preprocessing is allowed. Anything that
deletes, merges or reorders tokens breaks the alignment between tokens and
labels and makes the task unanswerable.

So this script does two jobs. It applies the preprocessing that is safe, and it
*measures* the damage the unsafe steps would do, so that "we did not remove
stopwords" is backed by a number instead of an opinion.

  applied      unicode normalisation, whitespace repair, a shape feature,
               lemma as a CRF feature
  not applied  stopword removal, lowercasing, stemming of the sequence itself -
               each with the measurement that justifies it

Output: data/processed/{split}_clean.jsonl, data/processed/train_small.jsonl,
        reports/02_preprocessing.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import unicodedata
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "02_preprocessing.txt"

SPLITS = ["train", "validation", "test"]

# The CRF is fitted on a subsample. 131,767 sentences with the feature set in
# approach_crf.py needs far more memory than this machine has; 20,000 keeps
# every entity type in the thousands and trains in minutes. The effect of the
# sample size is measured in reports/04_crf.txt.
SUBSAMPLE = 20_000
SEED = 42


def load(split: str) -> list[dict]:
    path = PROC / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit("run scripts/build_corpus.py first")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def normalize_token(token: str) -> str:
    """Unicode repair only. Case is preserved on purpose - see [3]."""
    token = unicodedata.normalize("NFKC", token)
    token = token.replace("’", "'").replace("‘", "'")
    token = token.replace("“", '"').replace("”", '"')
    token = token.replace("–", "-").replace("—", "-")
    token = re.sub(r"\s+", "", token)
    return token


def shape(token: str) -> str:
    """Xxxx / dddd / xxx style summary of a token's characters."""
    s = re.sub(r"[A-Z]", "X", token)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"[0-9]", "d", s)
    return re.sub(r"(.)\1{3,}", r"\1\1\1\1", s)


def entities(tokens: list[str], bio: list[str]) -> list[tuple[str, list[str]]]:
    out, current, label = [], [], None
    for token, tag in zip(tokens + ["."], bio + ["O"]):
        if tag.startswith("B-") or tag == "O":
            if current:
                out.append((label, current))
            current, label = ([token], tag[2:]) if tag.startswith("B-") else ([], None)
        elif tag.startswith("I-"):
            current.append(token)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=SUBSAMPLE)
    args = ap.parse_args()

    for package in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(package, quiet=True)
    stop = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    data = {split: load(split) for split in SPLITS}
    train = data["train"]

    say("[1] TOKENIZATION")
    say("    We did not tokenize. Few-NERD ships pre-tokenized sentences, and the")
    say("    labels are aligned to those exact tokens. Re-tokenizing would shift")
    say("    every label by an unknown amount.")
    say()
    say("    Few-NERD's own tokenization follows the Wikipedia dump processing in")
    say("    Ding et al. (2021): Penn Treebank style, so clitics are split")
    say("    (\"People 's Republic of China\") and punctuation is its own token.")
    say()
    say("    This matters for the spaCy approach. spaCy runs its own tokenizer, so")
    say("    it can split a token Few-NERD kept whole, and then its predicted spans")
    say("    do not line up with the gold spans. scripts/approach_spacy.py handles")
    say("    this by feeding spaCy the gold tokens directly with Doc(words=...),")
    say("    which disables its tokenizer and forces alignment. The measured")
    say("    mismatch if we had not done that is reported in reports/05_spacy.txt.")
    say()
    lengths = [len(d["tokens"]) for d in train]
    say(f"    training sentences        {len(train):,}")
    say(f"    tokens per sentence       mean {sum(lengths) / len(lengths):.1f}, "
        f"min {min(lengths)}, max {max(lengths)}")
    say()

    say("[2] NORMALIZATION  (applied)")
    changed = collections.Counter()
    for record in train[:20000]:
        for token in record["tokens"]:
            fixed = normalize_token(token)
            if fixed != token:
                changed[(token, fixed)] += 1
    say("    Applied per token, without changing how many tokens there are:")
    say("      - Unicode NFKC, so a full-width or composed character becomes its")
    say("        ordinary form and does not become its own vocabulary entry")
    say("      - curly quotes and dashes folded to ASCII")
    say("      - stray whitespace inside a token removed")
    say(f"    {sum(changed.values()):,} tokens changed in the first 20,000 "
        "sentences.")
    for (before, after), count in changed.most_common(6):
        say(f"      {before!r} -> {after!r}   x{count}")
    say()
    say("    Case is NOT normalised. Measurement in [3].")
    say()

    say("[3] LOWERCASING  (deliberately NOT applied)")
    ent_tokens = [(t, b) for d in train for t, b in zip(d["tokens"], d["bio"])
                  if b != "O"]
    cap = sum(1 for t, _ in ent_tokens if t[:1].isupper())
    o_tokens = [t for d in train for t, b in zip(d["tokens"], d["bio"]) if b == "O"]
    o_cap = sum(1 for t in o_tokens if t[:1].isupper())
    say(f"    {cap / len(ent_tokens):.1%} of entity tokens start with a capital "
        f"letter.")
    say(f"    {o_cap / len(o_tokens):.1%} of non-entity tokens do.")
    say("    Capitalisation is the single strongest surface clue an English NER")
    say("    system has. Lowercasing the text would delete it, so we keep case and")
    say("    give the models capitalisation as an explicit feature instead.")
    say()

    say("[4] STOPWORD REMOVAL  (deliberately NOT applied)")
    with_stop = 0
    examples: collections.Counter[str] = collections.Counter()
    total_ents = 0
    for record in train:
        for label, tokens in entities(record["tokens"], record["bio"]):
            total_ents += 1
            if any(t.lower() in stop for t in tokens):
                with_stop += 1
                examples[" ".join(tokens)] += 1
    say(f"    {with_stop:,} of {total_ents:,} entities ({with_stop / total_ents:.1%}) "
        "contain at least one")
    say("    stopword. Removing stopwords would break every one of them:")
    for phrase, count in examples.most_common(8):
        say(f"      {phrase:<34} x{count:,}")
    say()
    say("    It would also destroy the token-label alignment for the whole")
    say("    sentence, because the labels are positional. Stopword removal is a")
    say("    document-classification technique; it does not transfer to sequence")
    say("    labelling.")
    say()

    say("[5] STEMMING AND LEMMATIZATION  (not on the sequence, yes as a feature)")
    say("    We do not stem or lemmatize the text itself: the output of NER is a")
    say("    span of the original sentence, and 'Univers of Californ' is not a span")
    say("    of anything.")
    say()
    say("    We do use the lemma as one CRF feature among many, which is the")
    say("    standard way to get the benefit without damaging the text:")
    for word in ["Games", "Studies", "Airlines", "Brothers", "Islands"]:
        say(f"      {word:<12} stem -> {stemmer.stem(word):<10} "
            f"lemma -> {lemmatizer.lemmatize(word.lower())}")
    say("    The lemma helps the model learn that 'Islands' and 'Island' behave the")
    say("    same way inside a location name, while the surface form is still there")
    say("    for the model to use.")
    say()

    say("[6] SHAPE FEATURE  (applied, as a feature)")
    shapes = collections.Counter(
        shape(t) for d in train[:20000] for t, b in zip(d["tokens"], d["bio"])
        if b != "O")
    say("    Every token gets a shape string: capitals become X, lowercase x,")
    say("    digits d. This lets the CRF generalise over words it has never seen.")
    say("    Most common shapes inside entities:")
    for pattern, count in shapes.most_common(6):
        say(f"      {pattern:<10} {count:>8,}")
    say()

    # --- write the cleaned files -----------------------------------------
    for split in SPLITS:
        out = PROC / f"{split}_clean.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for record in data[split]:
                fh.write(json.dumps({
                    "id": record["id"],
                    "tokens": [normalize_token(t) or t for t in record["tokens"]],
                    "bio": record["bio"],
                }) + "\n")

    rng = random.Random(SEED)
    sample = rng.sample(train, min(args.subsample, len(train)))
    with (PROC / "train_small.jsonl").open("w", encoding="utf-8") as fh:
        for record in sample:
            fh.write(json.dumps({
                "id": record["id"],
                "tokens": [normalize_token(t) or t for t in record["tokens"]],
                "bio": record["bio"],
            }) + "\n")

    sample_types: collections.Counter[str] = collections.Counter()
    for record in sample:
        for label, _ in entities(record["tokens"], record["bio"]):
            sample_types[label] += 1

    say("[7] TRAINING SUBSAMPLE FOR THE CRF")
    say(f"    {len(sample):,} sentences drawn at random from the {len(train):,} "
        "training sentences,")
    say(f"    seed {SEED}. Entities in the sample:")
    for label, count in sample_types.most_common():
        say(f"      {label:<14} {count:>7,}")
    say()
    say("    Why subsample: the CRF feature set produces roughly two million")
    say("    features over the full training set, which does not fit in memory on")
    say("    the machine we are using. The gazetteer and spaCy approaches use the")
    say("    full training set, and the learning curve in reports/04_crf.txt shows")
    say("    what the CRF gives up.")
    say()
    say("[8] AUGMENTATION")
    say("    Handled separately in scripts/augment.py, because for NER the useful")
    say("    method is entity replacement, not the word-level swap and deletion the")
    say("    rubric lists. See reports/03_augmentation.txt.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {(PROC / 'train_small.jsonl').relative_to(ROOT)} "
          f"({len(sample):,} sentences)")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Approach 2 of 3: a Conditional Random Field.

A CRF labels the whole sentence at once. Unlike a per-token classifier it scores
the *sequence* of labels, so it can learn that I-person cannot follow B-location
and that an entity usually starts after a determiner. That is why CRFs were the
standard for NER for a decade before neural models.

What the model sees for each token: the token itself, its shape, its
capitalisation, its prefixes and suffixes, its lemma, its part of speech, and
the same information for the two tokens on each side. No word embeddings and no
pre-trained knowledge - everything it knows comes from the training sentences.

Two regularisation parameters are tuned on validation:
  c1  L1 penalty. Pushes useless feature weights to exactly zero.
  c2  L2 penalty. Keeps all weights small and stops any one feature dominating.

Output: models/crf.pkl, data/processed/pred_crf.jsonl, reports/04_crf.txt
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import nltk
import sklearn_crfsuite
from nltk.stem import WordNetLemmatizer

import nerlib

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORT = ROOT / "reports" / "04_crf.txt"

WINDOW = 2          # how many tokens either side the model may look at
MAX_ITERATIONS = 120
AFFIX = 3           # length of the prefix/suffix features

# Lemmatization is not applied to the text - the answer has to be a span of the
# original sentence. It is used here as one feature among many, which is how a
# CRF gets the generalisation ("Islands" and "Island" behave alike inside a
# location name) without damaging the tokens.
LEMMATIZER = WordNetLemmatizer()


def pos_tags(tokens: list[str]) -> list[str]:
    return [tag for _, tag in nltk.pos_tag(tokens)]


def token_features(tokens: list[str], pos: list[str], i: int) -> dict:
    token = tokens[i]
    features = {
        "bias": 1.0,
        "word": token,
        "word.lower": token.lower(),
        "word.lemma": LEMMATIZER.lemmatize(token.lower()),
        "word.shape": shape(token),
        "word.istitle": token.istitle(),
        "word.isupper": token.isupper(),
        "word.isdigit": token.isdigit(),
        "word.hasdigit": any(c.isdigit() for c in token),
        "word.hashyphen": "-" in token,
        f"word.suffix{AFFIX}": token[-AFFIX:].lower(),
        f"word.prefix{AFFIX}": token[:AFFIX].lower(),
        "pos": pos[i],
        "pos[:2]": pos[i][:2],
    }
    for offset in range(-WINDOW, WINDOW + 1):
        if offset == 0:
            continue
        j = i + offset
        key = f"{offset:+d}"
        if 0 <= j < len(tokens):
            neighbour = tokens[j]
            features.update({
                f"{key}:word": neighbour,
                f"{key}:word.lower": neighbour.lower(),
                f"{key}:word.istitle": neighbour.istitle(),
                f"{key}:word.shape": shape(neighbour),
                f"{key}:pos": pos[j],
            })
        else:
            features[f"{key}:padding"] = True
    if i == 0:
        features["BOS"] = True          # beginning of sentence
    if i == len(tokens) - 1:
        features["EOS"] = True          # end of sentence
    return features


def shape(token: str) -> str:
    import re
    s = re.sub(r"[A-Z]", "X", token)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"[0-9]", "d", s)
    return re.sub(r"(.)\1{3,}", r"\1\1\1\1", s)


def featurize(records: list[dict]) -> tuple[list, list]:
    X, y = [], []
    for record in records:
        tokens = record["tokens"]
        pos = pos_tags(tokens)
        X.append([token_features(tokens, pos, i) for i in range(len(tokens))])
        y.append(list(record["bio"]))
    return X, y


def fit(X, y, c1: float, c2: float) -> sklearn_crfsuite.CRF:
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=c1, c2=c2,
        max_iterations=MAX_ITERATIONS, all_possible_transitions=True)
    crf.fit(X, y)
    return crf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default="train_small",
                    help="train_small (20k) or train_clean (full)")
    ap.add_argument("--c1", type=float, default=None)
    ap.add_argument("--c2", type=float, default=None)
    ap.add_argument("--tag", default="crf",
                    help="name for the outputs; use a different one for a "
                         "second run so the first is not overwritten")
    ap.add_argument("--val-size", type=int, default=4000,
                    help="validation sentences used for tuning, for speed")
    args = ap.parse_args()

    for package in ["averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"]:
        nltk.download(package, quiet=True)

    train = nerlib.load(args.train_file)
    val = nerlib.load("validation_clean")[:args.val_size]
    test = nerlib.load("test_clean")

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("[1] SETUP")
    say(f"    training sentences   {len(train):,}  ({args.train_file})")
    say(f"    validation used      {len(val):,}  (for tuning only)")
    say(f"    test sentences       {len(test):,}")
    say(f"    context window       +-{WINDOW} tokens")
    say(f"    optimiser            L-BFGS, max {MAX_ITERATIONS} iterations")
    say("    all_possible_transitions=True - the model may learn a weight for")
    say("    every label pair, including ones never seen, which is what lets it")
    say("    learn that I-person after B-location is impossible.")
    say()

    t0 = time.time()
    X_train, y_train = featurize(train)
    X_val, y_val = featurize(val)
    X_test, y_test = featurize(test)
    say(f"    feature extraction took {time.time() - t0:.0f}s")
    n_keys = len({k for sent in X_train for tok in sent for k in tok})
    n_values = len({(k, str(v)) for sent in X_train for tok in sent
                    for k, v in tok.items()})
    say(f"    feature templates    {n_keys}  (word, shape, pos, +-2 context, ...)")
    say(f"    distinct feature-value pairs {n_values:,}  - this is what the model")
    say("    actually learns a weight for, and it is why memory is the binding")
    say("    constraint on the training set size.")
    say()

    say("[2] TUNING c1 AND c2 ON VALIDATION")
    say("    c1 is the L1 penalty: it drives weak feature weights to exactly zero,")
    say("    which matters here because most of our features are one-off words.")
    say("    c2 is the L2 penalty: it keeps every weight small so no single feature")
    say("    can dominate. Both are 0.1 in crfsuite's own examples, so that is the")
    say("    centre of the grid.")
    say()
    if args.c1 is not None and args.c2 is not None:
        grid = [(args.c1, args.c2)]
    else:
        grid = [(0.01, 0.01), (0.1, 0.1), (0.1, 0.01), (0.01, 0.1), (0.5, 0.05)]
    say(f"    {'c1':>6} {'c2':>6} {'val P':>8} {'val R':>8} {'val F1':>8} {'sec':>6}")
    best, best_f1 = grid[0], -1.0
    for c1, c2 in grid:
        t1 = time.time()
        crf = fit(X_train, y_train, c1, c2)
        pred = crf.predict(X_val)
        res = nerlib.score(y_val, pred)
        say(f"    {c1:>6} {c2:>6} {res['micro']['precision']:>8.3f} "
            f"{res['micro']['recall']:>8.3f} {res['micro']['f1']:>8.3f} "
            f"{time.time() - t1:>6.0f}")
        if res["micro"]["f1"] > best_f1:
            best, best_f1 = (c1, c2), res["micro"]["f1"]
    say()
    say(f"    chosen c1={best[0]}, c2={best[1]}  (validation micro F1 {best_f1:.3f})")
    say()

    t2 = time.time()
    crf = fit(X_train, y_train, *best)
    train_seconds = time.time() - t2
    predictions = crf.predict(X_test)
    result = nerlib.score(y_test, predictions)

    say("[3] RESULT ON TEST  (entity level, exact span match)")
    nerlib.report_table(result, say)
    say()
    say(f"    micro F1 {result['micro']['f1']:.3f}, "
        f"macro F1 {result['macro']['f1']:.3f}")
    say(f"    final fit took {train_seconds:.0f}s on {len(train):,} sentences")
    say()

    say("[4] WHAT THE MODEL LEARNED  (transition weights)")
    say("    These are learned, not written by us. A large negative weight means")
    say("    the model decided that label sequence is impossible.")
    transitions = sorted(crf.transition_features_.items(),
                         key=lambda kv: kv[1])
    say("    most impossible:")
    for (a, b), weight in transitions[:6]:
        say(f"      {a:<16} -> {b:<16} {weight:>8.2f}")
    say("    most expected:")
    for (a, b), weight in sorted(transitions, key=lambda kv: -kv[1])[:6]:
        say(f"      {a:<16} -> {b:<16} {weight:>8.2f}")
    say()

    say("[5] STRONGEST FEATURES PER LABEL")
    by_label: dict[str, list] = {}
    # sklearn-crfsuite keys this dict (attribute, label), not (label, attribute)
    for (feature, label), weight in crf.state_features_.items():
        by_label.setdefault(label, []).append((weight, feature))
    for label in sorted(by_label):
        if label == "O":
            continue
        top = sorted(by_label[label], reverse=True)[:5]
        say(f"    {label:<16} " + ", ".join(f"{f} ({w:.1f})" for w, f in top))
    say()
    say("    Reading these is the main advantage of a CRF over a neural model: the")
    say("    reason for every decision is a readable list of weighted features.")

    MODELS.mkdir(exist_ok=True)
    with (MODELS / f"{args.tag}.pkl").open("wb") as fh:
        pickle.dump(crf, fh)
    nerlib.save_predictions(args.tag, predictions)
    import json
    (ROOT / "data" / "processed" / f"score_{args.tag}.json").write_text(
        json.dumps(result, indent=1))
    report = (REPORT if args.tag == "crf"
              else ROOT / "reports" / f"04_{args.tag}.txt")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

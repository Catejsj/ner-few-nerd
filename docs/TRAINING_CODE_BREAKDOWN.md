# The training code, explained line by line

For the question "what did you actually run, and why". Three approaches, only
one of which is trained in the usual sense.

| Approach | Is it trained? | What "fitting" means here |
|---|---|---|
| Gazetteer | No | Counting. It builds a dictionary from the training labels |
| CRF | **Yes** | L-BFGS over 595,404 feature-value weights |
| spaCy | No | Already trained, by someone else, on a different corpus |

If you only have time to understand one, understand the CRF — that is the one a
teacher will ask about.

---

# Part 1 — The gazetteer (`scripts/approach_gazetteer.py`)

## What it is

A list of known names and their types, plus a rule for finding them in text.
No weights, no learning, no maths.

## Step 1 — build the dictionary

```python
seen = defaultdict(Counter)
for record in train_records:
    for label, start, end in to_spans(record["bio"]):
        surface = " ".join(record["tokens"][start:end])
        seen[surface][label] += 1
```

For every entity in the **training** data, record its text and its type.
`seen["Washington"]` ends up as `Counter({location: 412, person: 88})`.

**The single most important line is `for record in train_records`.** If that said
"all records" the gazetteer would contain the test answers and the score would be
meaningless. This is the easiest way to accidentally cheat in this project.

## Step 2 — resolve ambiguity

```python
label, count = counter.most_common(1)[0]
gazetteer[surface] = label
```

One surface form, one type — whichever was more common in training. **8,132
surface forms (4.4%) have more than one type**, and for those we are guaranteed
to be wrong some of the time. A dictionary has no context, so there is no better
option available to it. This is reported, not hidden.

## Step 3 — match, longest first

```python
while i < n:
    for length in range(min(MAX_SPAN, n - i), 0, -1):
        phrase = " ".join(tokens[i:i + length])
        if phrase in gazetteer:
            # tag B- then I- I- I-, then jump past the whole entity
            i += length
            break
    else:
        i += 1
```

At each position, try the longest phrase first (8 tokens), then shorter.
`range(8, 0, -1)` counts **down**, and that ordering is the whole trick: matching
shortest-first would tag *New York* inside *New York Times* and stop there.

`MAX_SPAN = 8` because 98.9% of Few-NERD entities are 6 tokens or shorter —
going higher costs time and finds almost nothing.

## The parameter

`min_count` — how often a name must appear in training before it is trusted.
Swept over 1, 2, 3, 5, 10 **on validation**, and 1 won. Higher values raise
precision and cost more recall than they gain.

## Why it scores what it scores

**57.8% of test entity mentions have their exact surface form in the training
data.** That is the hard ceiling on recall before matching even starts — the
other 42.2% are names the dictionary has never seen and cannot invent.

---

# Part 2 — The CRF (`scripts/approach_crf.py`) — the one that is actually trained

## What a CRF is, in one paragraph

An ordinary classifier labels each token independently and can happily output
`I-person` straight after `B-location`. A Conditional Random Field scores the
**whole label sequence at once**, so it learns both how likely each label is for
a token *and* how likely each label is to follow each other label. It then picks
the single best path through all possible label sequences with the Viterbi
algorithm. That is why it was the standard for NER for a decade.

## Step 1 — features: what the model is allowed to see

This is the part that decides how well the model does. A CRF has no embeddings
and no world knowledge; it knows only what these features tell it.

```python
features = {
    "bias": 1.0,                       # always on; learns the base rate
    "word": token,                     # the exact word
    "word.lower": token.lower(),       # word ignoring case
    "word.shape": shape(token),        # Xxxxx, XXX, dddd
    "word.istitle": token.istitle(),   # Starts With A Capital
    "word.isupper": token.isupper(),   # ALL CAPS
    "word.isdigit": token.isdigit(),
    "word.hasdigit": any(c.isdigit() for c in token),
    "word.hashyphen": "-" in token,
    "word.suffix3": token[-3:].lower(),
    "word.prefix3": token[:3].lower(),
    "pos": pos[i],                     # part of speech, from NLTK
    "pos[:2]": pos[i][:2],             # coarse POS: NN, VB, JJ
}
```

Each one earns its place:

- **`word`** — memorises specific names. Powerful on names it has seen, useless
  on new ones.
- **`word.shape`** — generalises. `Xxxxx Xxxxx` is a person pattern whether the
  model has seen that name or not. This is how the CRF beats the gazetteer on
  unseen entities.
- **`istitle` / `isupper`** — capitalisation. Measured in
  `reports/02_preprocessing.txt` §3: **86.8% of entity tokens are capitalised,
  against 7.5% of non-entity tokens.** The strongest single clue in English NER.
- **`suffix3`** — catches morphology the model was never told about: *-ton*,
  *-ville*, *-burg* in places; *-ism*, *-ology* in "other".
- **`pos`** — proper nouns are tagged NNP. A cheap, strong signal.

## Step 2 — the context window

```python
for offset in range(-WINDOW, WINDOW + 1):   # WINDOW = 2
    if offset == 0: continue
    j = i + offset
    features[f"{offset:+d}:word"] = tokens[j]
    features[f"{offset:+d}:word.istitle"] = tokens[j].istitle()
    ...
```

Every token also sees the two tokens on each side. This is what lets the model
learn *"born in ___"* means a location and *"CEO of ___"* means an organization —
the evidence is not in the word itself.

`WINDOW = 2` is the standard choice. Wider windows multiply the feature count
(each extra position adds ~5 features per token) for shrinking gains, and this
model already has millions of features.

`BOS` / `EOS` flags mark sentence start and end, because the first word of a
sentence is capitalised for grammatical reasons rather than because it is a name
— the model needs to know to discount it there.

## Step 3 — fitting

```python
crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=..., c2=...,
    max_iterations=120,
    all_possible_transitions=True)
crf.fit(X_train, y_train)
```

| Setting | What it does |
|---|---|
| `algorithm="lbfgs"` | Limited-memory BFGS, a quasi-Newton optimiser. It approximates the curvature of the loss surface from recent gradients instead of storing a full matrix — which is the only way to handle millions of features on a laptop. The crfsuite default |
| `c1` | **L1 penalty.** Pushes weak weights to exactly zero. Our features are mostly one-off words, so most of them *should* be zero. L1 is what does that |
| `c2` | **L2 penalty.** Keeps all weights small so no single feature can dominate |
| `max_iterations=120` | Cap on optimiser steps. Enough to converge here; a hard stop so a bad setting cannot run forever |
| `all_possible_transitions=True` | Lets the model learn a weight for **every** label pair, including ones never seen in training. This is what teaches it that `B-location -> I-person` is impossible. With `False` it can only score pairs it has observed, and unseen-but-illegal pairs get no penalty at all |

`fit()` maximises the conditional log-likelihood of the gold label sequences,
minus the two penalties. The gradient of that objective is computed with the
forward-backward algorithm — the same dynamic programming as in an HMM.

## Step 4 — tuning c1 and c2

Five combinations, scored on **validation**, best micro F1 wins:

```python
grid = [(0.01, 0.01), (0.1, 0.1), (0.1, 0.01), (0.01, 0.1), (0.5, 0.05)]
```

Centred on `(0.1, 0.1)`, which is crfsuite's own documented example.
Test is used exactly once, after the choice is made. The full grid with its
scores is in `reports/04_crf.txt` §2.

## Step 5 — prediction

```python
predictions = crf.predict(X_test)
```

Viterbi decoding: of all possible label sequences for the sentence, return the
highest-scoring one. Not a per-token argmax — that is the point of a CRF.

## Why we trained on 20,000 sentences and not 131,767

Honest answer, and it should be said plainly rather than hidden: the full
training set with this feature set produces roughly two million distinct
features, and crfsuite holds the whole feature-value matrix in memory while
fitting. It does not fit on this machine.

20,000 sentences still contains thousands of examples of every type. What it
costs is recorded in `reports/04_crf.txt`, and "train on the full set" is the
first item under future work.

## What you can read out of the trained model

Unlike a neural model, a CRF will tell you what it learned:

```python
crf.transition_features_   # {(from_label, to_label): weight}
crf.state_features_        # {(label, feature): weight}
```

`reports/04_crf.txt` §4 and §5 print both. The most negative transitions are the
sequences it decided are impossible; the strongest state features are the words
and shapes that most strongly indicate each type. **Being able to show this is
the CRF's main advantage over a neural model in a presentation.**

---

# Part 3 — spaCy (`scripts/approach_spacy.py`) — no training at all

## What we ran

```python
nlp = spacy.load("en_core_web_sm", exclude=["lemmatizer"])
```

A model trained by spaCy on **OntoNotes 5** (news, phone speech, weblogs),
predicting **18 OntoNotes entity types**. Zero Few-NERD sentences were used to
train it. Our corpus is Wikipedia, so there is a domain shift on top of the label
mismatch, and part of what we are measuring is how far an off-the-shelf model
transfers.

## Problem 1 — tokenization, and the line that fixes it

```python
docs = (Doc(nlp.vocab, words=record["tokens"]) for record in records)
for doc in nlp.pipe(docs, batch_size=256):
    ...
```

`Doc(nlp.vocab, words=...)` builds the document **from our tokens** and never
runs spaCy's tokenizer. Without it, spaCy re-tokenizes, its token boundaries
differ from Few-NERD's, and its predicted spans cannot be lined up with the gold
spans — we would be measuring tokenizer disagreement, not NER.

We measured what that would cost: spaCy's tokenizer produces a different token
sequence from Few-NERD's on **30.7% of sentences** (922 of 3,000 checked), and
every one of those would misalign the labels. `reports/05_spacy.txt` §2.

## Problem 2 — the label mismatch, and how we solved it honestly

OntoNotes has `GPE`, `NORP`, `FAC`, `WORK_OF_ART`. Few-NERD has `location`,
`person`, `building`, `art`, `other`. `PERSON -> person` is obvious. `NORP`
(nationalities, religious and political groups) is not.

We did **not** write the mapping by intuition. For every OntoNotes label we
looked at which Few-NERD type its exactly-matching spans actually coincide with
**on validation**, and mapped it there:

```python
for record, ents in zip(val_records, raw_predictions):
    gold = {(a, b): label for label, a, b in to_spans(record["bio"])}
    for label, a, b in ents:
        if (a, b) in gold:
            agree[label][gold[(a, b)]] += 1
mapping[label] = agree[label].most_common(1)[0][0]
```

The full table, with a purity column showing how consistent each mapping is, is
in `reports/05_spacy.txt` §3.

**The surprise worth putting on a slide:** `FAC` (facilities — airports,
bridges, buildings) maps to **location**, not to `building`, because Few-NERD's
annotators labelled those as locations more often than as buildings. The
intuitive mapping would have been wrong, and only checking against the data
revealed it. As a result **no OntoNotes label maps onto Few-NERD `building`**, so
spaCy scores exactly 0.000 on that type — 5,007 test entities it cannot reach.

## Dropped labels

`CARDINAL, DATE, MONEY, ORDINAL, PERCENT, QUANTITY, TIME` are discarded. Few-NERD
has no type for dates or numbers, so keeping them could only produce false
positives. **This is the mapping decision that most helps spaCy's score**, and it
should be stated openly rather than left for someone to find.

---

# Part 4 — Augmentation (`scripts/augment.py`)

## Why the rubric's examples do not work for NER

| Method | What it breaks |
|---|---|
| Random deletion | Deletes a token, so every label after it shifts by one. Annotation destroyed |
| Random insertion | If the new token lands inside an entity it silently corrupts that span |
| Back-translation | Returns fluent text with completely different token boundaries and no way to map the old labels on |

All three assume the label belongs to the *document*. In NER the label belongs to
a *position*.

## What we used instead: mention replacement

Keep the sentence, swap one entity's tokens for a different entity of the same
type, rebuild the labels around the new length:

```python
new = rng.choice(bank[label])
tokens.extend(new)
bio.extend([f"B-{label}"] + [f"I-{label}"] * (len(new) - 1))
```

Reference: Dai & Adel, *An Analysis of Simple Data Augmentation for Named Entity
Recognition*, COLING 2020.

The mention bank is built from the **full training split** — never validation or
test. That is where the extra information legitimately comes from.

## What it produced

Targeting the three rarest types, 2 copies per eligible sentence:
**20,000 → 29,410 sentences**, and the targeted types (`art`, `event`,
`building`) grew **+200%** each.

## The honest warning

The augmented sentences are often factually absurd — `"the 2020 Battle of the
Philippine Sea football season"`. That is fine for a CRF, which learns surface
patterns and context words, not world knowledge. It would **not** be fine for
fine-tuning a language model.

Whether it actually helped is an empirical question, and the answer is in
`reports/07_augmentation_effect.txt`.

---

# Part 5 — How everything is scored (`scripts/nerlib.py`)

All three approaches go through **one** scoring function. If the gazetteer and
the CRF were scored by different code, any difference between them could be the
scorer rather than the model.

```python
def score(gold_bio, pred_bio):
    for gold, pred in zip(gold_bio, pred_bio):
        g, p = to_spans(gold), to_spans(pred)
        tp += len(g & p)      # exact (type, start, end) match
        fp += len(p - g)
        fn += len(g - p)
```

**Exact span matching.** A predicted span must match the type, the first token
and the last token. A span that overlaps but starts one token early is a false
positive *and* leaves the gold entity as a false negative — punished twice. That
is the CoNLL convention and the strictest reasonable reading. The error analysis
in `reports/06_comparison.txt` §4 separates those near-misses out so the size of
that penalty is visible.

One detail in `to_spans` worth knowing about: a model can emit malformed BIO,
such as `I-person` with no `B-person` before it. We treat that as the start of an
entity rather than dropping it. Dropping malformed predictions would quietly
flatter every model by deleting some of its own false positives.

---

# The five things to remember

1. The gazetteer is built from **training only**, and 57.8% coverage of test
   mentions is its ceiling.
2. The CRF's power is **context features plus learned transitions**; its weights
   are readable, which no neural model gives you.
3. `all_possible_transitions=True` is what teaches it that `B-location ->
   I-person` is impossible.
4. spaCy needs `Doc(words=...)` or the whole comparison is invalid.
5. The label mapping was **learned from validation**, and it revealed that
   Few-NERD calls facilities *locations*.

# Named Entity Recognition — three approaches on Few-NERD

Text Mining project. Three NER approaches — a **gazetteer**, a **CRF**, and
**spaCy's pre-trained model** — built on the same corpus and scored by the same
function, with entity-level exact span matching.

Dataset: [Few-NERD](https://github.com/thunlp/Few-NERD) (Ding et al.,
ACL-IJCNLP 2021), supervised split, CC BY-SA 4.0. 188,238 Wikipedia sentences,
4.6M tokens, 485,763 entities, 8 coarse entity types.

## Research questions

1. How do the three classic approaches compare on one corpus under identical
   entity-level scoring?
2. Do they fail on the *same* entities, or on different ones?

## Setup

```bash
python3.11 -m venv .venv
./.venv/bin/pip install pandas pyarrow numpy scikit-learn sklearn-crfsuite seqeval matplotlib spacy nltk
./.venv/bin/python -m spacy download en_core_web_sm
./.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4','averaged_perceptron_tagger_eng']]"
```

Download the data (≈24 MB, three parquet files):

```bash
mkdir -p data/raw && for s in train validation test; do curl -sL -o "data/raw/supervised_$s.parquet" "https://huggingface.co/datasets/DFKI-SLT/few-nerd/resolve/main/supervised/$s-00000-of-00001.parquet"; done
```

## Pipeline — run in this order

| # | Command | Produces |
|---|---|---|
| 1 | `./.venv/bin/python scripts/build_corpus.py` | BIO jsonl, `reports/01_corpus.txt` |
| 2 | `./.venv/bin/python scripts/preprocess.py` | cleaned jsonl + CRF subsample, `reports/02_preprocessing.txt` |
| 3 | `./.venv/bin/python scripts/augment.py` | `train_small_aug.jsonl`, `reports/03_augmentation.txt` |
| 4 | `cd scripts && ../.venv/bin/python approach_gazetteer.py` | `reports/03_gazetteer.txt` |
| 5 | `cd scripts && ../.venv/bin/python approach_crf.py` | `models/crf.pkl`, `reports/04_crf.txt` |
| 6 | `cd scripts && ../.venv/bin/python approach_spacy.py` | `reports/05_spacy.txt` |
| 7 | `cd scripts && ../.venv/bin/python evaluate.py` | `reports/06_comparison.txt` |
| 8 | `cd scripts && ../.venv/bin/python make_figures.py` | `figures/fig1-4*.png` |
| 9 | `cd scripts && ../.venv/bin/python make_figure_crf.py` | `figures/fig5_crf_features.png` |

The approach scripts import `nerlib.py` as a sibling module, so run them from
inside `scripts/`. Step 5 takes about 10 minutes; everything else is seconds to
a couple of minutes.

To measure whether augmentation helped — same features, same tuning grid, same
test set, so the training data is the only difference:

```bash
cd scripts && ../.venv/bin/python approach_crf.py --train-file train_small_aug --tag crf_aug
```

```bash
cd scripts && ../.venv/bin/python augmentation_effect.py
```

## Results

Test set, entity level, exact span matching:

| approach | micro P | micro R | micro F1 | macro F1 |
|---|---|---|---|---|
| Gazetteer | 0.266 | 0.477 | 0.342 | 0.324 |
| **CRF** | 0.665 | 0.635 | **0.650** | **0.580** |
| spaCy (pre-trained, zero-shot) | 0.419 | 0.390 | 0.404 | 0.233 |

The three do not fail on the same entities. Together they get **78.0%** of gold
entities right against **63.5%** for the CRF alone, and all three agree on only
**21.6%**.

## Design decisions worth knowing

**Entity level, exact span matching, never token accuracy.** Only 21.1% of
tokens are inside an entity, so a model predicting "O" everywhere scores 78.9%
token accuracy and finds nothing. A predicted span that overlaps a gold entity
but starts one token early is a false positive *and* leaves the gold entity as a
false negative — punished twice, the CoNLL convention.

**8 coarse types, not Few-NERD's 66.** The rubric requires a pre-trained spaCy
model as one of the three approaches, and spaCy predicts 18 OntoNotes types — a
66-way comparison is impossible for one of the three. The fine labels are kept in
the raw files, so the choice is reversible.

**Most standard preprocessing is forbidden here, and we measured why.**
Lowercasing would delete the signal that 86.8% of entity tokens are capitalised
against 7.5% of other tokens. Stopword removal would break the 7.8% of entities
that contain one (*University **of** California*). Stemming would make the output
not a span of the original sentence. Lemmatization is used as a CRF *feature*
instead.

**The gazetteer is built from the training split only.** Building it from all the
data would put the test answers in the dictionary. Its ceiling is 57.8% — that is
the share of test entity mentions whose exact surface form appears anywhere in
training.

**spaCy's label mapping was learned from validation, not guessed.** The finding:
OntoNotes `FAC` maps to Few-NERD **location**, not *building*, so no OntoNotes
label reaches `building` and spaCy scores 0.000 on it.

**Augmentation is mention replacement**, not the word-level methods in the
rubric — deletion and insertion shift positional labels and destroy the
annotation.

## Layout

```
data/raw/          the three Few-NERD parquet files
data/processed/    BIO jsonl, the CRF subsample, predictions, scores
models/            the fitted CRF, the gazetteer, the spaCy label map
reports/           numbered plain-text reports, one per stage
figures/           the four slide figures
docs/              presentation plan, parameter Q&A, code walkthrough
scripts/           the pipeline
```

## Documents

- **`docs/PRESENTATION_PLAN.md`** — the whole presentation split one part per
  rubric criterion, with each person's slide bullets, script and sourced numbers.
- **`docs/SLIDES_MODEL_TRAINING.md`** — the Model Design & Training part (rubric
  criterion 5) in full: three slides, exact bullets for the slide maker, a spoken
  script, the figure explained, Q&A prep, and what belongs to other speakers.
- **`docs/TRAINING_CODE_BREAKDOWN.md`** — what each approach actually runs, line
  by line, and why each parameter has the value it has.
- **`docs/ANNOTATION_AND_IAA.md`** — why we used an existing dataset, the
  original authors' annotation process, their Cohen's Kappa of 0.7644, and how
  entity labels are represented.
- **`docs/PARAMETERS_AND_WHY.md`** — every parameter in the project as a spoken
  answer, for question prep.

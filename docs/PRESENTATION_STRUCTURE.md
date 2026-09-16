# Presentation structure

Maps the rubric's 8 criteria onto 7 parts, each backed by a specific
script/report/figure in this repo. Numbers are from the generated reports
(`reports/01_corpus.txt` … `reports/07_augmentation_effect.txt`) — re-check them
if the pipeline is re-run before the talk, since a re-run can shift them
slightly.

Total: ~20-25 min, 3-4 min per person + buffer. Order follows the pipeline's
data flow: corpus → annotation → preprocessing → models → evaluation →
limitations.

---

## Person 1 — Problem Definition & Motivation (rubric: 5 pts)

No script — pure framing, sets up the terminology every later part depends on.

- **RQ1**: how do the three classic NER approaches — gazetteer, CRF, pre-trained
  model — compare on one corpus under one scoring function? Supervised, with
  gold spans, so unlike topic modelling there is a right answer to score against.
- **RQ2**: do the three fail on the *same* entities, or on different ones? This
  is the question that motivates reporting all three rather than picking a
  winner; answered in Person 6's section.
- **Task definition**: NER is sequence labelling — a label per token, and the
  answer is a *span* of the original sentence. This is the fact that constrains
  Person 4's preprocessing options and forces Person 6's entity-level scoring.
- **Ambiguity is contextual, not lexical**: Few-NERD's own example, *"London is
  the fifth album by the British rock band Jesus Jones"* — `London` is `art`, not
  `location`. A dictionary cannot resolve this by construction (Person 5), which
  is the motivating gap for the CRF.
- **Scope**: 8 coarse types, not Few-NERD's 66. Forced by the rubric — spaCy's
  pre-trained model emits 18 OntoNotes types and cannot express 66, so a 66-way
  comparison is impossible for one of the three required approaches. Fine labels
  are retained in `data/raw/`, so the decision is reversible. Revisited in
  Person 7's section.
- **Unit of analysis**: one sentence = one sequence; one entity = one scored
  item. Scoring is never per token — see Person 2's 21.1% figure for why.

**Slides:** 1 — task definition and the two RQs, no output to show yet.

---

## Person 2 — Corpus Collection & Legal/Ethical Compliance (rubric: 10 pts)

**Source:** `scripts/build_corpus.py` → `reports/01_corpus.txt`.

**Dataset**: Few-NERD (Ding et al., ACL-IJCNLP 2021), **supervised** setting,
obtained as parquet from the HuggingFace mirror `DFKI-SLT/few-nerd`. Few-NERD
also ships `intra`/`inter` splits that hold out whole entity types for few-shot
work; those are not used, because RQ1 is ordinary supervised NER.

**Size**:

| split | sentences | tokens | entities |
|---|---|---|---|
| train | 131,767 | 3,227,563 | 340,180 |
| validation | 18,824 | 463,214 | 48,741 |
| test | 37,648 | 921,118 | 96,842 |
| **total** | **188,239** | **4,611,895** | **485,763** |

Mean sentence length 24.5 tokens. The published split is used unmodified — no
re-splitting, because Wikipedia sentences from one article landing on both sides
would leak entity surface forms into test and inflate Person 5's gazetteer in
particular.

**Class balance** (training entities): location 28.0%, person 22.3%,
organization 19.7%, other 9.9%, product 6.4%, building 5.2%, art 4.4%, event
4.1%. Largest-to-smallest ratio **6.8:1** — mild for NER, and every type has
>14,000 training examples, so no type is starved.

**The imbalance that actually matters** is not between types: **680,543 of
3,227,563 training tokens (21.1%) are inside an entity**. A model predicting `O`
everywhere scores 78.9% token accuracy and finds zero entities. This is the
reason Person 6 reports entity-level P/R/F1 and never token accuracy, and it
should be stated here rather than there.

**Entity length**: 43.2% single-token, 33.1% two-token, 98.9% ≤6 tokens. Drives
Person 5's `MAX_SPAN=8` gazetteer cap and Person 6's decision that boundary
errors need their own error category.

**Legal/ethical compliance** (10 of 100 rubric points):

- License: CC BY-SA 4.0 — permits reuse, modification and redistribution with
  attribution and share-alike. Paper cited on Person 7's references slide.
- Provenance: English Wikipedia dump, itself CC BY-SA 4.0. No scraping, no ToS
  question.
- Personal data: the corpus contains person names *by design* — it is a named
  entity dataset. Those names are public figures already documented on Wikipedia,
  appearing in the form Wikipedia published them.
- Anonymization is **not** applied, deliberately: anonymizing a NER corpus
  destroys the labels being predicted. Mitigations instead — no cross-sentence
  entity linking, no profile construction, no information added that was not in
  the source.
- Content sensitivity: entities include victims and offenders of real crimes.
  Results are reported only at the type level; no individual is presented as a
  finding.

**Slides:** 2 — size/split table + class balance with the 21.1% point;
legal/ethical bullet list.

---

## Person 3 — Annotation Quality & Inter-Annotator Agreement (rubric: 10 pts)

**Source:** `docs/ANNOTATION_AND_IAA.md` (Ding et al. 2021, §4).

**Route taken**: existing labelled dataset, not group annotation. The rubric
marks group annotation "not recommended in this class" for this task and asks
instead for a report on the original authors' guideline and agreement. Rationale
worth stating: NER annotation is a decision per token plus two boundary
decisions per entity, against an 8-way (here) or 66-way (in the source) type
schema — not one label per document.

**The original process**:

- **70 annotators + 10 experienced experts**, all required to have linguistic
  knowledge and instructed with formal annotation principles. Mean 32 hours per
  annotator; annotators compensated at market rate by workload (stated in the
  paper — relevant to Person 2's ethics section).
- **Two independent annotators per paragraph**, then an **expert adjudicates**
  and makes the final decision. Disagreements are resolved by adjudication, not
  majority vote.
- Batches of 1,000-3,000 sentences; **10% randomly double-checked** per batch;
  **below 95% sentence-level accuracy the whole batch is re-annotated**.
- **Cohen's Kappa = 0.7644** — *substantial* on Landis & Koch, the second-highest
  band. Note for context: span tasks lose kappa to boundary disagreements that
  are not semantic disagreements (*"the University of Toronto"* vs *"University
  of Toronto"*), so 0.76 is strong for NER specifically.
- **Guideline provenance**: schema adapted from FIGER (Ling & Weld 2012, 112
  tags), then refined across rounds of pre-annotation — `Country`,
  `Province/State`, `City`, `Restrict` merged into `GPE` because annotators could
  not separate them from context alone; `Person-Scholar` added because
  pre-annotation surfaced entities the schema could not express. The refinement
  loop is the part worth highlighting.
- Annotation done at **paragraph level** (66,000 paragraphs, mean 61.3 tokens) so
  annotators see the context needed to resolve the `London`-the-album case from
  Person 1.

**Label representation** (the rubric asks this explicitly):

Few-NERD ships **IO** tags — one label per token, no prefix; an entity is a run
of same-tagged tokens. Converted to **BIO** in `build_corpus.py::to_bio()`:

```
tokens     Barack    Obama     visited   New         York
Few-NERD   person    person    O         location    location
BIO        B-person  I-person  O         B-location  I-location
```

BIO is what sklearn-crfsuite, seqeval and spaCy all consume, so Person 5's three
approaches stay directly comparable. BIOES was not used: it doubles the label
count (33 vs 17) on a machine already memory-bound in Person 5's CRF, and
Few-NERD's published baselines use BIO.

**Cost of the conversion** (state it, do not wait to be asked): two same-type
entities that touch with no intervening token are already merged in the released
IO file. Unrecoverable, and unmeasurable — measuring it requires the information
that is missing. Bounded by entity length: 43.2% of entities are one token.
Revisited in Person 7's section.

**Slides:** 2 — annotation procedure + κ = 0.7644; BIO worked example with the
IO→BIO cost.

---

## Person 4 — Text Preprocessing & Augmentation (rubric: 15 pts)

**Source:** `scripts/preprocess.py` → `reports/02_preprocessing.txt`;
`scripts/augment.py` → `reports/03_augmentation.txt`;
`scripts/augmentation_effect.py` → `reports/07_augmentation_effect.txt`.

This section has an inverted shape and that is the point: **for sequence
labelling most standard preprocessing is forbidden**, and each refusal is backed
by a measurement rather than an assertion. Labels are positional — deleting
token 3 shifts every label after it.

1. **Tokenization** — not re-done. Few-NERD ships pre-tokenized sentences and the
   labels are aligned to those exact tokens; re-tokenizing shifts every label by
   an unknown amount. Few-NERD's own tokenization is Penn Treebank style, so
   clitics split (`"People 's Republic of China"`). Consequence for Person 5's
   spaCy approach: spaCy runs its own tokenizer and disagrees on **30.7%** of
   sentences, handled there with `Doc(words=...)`.
2. **Normalization** (applied) — Unicode NFKC, curly quotes/dashes folded to
   ASCII, intra-token whitespace stripped. Per token, so the token count never
   changes. 1,280 tokens altered per 20,000 sentences.
3. **Lowercasing** — **not applied**. **86.8% of entity tokens are capitalized
   against 7.5% of non-entity tokens.** Capitalization is the strongest surface
   cue English NER has; lowercasing deletes it. Preserved in the text and given
   to the models as explicit `istitle`/`isupper`/`shape` features (Person 5).
4. **Stopword removal** — **not applied**. **26,375 of 340,180 entities (7.8%)
   contain a stopword** — *University **of** California*, ***The** New York
   Times*, *World War **I***. Removal breaks every one of them and destroys the
   token-label alignment for the whole sentence. Stopword removal is a
   document-classification technique; it does not transfer to sequence labelling.
5. **Stemming** — **not applied** to the sequence: the output of NER must be a
   span of the original sentence, and *"Univers of Californ"* is not a span of
   anything. Porter comparison retained for the report (`Games→game`,
   `Studies→studi`, `Airlines→airlin`).
6. **Lemmatization** — applied as a **feature only** (`word.lemma` in
   `approach_crf.py::token_features`), which takes the generalization
   (*Islands*/*Island* behave alike inside a location name) without damaging the
   text.
7. **CRF subsample** — 20,000 of 131,767 training sentences, seed 42. Memory, not
   design: 20,000 sentences already produce **595,404 distinct feature-value
   pairs** and CRFsuite holds the feature matrix in memory while fitting. The
   gazetteer and spaCy use the full data. Bounds Person 5's CRF number as a floor,
   not a ceiling; revisited in Person 7's section.

**Data augmentation** — **mention replacement**, not the word-level methods the
rubric lists:

- Rubric methods and why they are unusable here: random deletion shifts every
  subsequent label; random insertion can land inside an entity and corrupt its
  span; back-translation returns different token boundaries with no mapping for
  the old labels. All three assume the label belongs to the *document*.
- Method used: keep the sentence, swap one entity's tokens for another entity of
  the same type drawn from a bank, rebuild BIO around the new length. Reference:
  Dai & Adel, COLING 2020. Bank built from the **full training split** — never
  validation or test.
- Scope: 3 rarest types (`art`, `event`, `building`), 2 copies per eligible
  sentence → **20,000 → 29,410 sentences**, targeted types **+200%** each.
- Known artifact, state it: outputs are frequently implausible (*"the 2020 Battle
  of the Philippine Sea football season"*). Acceptable for a feature-based CRF,
  which learns surface patterns and context words; would **not** be acceptable
  for fine-tuning a language model.
- **Measured effect** (`reports/07_augmentation_effect.txt`) — same features,
  same c1/c2 grid, same test set, training file the only difference:

| | micro F1 | macro F1 |
|---|---|---|
| baseline (20,000) | 0.650 | 0.580 |
| augmented (29,410) | 0.645 | 0.573 |

  It **cost the targeted types most** (mean −0.016) and barely touched the rest
  (−0.003) — the opposite of its purpose. Mechanism is visible once F1 is split:
  on targeted types **precision −0.099, recall +0.068**. Mention replacement
  multiplies entity surface forms but leaves the surrounding context unchanged,
  so the model meets identical sentence frames repeatedly with different names in
  the slot, over-weights the frame, and predicts those types too freely. Reported
  as a negative result with a mechanism, not omitted.

**Slides:** 2 — the not-applied table with its measurements (86.8/7.5, 7.8%);
mention replacement worked example + the measured null result.

---

## Person 5 — Model Design & Training (rubric: 25 pts)

**Source:** `scripts/approach_gazetteer.py` → `reports/03_gazetteer.txt`;
`scripts/approach_crf.py` → `reports/04_crf.txt`; `scripts/approach_spacy.py` →
`reports/05_spacy.txt`; `scripts/make_figure_crf.py` →
`figures/fig5_crf_features.png`. Full speaker document:
`docs/SLIDES_MODEL_TRAINING.md`.

All three required approaches, differing in kind — dictionary, trained
sequence model, transferred model. Scores belong to Person 6; this section ends
on a hand-off, not a winner.

**Gazetteer (required)** — surface form → most frequent training type, longest-
match scanning left to right, no overlaps, up to `MAX_SPAN=8` tokens (98.9% of
entities are ≤6, per Person 2).

- Built from the **training split only**. Building it from all data would place
  the test answers inside the dictionary — the most direct available form of
  leakage in this project.
- **185,158** entries. `min_count` swept 1/2/3/5/10 on **validation**, 1 selected
  (micro F1 0.345); higher values raise precision but lose recall faster.
- **8,125 entries (4.4%) are ambiguous** — *Washington* person vs location,
  *American* location vs person. Most-frequent type wins, so each is a guaranteed
  error whenever the other reading is correct. Structural: a dictionary has no
  context.
- **Ceiling: 57.8%** of test entity mentions have their exact surface form
  anywhere in training. 42.2% of recall is unreachable before matching starts.
- Counter-intuitive result worth volunteering: **recall exceeds precision**
  (0.477 vs 0.266), which is not classic dictionary behaviour. Cause is that a
  gazetteer induced from real text absorbs ordinary words that are also titles —
  *Today*, *Why*, *Hello* are all entered as `art` and then match every ordinary
  use. Produces **92,139 spurious predictions against 96,842 real entities**.

**CRF (required, the only trained model)** — `sklearn-crfsuite`, L-BFGS.

- Scores the **whole label sequence** and decodes with Viterbi, so it learns
  transition structure a per-token classifier cannot express. Learned evidence:
  `O → I-organization` has weight **−7.58**; `B-event → I-event` **+6.70**.
- **34 features per token**, 20 of them from the ±2 context window: `word`,
  `word.lower`, `word.lemma`, `word.shape` (`Xxx`), `istitle`, `isupper`,
  `isdigit`/`hasdigit`, `hashyphen`, 3-char prefix/suffix, POS and coarse POS,
  plus word/shape/istitle/POS for each of the four neighbours; `BOS`/`EOS` flags
  so sentence-initial capitalization is discounted. `word.shape` is what
  generalizes to unseen names — the gazetteer's ceiling does not bind here.
- `all_possible_transitions=True`: with `False` the model can only score label
  pairs observed in training, so unseen-but-illegal pairs carry no penalty at
  all. This setting is what produces the −7.58 above.
- `c1`/`c2` (L1/L2) tuned over 5 combinations on **validation**, centred on
  CRFsuite's documented (0.1, 0.1); **0.1/0.1** selected at validation micro F1
  0.649. Spread across the grid ~0.015 F1, so the result does not hinge on it.
  `max_iterations=120`. Final fit 91 s on 20,000 sentences.
- Feature weights are readable (`reports/04_crf.txt` §5) — e.g.
  `word.shape:XxXxxx → B-person`, `-1:word.lower:album → B-art`,
  `word.lemma:airport → I-building`. This interpretability is the CRF's argument
  against a neural model in a presentation.

**spaCy `en_core_web_sm` (required, pre-trained)** — trained on OntoNotes 5
(news, telephone speech, weblogs), **zero** Few-NERD sentences. Two structural
problems, both solved before comparison:

1. **Tokenization**: documents built with `Doc(nlp.vocab, words=gold_tokens)`,
   bypassing spaCy's tokenizer entirely. Without it, token sequences differ on
   **30.7%** of sentences (922 of 3,000 checked) and the score would measure
   tokenizer disagreement, not NER.
2. **Label mismatch**: 18 OntoNotes types vs our 8. The mapping is **learned from
   validation**, not written from intuition — for each OntoNotes label, the
   Few-NERD type its exactly-matching spans coincide with, with a purity column:
   `GPE→location` 83.8%, `EVENT→event` 83.3%, `PERSON→person` 72.8`%`,
   `ORG→organization` 43.9%, `PRODUCT→product` 30.1%, `LAW→other` 26.9%. Low
   purity means a compromise, and is shown as such.
   - Seven numeric labels (`CARDINAL, DATE, MONEY, ORDINAL, PERCENT, QUANTITY,
     TIME`) are **dropped**: Few-NERD deliberately excludes value/numerical/time
     entities, so predicting them can only generate false positives. This is the
     mapping decision that most helps spaCy's score and should be stated openly.
   - **Finding**: `FAC` (facilities — airports, bridges) maps to **`location`**,
     not `building`, at 47.7% purity — Few-NERD's annotators treat facilities as
     places. Consequence: **no OntoNotes label reaches `building`**, so spaCy
     scores exactly **0.000** on it, 5,007 test entities it cannot reach. The
     intuitive mapping would have been wrong; only checking against data exposed
     it.

**Slides:** 2 — gazetteer and spaCy as a two-column slide (untrained
approaches, with the `FAC` finding); CRF full width with
`fig5_crf_features.png`.

---

## Person 6 — Evaluation & Error Analysis (rubric: 25 pts)

**Source:** `scripts/nerlib.py::score` (one scorer for all three approaches),
`scripts/evaluate.py` → `reports/06_comparison.txt`, `data/processed/confusion_*.csv`;
`scripts/make_figures.py` → `figures/fig1`-`fig4`.

**Scoring protocol** (the rubric asks for this to be specified explicitly):

- **Entity level, EXACT span matching** — type, start token and end token must
  all match. A span overlapping a gold entity but starting one token early is a
  false positive **and** leaves the gold entity a false negative: penalized
  twice. CoNLL convention, and the strictest reasonable reading.
- One scoring function for all three approaches, so any difference between them
  is the model rather than the measurement.
- Malformed predictions (`I-person` with no preceding `B-`) are treated as entity
  starts, not discarded — discarding them would silently delete each model's own
  false positives.

**Headline** (test set):

| approach | micro P | micro R | micro F1 | 95% CI | macro F1 |
|---|---|---|---|---|---|
| Gazetteer | 0.266 | 0.477 | 0.342 | [0.339, 0.345] | 0.324 |
| **CRF** | 0.665 | 0.635 | **0.650** | [0.646, 0.653] | **0.580** |
| spaCy | 0.419 | 0.390 | 0.404 | [0.400, 0.408] | 0.233 |

CIs are 1,000-resample bootstraps over sentences; narrow because the test set is
37,648 sentences, so the ordering is not noise.

**micro vs macro**: micro averages over entities and is dominated by location,
person and organization (69% of test entities); macro averages over the 8 types
equally and penalizes ignoring rare ones. spaCy's gap (0.404 → 0.233) is the
largest — it transfers on the common types and collapses on the rest.

**Per-type F1**: the CRF leads on every type, but by margins from 0.16
(location) to 0.53 (art). spaCy's `building` = 0.000 is structural, not a failure
of the model (Person 5).

**Error analysis** — every gold entity and every prediction assigned one MUC
category:

| of 96,842 gold entities | Gazetteer | CRF | spaCy |
|---|---|---|---|
| correct | 47.7% | 63.5% | 39.0% |
| type error (exact span, wrong type) | 5.8% | 14.6% | 20.2% |
| boundary error (overlap, right type) | 17.0% | 6.8% | 12.0% |
| type + boundary | 13.8% | 5.1% | 12.9% |
| missed (nothing predicted) | 15.7% | 9.9% | 16.0% |
| spurious (predicted where nothing is) | **95.1%** | 6.9% | 10.3% |

The distinction carries the analysis: a boundary error means the model located
the entity and mis-cut it — a feature problem; a missed entity means no evidence
at all. For the CRF, **mislabelled-but-found (14.6%) exceeds missed (9.9%)**,
which is the more tractable failure mode. The gazetteer's 95.1% spurious rate is
the ordinary-words effect from Person 5 quantified.

**Confusion** (`fig4_confusion.png`, row-normalized — each row is where that
true type ended up): worst pair for all three approaches is
**organization ↔ location** (CRF: 1,741 org→loc, 1,296 loc→org), consistent with
metonymy in the source text. spaCy additionally collapses `building` and `other`
into `organization`.

**RQ2 — do they fail on the same entities?** No:

- CRF correctly finds 61,490, of which **15,026** no other approach finds.
- Gazetteer 46,187, of which **8,648** are unique to it.
- spaCy 37,789, of which **2,783** are unique.
- **Union: 78.0%** of gold entities; **all three agree on only 21.6%**.

A perfect oracle over the three reaches 78.0% recall against 63.5% for the best
single approach. This is the quantitative argument for Person 7's ensemble
future-work item.

**Slides:** 3 — scoring protocol + `fig1_headline_f1.png`;
`fig2_per_type_f1.png`; `fig3_error_breakdown.png` + `fig4_confusion.png` with
the union/agreement numbers.

---

## Person 7 — Limitations, Future Studies, and Presentation Synthesis (rubric: Limitation 5 pts + Presentation & Report Clarity 5 pts)

**Source:** all reports; `figures/`.

**Limitations:**

- **CRF training-set truncation (Person 4)**: 20,000 of 131,767 sentences, a
  memory bound not a design choice. The reported 0.650 is a floor; the gap to the
  full-data number is unmeasured. Largest single lever for a follow-up.
- **8 coarse types, not 66 (Person 1)**: forced by spaCy's fixed 18-type output.
  The fine-grained task Few-NERD was built for is untouched, and the two
  approaches that *could* express 66 types were held back to keep the comparison
  three-way.
- **IO→BIO conversion (Person 3)**: adjacent same-type entities are already
  merged in the released data. Not recoverable, and not measurable — quantifying
  it requires exactly the information that is missing.
- **spaCy is scored off-task (Person 5)**: ~0.85 F1 on OntoNotes text with
  OntoNotes labels; 0.404 here. The gap measures domain plus label-scheme
  transfer, not model quality, and should not be presented as a verdict on spaCy.
- **The learned label mapping is fitted to this corpus (Person 5)**: derived from
  Few-NERD validation data, so it would not transfer to another target schema.
- **Single domain**: Wikipedia prose — well-formed, well-capitalized English.
  Since 86.8% capitalization is the strongest CRF feature (Person 4), none of
  these results transfer to lowercase text, tweets or speech transcripts.
- **Augmentation negative result (Person 4)**: mention replacement was measured,
  not assumed, and cost 0.005 micro F1. The conclusion is specific — adding
  entity variety without context variety does not help a context-driven model —
  not the general claim that augmentation fails for NER.

**Future studies** (paired with the limitation each addresses):

- Fit the CRF on all 131,767 sentences on hardware with the memory for it, and
  report the delta against the 20,000-sentence floor.
- **Gazetteer membership as a CRF feature** — a hybrid of approaches 1 and 2,
  near-zero cost, and directly targeted at the 8,648 entities the gazetteer finds
  that the CRF misses.
- **Ensemble the three**, motivated by Person 6's 78.0% union against 63.5%
  single-best; the 21.6% three-way agreement rate indicates the headroom is real.
- Context-aware augmentation (paraphrase the sentence frame, not only the entity)
  to test the mechanism Person 4's negative result identified.
- Fine-tune a transformer (BERT/RoBERTa) as the fourth approach the rubric marks
  optional, and re-run at 66 fine-grained types for the two approaches that can
  express them.

**Presentation & Report Clarity** — cross-checking the other six parts for
consistency:

- Terminology: "entity" is a labelled span, never a token; "mention" is a surface
  form; "type" is one of the 8 classes. "Exact span match" stated once by Person 6
  and not redefined elsewhere. Nobody says "accuracy" — the reason is Person 2's
  21.1%.
- Figure ownership: `fig1`-`fig4` are Person 6's, `fig5_crf_features.png` is
  Person 5's. No figure appears twice.
- Every number traces to a report file and section (this document's citations are
  the map for that check). Entity counts, split sizes and F1 values must match
  across every slide citing them.
- Q&A prep consolidated from `docs/PARAMETERS_AND_WHY.md` — the three points most
  likely to be interrogated are the 20,000-sentence subsample, the numeric-label
  drop that flatters spaCy, and the augmentation negative result.

**Slides:** 2 — limitations/future-work pairs; closing summary, references and
Q&A transition.

---

## Timing notes

- Persons 5 and 6 carry the 25-point criteria — most rehearsal time and Q&A prep
  belongs there.
- Person 3's annotation section is short in slide count but is 10 rubric points,
  and it is entirely secondary reporting (nobody in the group annotated
  anything), so it needs deliberate pacing rather than a speed-run.
- Person 4 inverts the usual shape — the section is mostly about what was *not*
  applied. Lead with the measurement (86.8% vs 7.5%), not with the refusal, or it
  sounds like work that was skipped.
- Hand-offs follow the pipeline: Person 2 → 3 (who labelled these 485,763
  entities, and how well), Person 3 → 4 (BIO tags are what the pipeline now
  consumes), Person 4 → 5 (20,000 sentences and 595,404 feature-value pairs go
  into the CRF), Person 5 → 6 (three trained/loaded systems, one scorer),
  Person 6 → 7 (what the 78.0% union leaves open).

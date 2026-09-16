# Presentation plan — one part per rubric criterion

The NER rubric has **eight** criteria. This plan makes **seven parts** by giving
the last person both "Limitations" and "Presentation & Report Clarity", since
clarity is not presentable content — it is earned by owning the deck.

If your group has eight people, split part 5 (Model Design, 25 points) into
two: one person takes the Gazetteer and the CRF, the other takes spaCy and the
comparison. If it has six, fold part 1 into part 7's introduction.

| # | Person | Rubric criterion | Points | Slides | Time |
|---|---|---|---|---|---|
| 1 | | Problem Definition & Motivation | 5 | 1 | 1 min |
| 2 | | Corpus Collection & Legal / Ethical | 10 | 2 | 2 min |
| 3 | | Annotation Quality & Inter-Annotator Agreement | 10 | 2 | 2 min |
| 4 | | Text Preprocessing & Augmentation | 15 | 2 | 2.5 min |
| 5 | | Model Design & Training — all three approaches | 25 | 3 | 4 min |
| 6 | | Evaluation & Error Analysis | 25 | 3 | 4 min |
| 7 | | Limitations, Future Work, Conclusion + deck owner | 10 | 2 | 2 min |

**15 slides, about 18 minutes.** Parts 5 and 6 carry half the marks between
them — give those to the two most confident speakers.

Every number below has a report file next to it. If a number is not in this
document, do not say it.

---

# Part 1 — Problem Definition & Motivation (5 pts)

### Slide: "What we are doing"

- **Task:** Named Entity Recognition — find every name in a sentence and say
  what kind of thing it is
- **RQ1:** how well do the three classic approaches — dictionary, CRF, and a
  pre-trained model — do on the same corpus, scored identically?
- **RQ2:** where does each one fail, and do they fail on the *same* entities?
- **Why it matters:** NER is the first step in search, question answering and
  information extraction. If the entity is missed, everything downstream fails
- **Corpus:** Few-NERD — 188,238 Wikipedia sentences, 8 entity types

### Say (about 60 seconds)

> Named entity recognition is the task of finding the names in a sentence and
> saying what they are. In "Barack Obama visited New York", a NER system has to
> return Barack Obama as a person and New York as a location.
>
> It is harder than it sounds, for two reasons. The system has to find the exact
> boundary — "the University of Toronto" is not the same span as "University of
> Toronto" — and the same string can be different things in different sentences.
> The Few-NERD paper's own example is the word "London": in "London is the fifth
> album by the British rock band Jesus Jones", London is a work of art, not a
> place.
>
> Our question is not "what is the best NER system" — that is settled, it is a
> large neural model. Our question is what the three classic approaches actually
> do on the same corpus with the same scoring: a dictionary, a CRF, and an
> off-the-shelf pre-trained model. And more interestingly, whether they fail on
> the same entities or on different ones.

---

# Part 2 — Corpus Collection & Legal / Ethical (10 pts)

**Source:** `reports/01_corpus.txt`

### Slide A: "Few-NERD"

- **Few-NERD** (Ding et al., ACL-IJCNLP 2021) — the largest human-annotated NER
  dataset
- **188,238 sentences**, **4.6 million tokens**, **485,763 entities**
- Text is **English Wikipedia**; licence **CC BY-SA 4.0**
- We use the **supervised** split: train 131,767 / validation 18,824 / test 37,648
- We did **not** re-split — the published split is used as published

### Slide B: "Balance, and the imbalance that matters"

| type | train entities | share |
|---|---|---|
| location | 95,339 | 28.0% |
| person | 75,945 | 22.3% |
| organization | 66,920 | 19.7% |
| other | 33,611 | 9.9% |
| product | 21,835 | 6.4% |
| building | 17,599 | 5.2% |
| art | 14,870 | 4.4% |
| event | 14,061 | 4.1% |

- Type imbalance is only **6.8 to 1** — mild for NER
- **The real imbalance: only 21.1% of tokens are inside an entity at all**
- A model predicting "O" everywhere is **78.9% accurate** and finds nothing →
  that is why we never report token accuracy

### Say (about 2 minutes)

> Our corpus is Few-NERD, published at ACL 2021 and built from English
> Wikipedia. It is 188,000 sentences, 4.6 million words, and 485,000 labelled
> entities, which makes it the largest human-annotated NER dataset available.
>
> On the legal side: Few-NERD is released under Creative Commons BY-SA 4.0, so we
> may use, modify and redistribute it as long as we credit the authors, which we
> do on the references slide. The underlying text is Wikipedia, also CC BY-SA, so
> there is no scraping and no terms of service issue.
>
> On personal data, we want to be direct rather than pretend the issue does not
> exist. This corpus contains people's names — that is the point of a NER
> dataset. Those names are public figures already documented on Wikipedia, not
> private individuals, and they appear exactly as Wikipedia published them. We do
> not anonymise, because anonymising a NER corpus would delete the labels we are
> trying to predict. What we do instead is not link entities across sentences,
> not build a profile of anyone, and report results only at the type level.
>
> On balance: the eight types range from 28% down to 4%, a ratio under 7 to 1,
> which is mild. But the imbalance that actually matters is different. Only
> twenty-one per cent of tokens are inside an entity at all. So a model that
> predicts "not an entity" for every token scores 79% token accuracy and finds
> exactly zero entities. That is why every number in this presentation is at the
> entity level, never token accuracy.

---

# Part 3 — Annotation Quality & Inter-Annotator Agreement (10 pts)

**Source:** `docs/ANNOTATION_AND_IAA.md` — read that file, it is written for this
part.

### Slide A: "We used an existing dataset — here is their process"

- The rubric recommends against group annotation for NER, and we agree
- **70 annotators + 10 experts**, all with linguistic training
- **Two independent annotators per paragraph**, then an **expert adjudicates**
- Batches of 1,000–3,000 sentences; **10% randomly audited**
- **Below 95% sentence-level accuracy → the whole batch is re-annotated**
- **Cohen's Kappa = 0.7644** → *substantial agreement* (Landis & Koch)

### Slide B: "How the labels are represented"

```
tokens     Barack    Obama     visited   New         York
Few-NERD   person    person    O         location    location
BIO        B-person  I-person  O         B-location  I-location
```

- Few-NERD ships **IO** tags; we convert to **BIO**
- **B-** = first token, **I-** = continuation, **O** = not an entity
- Not BIOES: doubles the label count, and Few-NERD's own baselines use BIO
- **Honest cost:** two same-type entities that touch cannot be separated in IO,
  so that boundary is already lost in the source

### Say (about 2 minutes)

> The rubric says group annotation is not recommended for this task, and having
> looked at what it involves we agree. Sentiment annotation is one label per
> document. NER is a decision for every token — is this inside an entity, where
> does it start, where does it end, which of eight types. Few-NERD used eighty
> people; six students would produce a few hundred noisy sentences.
>
> So what the rubric asks for instead is the original authors' process, and it is
> worth reporting because it is genuinely rigorous. Seventy annotators, all with
> linguistic knowledge, plus ten experienced experts. Every paragraph is
> annotated independently by two people, then an expert reviews it and makes the
> final call — so disagreements are adjudicated, not averaged. The data comes in
> batches, ten per cent of each batch is randomly double-checked, and if that
> sample falls below ninety-five per cent accuracy the entire batch is thrown
> back and re-annotated.
>
> Their Cohen's Kappa is 0.7644, which is substantial agreement on the Landis and
> Koch scale. That is high for NER specifically, because span tasks lose kappa to
> boundary disagreements that are not really disagreements about meaning — two
> annotators who both see "University of Toronto" but differ on whether "the" is
> included count as disagreeing.
>
> One thing about their schema is worth copying: they did several rounds of
> pre-annotation and changed the guideline where it failed. They merged Country,
> Province, City and Restrict into one GPE type because annotators could not
> reliably separate them from context, and they added Person-Scholar because
> pre-annotation turned up scientists the schema had no slot for.
>
> On representation: Few-NERD ships one tag per token with no prefix. We convert
> to BIO, where B marks the first token of an entity and I marks continuation,
> because that is what our three tools all expect. The cost of that conversion is
> that two same-type entities standing next to each other were already merged in
> the source file, and no conversion can recover that boundary.

---

# Part 4 — Preprocessing & Augmentation (15 pts)

**Source:** `reports/02_preprocessing.txt`, `reports/03_augmentation.txt`

This part has an unusual shape and it is a strength, not a weakness: **for NER,
most standard preprocessing is forbidden**, and we measured why.

### Slide A: "What we did NOT do, and the number that proves it"

| step | applied? | the evidence |
|---|---|---|
| tokenization | **not re-done** | Few-NERD ships pre-tokenized; labels are aligned to those exact tokens |
| lowercasing | **no** | **86.8%** of entity tokens are capitalised vs **7.5%** of non-entity tokens |
| stopword removal | **no** | **7.8%** of entities *contain* a stopword — *University **of** California*, ***The** New York Times* |
| stemming | **no** | the output must be a span of the original sentence |
| lemmatization | **as a feature only** | *Islands → island* helps the CRF generalise without damaging the text |
| Unicode normalisation | **yes** | curly quotes and dashes folded; 1,280 tokens fixed per 20,000 sentences |

### Slide B: "Augmentation: mention replacement"

```
before   Barack Obama visited New York .
         B-person I-person O B-location I-location O
after    Angela Merkel visited New York .
         B-person I-person O B-location I-location O
```

- Deletion, insertion and back-translation **all break the labels** — they shift
  or destroy positions
- **Mention replacement** (Dai & Adel, COLING 2020) keeps the sentence and swaps
  one entity for another of the same type
- Targeted the 3 rarest types → **20,000 → 29,410 sentences**, *art*, *event* and
  *building* **+200%** each
- **We measured whether it worked. It did not: micro F1 0.650 → 0.645**
- And it hurt the *targeted* types most (−0.016) while barely touching the rest
  (−0.003) — the opposite of its purpose
- **Why:** on those types precision fell **−0.099** while recall rose **+0.068**.
  More entity names in the same unchanged contexts made the model over-confident
  that those contexts mean *art* — it guessed more, and guessed wrong

### Say (about 2.5 minutes)

> Preprocessing for NER is different from everything else in this course, and
> that is the interesting part of our section.
>
> NER labels belong to *positions*, not to documents. Token five is B-person. If
> you delete a stopword, every label after it shifts by one and the annotation is
> destroyed. So most of the standard pipeline is simply not available, and rather
> than assert that, we measured it.
>
> Lowercasing: 86.8 per cent of entity tokens start with a capital letter,
> against 7.5 per cent of non-entity tokens. Capitalisation is the strongest
> single clue English NER has. Lowercasing deletes it. So we keep case and give
> the models capitalisation as an explicit feature instead.
>
> Stopword removal: 7.8 per cent of the entities in this corpus *contain* a
> stopword. "University of California". "The New York Times". "World War I".
> Removing stopwords would break every one of them.
>
> Stemming: the answer a NER system returns is a span of the original sentence.
> "Univers of Californ" is not a span of anything. But we do use the lemma as one
> CRF feature among many — that gets the generalisation without damaging the
> text.
>
> What we do apply is Unicode normalisation, which fixes curly quotes and dashes
> without changing how many tokens there are.
>
> For augmentation, the methods the rubric lists — deletion, insertion,
> back-translation — all break positional labels. The method that works for NER
> is mention replacement: keep the sentence, swap one entity for another of the
> same type, and rebuild the labels around the new length. We targeted the three
> rarest types and tripled them. The augmented sentences are often factually
> absurd — one of ours reads "the 2020 Battle of the Philippine Sea football
> season" — and that is acceptable for a CRF, which learns surface patterns, but
> would not be for fine-tuning a language model.

---

# Part 5 — Model Design & Training (25 pts) — the three approaches

**Source:** `reports/03_gazetteer.txt`, `reports/04_crf.txt`,
`reports/05_spacy.txt`, and `docs/TRAINING_CODE_BREAKDOWN.md` for the detail.

### Slide A: "Approach 1 — Gazetteer"

- A dictionary of names built **from the training split only**
- **185,158** entries; longest-match scanning, up to 8 tokens
- `min_count` swept 1/2/3/5/10 on **validation** → 1 chosen
- **8,125 entries (4.4%) are ambiguous** — *Washington* is a person and a place
- **Ceiling: only 57.8%** of test entity mentions appear in the training data at
  all

### Slide B: "Approach 2 — CRF"

- Labels the **whole sentence at once**, so it learns that `B-location` cannot be
  followed by `I-person`
- Features per token: word, shape (`Xxxxx`), capitalisation, prefix/suffix,
  lemma, POS — **plus all of that for the 2 tokens on each side**
- `c1` and `c2` (L1 and L2 penalties) tuned on validation → **0.1 / 0.1**
- Trained on a **20,000-sentence subsample** — memory, not choice
- **Best of the three: micro F1 0.650, macro 0.580**
- **It will tell you what it learned** — worth showing:
  - `word.shape:XxXxxx` → **B-person** (the *McDonald* / *DiCaprio* pattern)
  - `-1:word.lower:album` and `-1:word.lower:film` → **B-art**
  - `word.lemma:airport`, `library`, `hospital` → **I-building**
  - `O → I-organization` weight **−7.58** — it learned that is impossible

### Slide C: "Approach 3 — spaCy, pre-trained"

- `en_core_web_sm`, trained on **OntoNotes 5**, **zero** Few-NERD sentences
- Two problems had to be solved before the comparison was fair:
  1. **Tokenization** — spaCy's own tokenizer disagrees with Few-NERD's on
     **30.7%** of sentences; forced alignment with `Doc(words=gold_tokens)`
  2. **Label mismatch** — 18 OntoNotes types vs our 8, mapping **learned from
     validation**, not guessed
- **The finding:** `FAC` (facilities) maps to **location**, not *building* —
  Few-NERD's annotators call facilities locations. So **no OntoNotes label maps
  to `building`**, and spaCy scores exactly **0.000** on it

### Say (about 4 minutes)

> Three approaches, and they are different in kind, which is the point.
>
> The gazetteer is a dictionary. We take every entity in the training data, note
> its text and its type, and then scan each test sentence for the longest
> matching phrase. Longest first matters — matching shortest-first would tag "New
> York" inside "New York Times" and stop there. The critical detail is that it is
> built from training only; building it from all the data would put the test
> answers in the dictionary.
>
> It has two problems by construction. Four per cent of entries are ambiguous —
> Washington is a person and a place, and a dictionary has no context to choose
> with. And only 57.8 per cent of test entity mentions appear anywhere in the
> training data, so before matching even starts, 42 per cent of recall is
> unreachable.
>
> The CRF is the one we actually train. Unlike a per-token classifier, it scores
> the whole label sequence at once, so it learns both what each token looks like
> and which labels can follow which. For each token it sees the word, its shape —
> capitals become X, digits become d — whether it is capitalised, its prefix and
> suffix, its lemma and its part of speech. And it sees all of that for the two
> tokens on each side, which is how it learns that "born in blank" means a
> location.
>
> Two regularisation parameters, c1 and c2, control how aggressively weak
> features are pushed to zero. We tuned them on validation, not test.
>
> One thing to be open about: we trained on 20,000 of the 131,767 training
> sentences. That is a memory limit on our machine, not a design choice, and
> training on all of it is the first item in future work.
>
> The third approach is spaCy's pre-trained model, which never saw Few-NERD. Two
> things had to be fixed before comparing it fairly. First, spaCy runs its own
> tokenizer, which splits differently from Few-NERD, so its spans would not line
> up with the gold spans — we force alignment by building the document from our
> tokens directly. Second, spaCy predicts eighteen OntoNotes types and we have
> eight Few-NERD types. Rather than guess the mapping, we learned it from
> validation: for each OntoNotes label, which Few-NERD type do its correct spans
> actually coincide with.
>
> That produced a genuine surprise. FAC — facilities, so airports and bridges —
> maps to *location*, not to *building*, because Few-NERD's annotators labelled
> them that way. Which means no OntoNotes label maps onto Few-NERD's building
> type at all, and spaCy scores exactly zero on it. Five thousand test entities it
> cannot reach no matter how good it is.

---

# Part 6 — Evaluation & Error Analysis (25 pts)

**Source:** `reports/06_comparison.txt` — the numbers in the slides below must be
copied from that file after the final run.

### Slide A: "How we score, and the headline"

- **Entity level, EXACT span match** — type, first token and last token must all
  match
- A span that overlaps but starts one token early is a false positive **and**
  leaves the gold entity as a false negative — punished twice (CoNLL convention)

| approach | micro P | micro R | **micro F1** | 95% CI | macro F1 |
|---|---|---|---|---|---|
| Gazetteer | 0.266 | 0.477 | **0.342** | [0.339, 0.345] | 0.324 |
| **CRF** | 0.665 | 0.635 | **0.650** | [0.646, 0.653] | **0.580** |
| spaCy | 0.419 | 0.390 | **0.404** | [0.400, 0.408] | 0.233 |

- **micro** = every entity counts equally; **macro** = every type counts equally
- spaCy's micro–macro gap (0.404 → 0.233) is the largest — it works on the
  common types and collapses on the rare ones
- Figure: `figures/fig1_headline_f1.png`

### Slide B: "Per type, and where each one breaks"

- `figures/fig2_per_type_f1.png`

| type | support | Gazetteer | CRF | spaCy |
|---|---|---|---|---|
| location | 27,235 | 0.581 | **0.737** | 0.571 |
| person | 21,565 | 0.361 | **0.775** | 0.599 |
| organization | 19,078 | 0.250 | **0.571** | 0.267 |
| other | 9,558 | 0.349 | **0.463** | 0.029 |
| product | 6,231 | 0.283 | **0.444** | 0.075 |
| building | 5,007 | 0.342 | **0.543** | 0.000 |
| event | 4,104 | 0.361 | **0.514** | 0.145 |
| art | 4,064 | 0.066 | **0.597** | 0.174 |

- The CRF wins every type — but *how far* it wins varies enormously
- spaCy's **0.000 on building is structural**, not a failure of the model
- The gazetteer's **0.066 on art** is the ordinary-words problem: *Today*,
  *Why* and *Hello* are all in its list as `art`, so they match everywhere

### Slide C: "The errors, categorised"

- `figures/fig3_error_breakdown.png` and `figures/fig4_confusion.png`

| of 96,842 gold entities | Gazetteer | CRF | spaCy |
|---|---|---|---|
| correct | 47.7% | **63.5%** | 39.0% |
| type error (right span, wrong label) | 5.8% | 14.6% | 20.2% |
| boundary error (right type, mis-cut) | 17.0% | 6.8% | 12.0% |
| type + boundary | 13.8% | 5.1% | 12.9% |
| missed (nothing predicted) | 15.7% | 9.9% | 16.0% |
| **spurious** (predicted where nothing is) | **95.1%** | 6.9% | 10.3% |

- **The gazetteer predicts almost as many phantom entities as there are real
  ones** — 92,139 spurious against 96,842 gold
- Most CRF mistakes are entities it **did** find and then mis-labelled (14.6%),
  not entities it missed (9.9%)
- Worst confusion for every approach: **organization ↔ location**
- Ensemble result: the three together get **78.0%** of gold entities right,
  against **63.5%** for the CRF alone. The CRF uniquely finds 15,026, the
  gazetteer 8,648, spaCy 2,783 — **all three agree on only 21.6%**

### Say (about 4 minutes)

> Everything here is entity level with exact span matching. A prediction counts
> only if the type, the first token and the last token all match. A span that
> overlaps a gold entity but starts one token early is counted as a false
> positive *and* leaves the gold entity as a false negative — it is punished
> twice. That is the standard CoNLL convention and the strictest reasonable
> reading, and our error analysis separates those near-misses out so you can see
> how big that penalty is.
>
> We report micro and macro F1 because they answer different questions. Micro
> averages over entities, so it is dominated by location, person and organization,
> which are 69 per cent of the test set. Macro averages over the eight types
> equally, so a system that ignores the rare types gets punished for it.
>
> [Walk through fig1, then fig2.] No single approach wins on every type, which is
> exactly why we report all three rather than picking one.
>
> [Walk through fig3.] This is the part worth your attention. Every mistake is
> one of five kinds. A type error means the model found the right span and picked
> the wrong label. A boundary error means it saw the entity but cut it in the
> wrong place. A missed entity means it predicted nothing at all there. Those are
> very different problems: a boundary error is nearly-right and a better feature
> set fixes it, while a missed entity means the model had no evidence.
>
> The row that surprised us is "spurious" for the gazetteer: ninety-five per
> cent. It predicts almost as many entities that are not there as there are real
> entities in the whole test set. That is the ordinary-words problem — "Today",
> "Why" and "Hello" are all in its list as works of art, because somewhere in the
> training data they were song titles. Now they match every ordinary use of those
> words.
>
> And for the CRF, most mistakes are entities it *did* find and then labelled
> wrongly — fifteen per cent — rather than entities it missed, which is ten per
> cent. That is a more hopeful kind of error: the model has the evidence, it just
> picked the wrong type.
>
> [Walk through fig4, the confusion matrix.] Each row is a true type and shows
> where those entities ended up. Organization and location confuse each other for
> every approach, which makes sense — "Manchester United" and "Manchester".
>
> The last result is about the three together. The CRF uniquely finds fifteen
> thousand entities no other approach gets. But the gazetteer uniquely finds
> another eight and a half thousand, and spaCy nearly three thousand. All three
> agree on only twenty-one per cent. Together they would reach seventy-eight per
> cent of all gold entities against sixty-three for the CRF alone. They are not
> three attempts at the same thing — they see different entities, and that gap is
> our main future work item.

---

# Part 7 — Limitations, Future Work, Conclusion (10 pts)

### Slide A: "What our study cannot tell you"

Every one of these is a real limitation of *our* work, not a generic one:

- **The CRF saw 20,000 of 131,767 training sentences** — a memory limit. Its
  numbers are a floor, not its ceiling
- **We model 8 coarse types, not Few-NERD's 66.** Forced by the rubric: spaCy
  cannot produce 66 types, so a 66-way comparison was impossible for one of the
  three required approaches
- **The IO → BIO conversion cannot separate touching same-type entities**, and we
  cannot measure how often that costs us, because the information is missing
- **spaCy is scored on a task it was not built for.** It reaches about 0.85 F1 on
  OntoNotes text; the gap here is the cost of domain plus label mismatch, not its
  quality
- **Wikipedia only.** Well-formed, well-capitalised English prose. None of these
  results transfer to tweets, speech transcripts or lowercase text
- **Our label mapping for spaCy was learned on validation**, so it is fitted to
  this corpus and would not transfer

### Slide B: "Future work" then "Conclusion + references"

- Train the CRF on the **full 131,767 sentences** — the cheapest real gain
- Add **gazetteer membership as a CRF feature** — a hybrid of approaches 1 and 2
- **Ensemble the three**, since they fail on different entities
- Fine-tune a **transformer** (BERT) as the fourth approach the rubric calls
  optional
- Move to the **66 fine-grained types** with the two approaches that can express
  them

**References slide:**

- Ding et al. (2021), *Few-NERD: A Few-shot Named Entity Recognition Dataset*,
  ACL-IJCNLP. Dataset CC BY-SA 4.0
- Lafferty, McCallum & Pereira (2001), *Conditional Random Fields*
- Dai & Adel (2020), *An Analysis of Simple Data Augmentation for NER*, COLING
- Ling & Weld (2012), *Fine-Grained Entity Recognition* (FIGER) — the schema
  Few-NERD adapts
- Tools: sklearn-crfsuite, spaCy `en_core_web_sm`, NLTK, scikit-learn
- Code: this repository

### Say (about 2 minutes)

> Our limitations, honestly.
>
> The CRF trained on 20,000 of 131,767 sentences because the full feature set
> does not fit in memory on our machine. So its number is a floor, not its
> ceiling, and training on everything is our first future work item.
>
> We model eight coarse types instead of Few-NERD's sixty-six. That was forced:
> the rubric requires a pre-trained spaCy model as one of the three approaches,
> and spaCy cannot produce sixty-six types, so the comparison would have been
> impossible for one of the three.
>
> The BIO conversion cannot separate two same-type entities that touch, and we
> cannot even measure how often that costs us, because the information needed to
> measure it is the information that is missing.
>
> And spaCy is being judged on a task it was never built for. On its own domain
> and its own labels it scores around 0.85. What we measured is the cost of
> transfer, and we should say that rather than let it look like a verdict on
> spaCy.
>
> For future work, the strongest single move is a hybrid: give the CRF a feature
> saying "this token is in the gazetteer". That combines approaches one and two
> and costs almost nothing. Beyond that, an ensemble, because we showed the three
> fail on different entities, and then a fine-tuned transformer as the fourth
> approach.
>
> To close: three classic approaches, one corpus, one scoring function, exact
> span matching throughout. They are not interchangeable — they fail differently,
> and knowing how they fail is more useful than knowing which one has the higher
> F1. Everything, including every number in these slides, is in our repository.
> Thank you.

---

# Slide order for the deck builder

| Slide | Owner | Content |
|---|---|---|
| 1 | 7 | Title, group members, the task in one line |
| 2 | 1 | What NER is, our two research questions |
| 3 | 2 | Few-NERD — source, size, licence, ethics |
| 4 | 2 | Class balance + the 21% entity-token point |
| 5 | 3 | The original annotation process and κ = 0.7644 |
| 6 | 3 | BIO representation, with the worked example |
| 7 | 4 | What we did NOT do, with the evidence table |
| 8 | 4 | Mention replacement augmentation |
| 9 | 5 | Approach 1 — gazetteer |
| 10 | 5 | Approach 2 — CRF |
| 11 | 5 | Approach 3 — spaCy, and the FAC finding |
| 12 | 6 | Scoring + headline figure |
| 13 | 6 | Per-type figure |
| 14 | 6 | Error breakdown + confusion matrix |
| 15 | 7 | Limitations, future work, conclusion, references |

**Backup slides:** the CRF's learned transition weights, the full spaCy mapping
table with purity, the gazetteer ambiguity examples, the augmentation worked
examples.

---

# Rules for everyone

1. **Bullets on slides, sentences in your mouth.**
2. **Never quote a number that is not in this document or in `reports/`.**
3. **Say the weak parts first** — the 20,000-sentence subsample, spaCy's zero on
   *building*, the BIO conversion loss. Volunteering them is worth more than
   being caught.
4. **Hand over by name.** "Now [name] takes the preprocessing."
5. **One person answers each question** — whoever owns that part.
6. Nobody says "accuracy". This project reports precision, recall and F1 at the
   entity level, and the reason is on slide 4.

# Every parameter, and why — question prep

Written as spoken answers, not documentation. Each row is something a teacher can
ask and a reply you can give out loud.

The safe general answer when you are stuck:

> "We did not tune that one — it is the library default, and we checked the
> result was not sensitive to it."

That is a good answer. Claiming everything was optimised is a bad answer, because
the follow-up is "show me the tuning".

---

# 1 — Corpus (`scripts/build_corpus.py`)

| Setting | Value | Why |
|---|---|---|
| Few-NERD split | **supervised** | Few-NERD also ships `intra` and `inter`, which hold out whole entity types for few-shot experiments. Our question is ordinary supervised NER |
| Train/val/test | **131,767 / 18,824 / 37,648** | The split published with the dataset |
| Re-splitting | **none** | Wikipedia sentences from one article would land on both sides and leak entities into test |
| Label granularity | **8 coarse**, not 66 fine | See below |
| Tag scheme | IO → **BIO** | BIO is what sklearn-crfsuite, seqeval and spaCy all expect |

**Why 8 types and not 66?**
Three reasons, and the first is decisive. The rubric requires a pre-trained spaCy
model as one of the three approaches; spaCy predicts 18 OntoNotes types and
cannot produce 66, so a 66-way comparison would be impossible for one of the
three required approaches. Second, a gazetteer over 66 types would be 66 lists
fighting over the same surface forms. Third, a 66×66 confusion matrix cannot be
read in a presentation. The fine labels are still in the raw files.

**Why BIO and not BIOES?**
BIOES adds E- for the last token and S- for single-token entities. It gives a CRF
more signal and is a defensible choice. It also doubles the label count — 33
labels instead of 17 — which costs training time on a machine that is already
memory-bound, and Few-NERD's published baselines use BIO, so staying with BIO
keeps our numbers comparable to theirs.

**What did the IO → BIO conversion cost?**
Two entities of the same type standing next to each other with no token between
them are already merged in Few-NERD's released file. We cannot separate them and
we cannot measure how often it happens, because measuring it needs the
information that is missing. 43.2% of entities are one token and 33.1% are two,
so most entities are short and the problem is rare — but rare is not never.

---

# 2 — Preprocessing (`scripts/preprocess.py`)

| Setting | Value | Why |
|---|---|---|
| Re-tokenization | **none** | Labels are aligned to Few-NERD's exact tokens; re-tokenizing shifts every label |
| Unicode | **NFKC** + quote/dash folding | Stops a curly apostrophe becoming its own vocabulary entry. Applied per token, so the token count never changes |
| Lowercasing | **not applied** | 86.8% of entity tokens are capitalised vs 7.5% of others |
| Stopword removal | **not applied** | 7.8% of entities contain a stopword |
| Stemming | **not applied** | The output must be a span of the original sentence |
| Lemmatization | **feature only** | `word.lemma` is one CRF feature; the text is untouched |
| CRF subsample | **20,000** sentences, seed 42 | Memory. See below |

**Why is preprocessing so different here?**
Because NER labels belong to *positions*, not to documents. Token 5 is B-person.
Delete a stopword at token 3 and every label after it is wrong. Stopword removal
and stemming are document-classification techniques; they do not transfer to
sequence labelling. Rather than assert that, we measured what each would cost.

**Why 20,000 sentences for the CRF and not all 131,767?**
Straight answer: the full training set with this feature set does not fit in
memory on our machine — 20,000 sentences already produce **595,404 distinct
feature-value pairs**, and crfsuite holds the feature matrix in memory while
fitting. 20,000 still leaves thousands of examples of every type. It is a floor
on the CRF's performance, not its ceiling, and it is the first item under future
work. The gazetteer and spaCy both use the full data.

**Why seed 42?**
So the subsample is reproducible. Any fixed number would do; 42 is convention.

---

# 3 — Gazetteer (`scripts/approach_gazetteer.py`)

| Setting | Value | Why |
|---|---|---|
| Built from | **training split only** | Building it from all the data would put the test answers in the dictionary |
| Entries | **185,158** surface forms | |
| `MAX_SPAN` | **8** tokens | 98.9% of entities are 6 tokens or shorter; longer costs time and finds nothing |
| Matching | **longest first**, no overlaps | Shortest-first would tag *New York* inside *New York Times* and stop |
| `min_count` | **1**, chosen on validation | Swept 1 / 2 / 3 / 5 / 10; 1 won on validation micro F1 |
| Case | **respected** | `--ignore-case` exists; it raises recall and destroys precision |
| Ambiguity | most frequent type wins | 8,125 entries (4.4%) have more than one type |

**Why is recall higher than precision? That is backwards for a dictionary.**
This is the most interesting thing in that report and worth volunteering. A
gazetteer built from real text contains ordinary words that happen to be titles
of songs, films and books. *Today*, *Why* and *Hello* are all in ours as `art`,
so they now match every ordinary use of those words. That is why `art` has huge
false positives - 0.066 F1 - and why the gazetteer predicts 92,139 spurious
entities against 96,842 real ones. A hand-curated gazetteer would behave the classic way; one learned
from a corpus does not.

**Why didn't raising `min_count` fix that?**
It does raise precision — at `min_count=10` precision goes from 0.27 to 0.41 —
but recall falls faster, so F1 drops. We chose on F1, on validation, and reported
the whole sweep rather than the value that flatters us.

**What is the ceiling on this approach?**
**57.8%** of test entity mentions have their exact surface form somewhere in the
training data. The other 42.2% are names the dictionary has never seen and cannot
invent. That single number explains the result.

---

# 4 — CRF (`scripts/approach_crf.py`)

| Setting | Value | Why |
|---|---|---|
| Library | **sklearn-crfsuite** (CRFsuite) | The standard Python CRF for sequence labelling |
| `algorithm` | **lbfgs** | Limited-memory BFGS — approximates curvature from recent gradients instead of storing a full matrix, which is what makes millions of features tractable. CRFsuite's default |
| `c1` (L1) | tuned, **0.1** | Pushes weak weights to exactly zero. Most of our features are one-off words that *should* be zero |
| `c2` (L2) | tuned, **0.1** | Keeps all weights small so no single feature dominates |
| `max_iterations` | **120** | Enough to converge here; a hard stop so a bad setting cannot run forever |
| `all_possible_transitions` | **True** | Lets it learn a weight for every label pair, including unseen ones |
| `WINDOW` | **±2 tokens** | Standard. Each extra position adds ~5 features per token for shrinking gains |
| `AFFIX` | **3 characters** | Catches *-ton*, *-ville*, *-burg*, *-ism* without exploding the feature count |

**What is `all_possible_transitions=True` actually doing?**
With `False`, the model can only score label pairs it saw in training — an unseen
but illegal pair like `B-location → I-person` gets no penalty at all. With `True`
it learns a weight for every pair, so it can assign a large negative weight to
the impossible ones. `reports/04_crf.txt` §4 prints those learned weights, and
the most negative ones are exactly the sequences that cannot occur.

**How did you choose c1 and c2?**
Five combinations on **validation**, best micro F1 wins. The grid is centred on
(0.1, 0.1), which is CRFsuite's own documented example. Test was used once,
after the choice. The full grid with scores is in the report — the spread across
the five settings is small, about 0.015 F1, so this is not a parameter the
result hinges on.

**Why these features and not word embeddings?**
Because the point of including a CRF is to show what the classic feature-based
approach does. Adding embeddings would make it a different model and would blur
the comparison with the pre-trained system, which is precisely the approach that
*does* bring outside knowledge.

**Which feature matters most?**
Capitalisation and shape. `reports/04_crf.txt` §5 prints the strongest learned
features per label. Shape is what lets it beat the gazetteer on names it has
never seen: `Xxxxx Xxxxx` is a person pattern whether or not the model knows
that particular name.

**Why is a CRF better than classifying each token separately?**
Because it scores the whole label sequence and picks the best path through it
(Viterbi), so it can learn that `I-person` cannot follow `B-location`. A per-token
classifier has no way to express that.

---

# 5 — spaCy (`scripts/approach_spacy.py`)

| Setting | Value | Why |
|---|---|---|
| Model | **en_core_web_sm** | The standard small English pipeline |
| Trained on | **OntoNotes 5** | News, phone speech, weblogs — not Wikipedia |
| Fine-tuned on Few-NERD | **no, zero sentences** | The rubric asks for a *pre-trained* approach |
| Tokenization | **`Doc(nlp.vocab, words=gold_tokens)`** | Bypasses spaCy's tokenizer so spans align with gold. Without it, spaCy's tokens differ from Few-NERD's on **30.7%** of sentences |
| `batch_size` | 256 | Throughput only; changes nothing about the result |
| `exclude` | `lemmatizer` | Not needed for NER, and it is the slowest component |
| Label mapping | **learned on validation** | Not guessed |
| Numeric labels | **dropped** | Few-NERD has no type for dates, money or counts |

**Why not use the bigger model, `en_core_web_trf`?**
A fair question and the honest answer is that `sm` is the standard baseline and
runs on this machine in minutes. Using `trf` would raise the numbers; it would
not change the two structural findings — the tokenization problem and the
`building` gap — which are what our section is actually about.

**How was the label mapping learned?**
For every OntoNotes label, we looked at which Few-NERD type its exactly-matching
spans coincide with **on validation**, and mapped it there. The report prints a
purity column showing how consistent each mapping is: `GPE → location` is 83.8%
pure, `PRODUCT → product` only 30.1%. Low purity means that mapping is a
compromise, and we show it rather than hide it.

**Why drop CARDINAL, DATE, MONEY, ORDINAL, PERCENT, QUANTITY and TIME?**
Few-NERD deliberately excludes value, numerical, time and date entities — the
paper says so. So a prediction of one of those can only ever be a false positive.
Dropping them is the mapping decision that most helps spaCy's score, and we state
it openly rather than let someone find it.

**Why does spaCy score exactly 0.000 on `building`?**
Because no OntoNotes label maps onto it. The obvious candidate is `FAC`
(facilities — airports, bridges, buildings), but when we checked against
validation, `FAC` spans coincide with Few-NERD **location** more often than with
`building` — Few-NERD's annotators treat facilities as places. So `FAC → location`
is what the data says, and `building` is left unreachable: 5,007 test entities
spaCy cannot score above zero on, no matter how good it is. **This is our best
example of why the mapping had to be learned rather than assumed.**

---

# 6 — Augmentation (`scripts/augment.py`)

| Setting | Value | Why |
|---|---|---|
| Method | **mention replacement** | The only listed-adjacent method that preserves positional labels |
| Mention bank | built from the **full training split** | Legitimate extra information; never validation or test |
| `--rare` | **3** rarest types | `art`, `event`, `building` |
| `--copies` | **2** per eligible sentence | Roughly triples the targeted types without swamping the corpus |
| Seed | 42 | Reproducible |
| Result | 20,000 → **29,410** sentences | Targeted types **+200%** each |

**Why not random deletion / insertion / back-translation, which the rubric lists?**
Because all three assume the label belongs to the document. Deletion shifts every
label after the deleted token. Insertion can land inside an entity and corrupt
its span. Back-translation returns fluent text with different token boundaries
and no way to map the old labels on. Reference for the method we used instead:
Dai & Adel, COLING 2020.

**Your augmented sentences are nonsense.**
Yes — *"the 2020 Battle of the Philippine Sea football season"* is one of ours.
That is acceptable for a CRF, which learns surface patterns and context words,
not world knowledge. It would **not** be acceptable for fine-tuning a language
model, which implausible text can damage. We say this in the report ourselves.

**Did it help?**
Measured, not asserted: `scripts/augmentation_effect.py` reruns the CRF on the
augmented file with the same features, the same tuning grid and the same test
set, so the only difference is the training data. The result is in
`reports/07_augmentation_effect.txt` — read it before the presentation and quote
whichever way it went. A negative result honestly reported is worth more than a
positive one quietly obtained.

---

# 7 — Evaluation (`scripts/evaluate.py`, `scripts/nerlib.py`)

| Setting | Value | Why |
|---|---|---|
| Level | **entity**, never token | 78.9% of tokens are O; token accuracy is meaningless here |
| Matching | **exact span** | Type, first token and last token must all match |
| Averages | **micro and macro** | They answer different questions |
| Bootstrap | **1,000** resamples of sentences | Confidence interval on micro F1 |
| Error categories | the **five MUC types** | Type / boundary / both / missed / spurious |

**What exactly counts as correct?**
Type, start token and end token all match. A span that overlaps a gold entity but
starts one token early is a false positive *and* leaves the gold entity as a
false negative — punished twice. That is the CoNLL convention and the strictest
reasonable reading. Our error breakdown separates those near-misses out so the
size of that penalty is visible rather than hidden.

**Why report both micro and macro?**
Micro averages over entities, so it is dominated by location, person and
organization — 69% of the test set. Macro averages over the eight types equally,
so a system that ignores rare types is punished. Micro answers "how often is this
system right"; macro answers "does it work for every type". A big gap between
them means the system is ignoring the rare types.

**Why not use seqeval?**
We could — it is installed. We wrote the scorer so that all three approaches go
through **one** function, and so we could add the MUC error categories and the
bootstrap interval, which seqeval does not provide. The definition of a correct
entity is identical to seqeval's default strict mode.

**What do you do with malformed predictions?**
A model can emit `I-person` with no `B-person` before it. We treat that as the
start of an entity rather than dropping it. Dropping malformed predictions would
quietly flatter every model by deleting some of its own false positives.

---

# 8 — The numbers to have memorised

| | |
|---|---|
| **188,238** | sentences in Few-NERD; 485,763 entities; 8 types |
| **0.7644** | the original authors' Cohen's Kappa — substantial agreement |
| **21.1%** | of tokens are inside an entity — why we never report accuracy |
| **57.8%** | of test entity mentions appear in training — the gazetteer's ceiling |
| **86.8% vs 7.5%** | capitalised entity tokens vs other tokens — why we keep case |
| **7.8%** | of entities contain a stopword — why we do not remove them |

**The three results** (test, entity level, exact span match):

| approach | micro F1 | macro F1 |
|---|---|---|
| Gazetteer | 0.342 | 0.324 |
| **CRF** | **0.650** | **0.580** |
| spaCy (pre-trained) | 0.404 | 0.233 |

**78.0%** — what all three together get right, against 63.5% for the CRF alone.
**21.6%** — how much all three agree on. They see different entities.

---

# 9 — If you genuinely do not know

> "I did not set that one — it is the library default. I can tell you what it
> controls and why it did not change our result."

or

> "That is in the repo, section 2 of the CRF report. I can show you after."

Both beat guessing. A wrong number said confidently is the only answer that
actually costs marks.

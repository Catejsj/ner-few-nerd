# Slides — Model Design & Training (Seth)

My part of the NER presentation. Rubric criterion 5, **25 points** — the joint
biggest section, along with evaluation.

Two slides.

- **Part A** is for whoever builds the slides. Copy the boxes onto the slide.
  Nothing else goes on.
- **Part B** is what I say out loud. Nobody reads it but me.
- **Part C** explains my picture: what every part of it means.
- **Part D** explains the technical words.
- **Parts E–G** are Q&A prep, where each number came from, and what I must not
  say because it belongs to someone else.

**My job in one line:** build the three NER approaches the rubric asks for —
a dictionary, a CRF, and a pre-trained model — and explain how each one is
trained and why it is set up the way it is.

**The scores belong to the evaluation person, not me.** I set the models up and
hand over. See Part G.

---

# PART A — for the slide maker

**Two slides.** Slide 1 is a **two-column layout** — the two approaches we did
not train, side by side. Slide 2 is full width because the picture needs it.

---

## Slide 1 — TWO COLUMNS

**Title of the slide:**

> Two approaches that need no training

### LEFT COLUMN

**Column heading:** `1. Gazetteer — a dictionary`

- A **list of known names**, then look them up
- Built from the **training sentences only** — using all the data would put the
  test answers in the dictionary
- **185,158** names, matched **longest phrase first**
- No learning, no weights — the baseline the others must beat

**Small monospace box under the bullets:**

```
"New York Times"  -> organization   (longest first)
"New York"        -> would stop here, and be wrong
```

**Two red-flag lines:**

- **4.4%** of names have **two types** — *Washington* is a person and a place
- Only **57.8%** of test names appear in training at all

### RIGHT COLUMN

**Column heading:** `3. spaCy — already trained by someone else`

- `en_core_web_sm`, trained on **OntoNotes** — news, not Wikipedia
- **Zero** Few-NERD sentences. We trained nothing

**Small monospace box:**

```
Problem   its tokenizer disagrees with ours on 30.7% of sentences
Fix       feed it our tokens - Doc(words=gold_tokens)

Problem   it predicts 18 OntoNotes types, we have 8
Fix       learn the mapping from validation, do not guess
```

**Then the finding — make this stand out, it is the best thing on the slide:**

- We assumed `FAC` (airports, bridges) → `building`.
  **The data said `FAC` → `location`.**
- So **nothing maps to `building`** — spaCy scores **0.000** on it,
  5,007 entities it cannot reach

**No picture on this slide.** The two boxes are the visual. Keep the columns
clearly separated — a vertical rule or real white space between them, not just a
gap.

---

## Slide 2 — FULL WIDTH

**Title of the slide:**

> Approach 2 — the CRF, the one we actually train

**Bullets across the top, short — the picture does the work:**

- Labels the **whole sentence at once**, not one word at a time
- So it learns that `I-person` **cannot** follow `B-location`
- Sees each word **plus the 2 words on each side**
- **595,404** feature weights, fitted with L-BFGS
- `c1 = 0.1`, `c2 = 0.1`, tuned on validation
- Trained on **20,000** of 131,767 sentences — memory limit, not a choice

**Picture:** `figures/fig5_crf_features.png`
Full width, under the bullets. **This is the most important picture in my part.**

---

**If the deck has room and someone wants three slides**, split slide 1 back into
two — gazetteer and spaCy each get their own, with the same content. Nothing
needs rewriting; the columns just become slides.

# PART B — what I say out loud

About 4 minutes total. This is the shape, not a script to memorise.

**Order on slide 1: gazetteer first (left), then spaCy (right), then move to
slide 2 for the CRF.** Two shortcuts, then the one that works — the CRF is the
payoff, so it goes last.

## Slide 1, left column — the gazetteer  (about 60 seconds)

> My part is building the three approaches. They are different in kind, which is
> the point of comparing them.
>
> Two of them need no training at all, and I will take those together first.
> The first is a gazetteer, which is just a dictionary. We go through the
> training sentences, and every time there is a labelled entity we write down its
> text and its type. "Barack Obama, person." "New York, location." That gives us
> 185,000 names. Then for a new sentence we scan along looking for the longest
> phrase that is in our list.
>
> Longest first matters. If we matched shortest first, "New York Times" would be
> tagged as just "New York", a location, and we would stop there and miss the
> newspaper entirely.
>
> The one thing I want to stress is that the dictionary is built from the
> **training** sentences only. If we had built it from all the data, the test
> answers would be sitting inside the dictionary and the score would be
> meaningless. That is the easiest way to accidentally cheat in a project like
> this.
>
> It has two problems it cannot escape. Four per cent of the names have more than
> one type — Washington is a person and a place — and a dictionary has no context
> to choose with, so we take whichever was more common and accept being wrong
> sometimes. And only fifty-eight per cent of the names in the test set appear in
> the training data at all. The other forty-two per cent are names it has never
> seen and cannot invent.

## Slide 1, right column — spaCy  (about 80 seconds)

> The other untrained one is spaCy's off-the-shelf model.
> It was trained by spaCy on OntoNotes, which is news and telephone speech, and
> it has never seen a single Few-NERD sentence. So part of what we are measuring
> here is how far an off-the-shelf model transfers to a new domain and a new
> label set.
>
> Two things had to be fixed before it was a fair comparison.
>
> First, tokenization. spaCy normally splits the text itself, and we measured
> that its split disagrees with our corpus on thirty per cent of sentences. If we
> had let it, its predicted positions would not line up with our gold positions,
> and we would be measuring tokenizer disagreement instead of entity recognition.
> So we feed it our tokens directly, which turns its tokenizer off.
>
> Second, the labels do not match. spaCy predicts eighteen OntoNotes types and we
> have eight. Some are obvious — PERSON is person. Others are not: NORP is
> nationalities and religious groups, and there is no obvious home for it.
>
> So rather than write the mapping by hand, we learned it. For each OntoNotes
> label we looked at which of our types its correct spans actually line up with,
> on the validation set, and mapped it there.
>
> That produced the finding I would most like you to remember. We expected FAC —
> facilities, so airports and bridges — to map to our "building" type. The data
> said it maps to "location", because Few-NERD's annotators labelled facilities
> as places more often than as buildings. Which means no OntoNotes label reaches
> our building type at all, and spaCy scores exactly zero on it — five thousand
> test entities it cannot touch no matter how good it is.
>
> If we had written that mapping from intuition we would never have found out.

## Slide 2 — the CRF  (about 90 seconds)

> Now the one we actually train: a Conditional Random Field.
>
> The important difference is that it does not label words one at a time. It
> scores the whole sequence of labels for the sentence together, and then picks
> the single best path through all the possibilities. That means it can learn
> rules about label order — for example that an "inside-person" tag cannot follow
> a "begin-location" tag, because that is not a thing that can happen. A
> word-by-word classifier has no way to even express that.
>
> [Point at the picture.] This is what it sees for one word. We are labelling
> "New". The blue band is the context window — it can see two words to the left
> and two to the right.
>
> From the word itself it takes: the word, its lowercase form, its lemma, its
> shape — capital letters become X, so "New" becomes X-x-x — whether it starts
> with a capital, whether it is all caps, whether it has digits, and its first
> and last three letters, plus its part of speech.
>
> Then it takes the same kind of information from the four surrounding words.
> Thirty-four features for this one word, twenty of them from the context.
>
> And the context is the whole point. The word "New" on its own tells you almost
> nothing — "new car", "New York". What tells the model this is the start of a
> location is that "visited" comes before it and "York" comes after. Those are
> literally two of the features in the list.
>
> Training means finding a weight for each of those 595,000 feature-value pairs,
> using L-BFGS. Two settings control how hard the model is pushed to keep those
> weights small, and we tuned them on the validation set — never on test.
>
> One thing to be upfront about: we trained on 20,000 of the 131,767 training
> sentences. That is a memory limit on our machine, not a design decision. So our
> CRF number is a floor, not a ceiling, and training on all of it is the first
> item in our future work.
>
> [Name] will now take you through what all three actually scored.

---

# PART C — my picture, explained

`figures/fig5_crf_features.png`. Read this before presenting — if someone points
at it, the answer is here.

### The top strip

- **tokens row** — a real sentence broken into words, exactly as the model gets
  it. "Barack Obama visited New York in 1961 ."
- **The blue box on "New"** — the word currently being labelled. The model does
  this once for every word in the sentence.
- **The pale blue band** — the ±2 context window. It covers *Obama, visited,
  **New**, York, in*. The model is allowed to use all five when deciding about
  "New", and nothing outside.
- **`-2`, `+2` labels** — how far each word is from the one being labelled. Those
  numbers appear in the feature names below, which is how you can tell which
  feature came from which position.
- **POS tag row** — the part of speech from NLTK. `NNP` is a proper noun, `VBD`
  is a past-tense verb, `IN` a preposition, `CD` a number.
- **gold BIO row** — the right answer for this sentence. `B-` starts an entity,
  `I-` continues it, `O` is not an entity. Green means it is part of an entity.

### The feature lists

The heading says **34** — that is how many features this single word produces.
Every one is computed from the text; none are written by hand.

**Left column, from the word itself:**

| feature | value here | what it is for |
|---|---|---|
| `word` | New | the exact word — memorises names it has seen |
| `word.lower` | new | so *New* and *new* can share evidence |
| `word.lemma` | new | dictionary form, so *Islands* and *Island* match |
| `word.shape` | Xxx | capitals→X, lowercase→x, digits→d. **Generalises to names it has never seen** |
| `word.istitle` | True | starts with a capital |
| `word.isupper` | False | all caps, e.g. *NASA* |
| `word.isdigit` / `hasdigit` | False | numbers behave differently |
| `word.hashyphen` | False | *Rolls-Royce*, *Coca-Cola* |

**Right column, from the context** — 20 of the 34:

`-2:word = Obama`, `-1:word = visited`, `+1:word = York`, `+2:word = in`, plus
capitalisation and POS for each of those four positions.

### The sentence at the bottom

This is the line to say out loud:

> "New" alone is ambiguous — *new car* versus *New York*. What tells the model
> this starts a location is `-1:word = visited` and `+1:word = York`. The
> evidence is not in the word; it is in the neighbours.

### Why the picture exists at all

"The CRF uses context features" is a sentence nobody can picture. This replaces a
paragraph of explanation with one look. It is also proof we know what our own
model does, rather than having called `.fit()` and reported a number.

### If someone points at it

> "This is one word being labelled. The blue band is everything the model is
> allowed to look at. The list below is what it extracts — thirty-four features,
> twenty of them from the neighbouring words. That is the whole input; there are
> no embeddings and no outside knowledge."

---

# PART D — the words explained

**NER** — find the names in a sentence and say what kind of thing each one is.

**Token** — one word (or punctuation mark). The model makes one decision per
token.

**BIO** — the labelling scheme. `B-person` = first word of a person's name,
`I-person` = a continuing word of it, `O` = not part of any entity.

**Gazetteer** — just a dictionary of known names with their types. No maths.

**CRF (Conditional Random Field)** — a model that scores the whole sequence of
labels for a sentence at once, instead of each word separately. That lets it
learn which labels can follow which.

**Feature** — one piece of information about a word that the model is allowed to
use. "Is it capitalised" is a feature. Our model has 34 per word.

**Context window** — how many words either side the model may look at. Ours is
±2.

**Weight** — a number the model learns for each feature, saying how strongly it
points at each label. Training means finding all 595,404 of them.

**L-BFGS** — the optimisation method that finds those weights. It works out
which direction to adjust them from recent gradients, instead of storing a huge
matrix — which is what makes a model this size fit in memory.

**c1 and c2** — two penalties that stop the model trusting any single feature too
much. c1 pushes useless weights to exactly zero; c2 keeps them all small.

**Viterbi** — the algorithm that picks the single best label sequence out of all
possible ones. This is what "predict" means for a CRF.

**Pre-trained model** — one somebody else already trained, that we just run.

**OntoNotes** — the corpus spaCy's model was trained on: news, phone calls,
weblogs. Not Wikipedia. That difference is called a *domain shift*.

**Validation set** — data used for choosing settings. Test is used once, at the
end, after everything is decided.

---

# PART E — questions I might get

**Why use a gazetteer at all if it is so weak?**
Because it is the honest baseline. If a trained model cannot beat a dictionary
lookup, the training added nothing. It is also genuinely the right tool for
closed classes — a list of countries or currencies is hard to improve on. Ours
underperforms because entity names in Wikipedia are an open set, and we show
exactly why with the 57.8% coverage number.

**Why is a CRF better than classifying each word separately?**
Because a per-word classifier can output `B-location` followed by `I-person`,
which is not a possible entity. The CRF scores the whole sequence, so it learns
those transitions. In our trained model `O → I-organization` has a weight of
**−7.58** — it worked out on its own that you cannot be inside an organization
name without having started one.

**What does `all_possible_transitions=True` do?**
With it off, the model can only score label pairs it actually saw in training, so
an illegal-but-unseen pair gets no penalty at all. With it on, it learns a weight
for every pair, including the impossible ones, and can push those strongly
negative. That is where the −7.58 comes from.

**How did you pick c1 and c2?**
Five combinations, scored on validation, best micro F1 wins — centred on
(0.1, 0.1), which is CRFsuite's own documented example. The spread across the
five was about 0.015 F1, so the result does not hinge on this. Test was touched
once, after the choice was made.

**Why only 20,000 training sentences? That looks lazy.**
It is a memory limit and I would rather say so than dress it up. 20,000 sentences
already produce 595,404 distinct feature-value pairs, and CRFsuite holds the
whole feature matrix in memory while fitting. The full set is 6.6 times larger
and does not fit on our machine. It still leaves thousands of examples of every
entity type, our number is a floor rather than a ceiling, and "train on
everything" is the first line of our future work.

**Why not word embeddings, or BERT?**
Because the rubric asks for these three specific approaches, and the point of
including a CRF is to show what the classic feature-based method does. Adding
embeddings would make it a different model and blur the comparison with the
pre-trained system — which is exactly the approach that *does* bring outside
knowledge. A fine-tuned transformer is listed as our fourth-approach future work.

**Why didn't you fine-tune spaCy on Few-NERD?**
Because the rubric asks for a *pre-trained* model as the third approach. Its
value in this comparison is precisely that it has never seen our data.

**Isn't dropping spaCy's date and number labels unfair to the others?**
It is the decision that most helps spaCy, and I would rather state it than have
it found. Few-NERD deliberately excludes date, time, money and numeric entities —
the paper says so. So a `DATE` prediction can only ever be a false positive. We
drop those seven labels; everything else is mapped.

**Could you combine the three?**
Yes, and that is the most interesting future work. They find different entities —
[name] has the numbers on that in the evaluation section. The cheapest version is
a hybrid: give the CRF one extra feature saying "this token is in the gazetteer".

---

# PART F — where my numbers come from

| Number I say | File | How |
|---|---|---|
| 185,158 gazetteer entries | `reports/03_gazetteer.txt` §1 | distinct entity surface forms in training |
| 8,125 ambiguous (4.4%) | §2 | entries seen with more than one type |
| 57.8% coverage | §3 | test entity mentions whose exact text appears in training |
| `min_count` = 1 | §0 | swept 1/2/3/5/10, best validation micro F1 |
| 34 features per token | the figure | counted by `token_features()` itself |
| 595,404 feature-value pairs | `reports/04_crf.txt` §1 | distinct (feature, value) pairs in the training set |
| c1 = 0.1, c2 = 0.1 | §2 | best of 5 combinations on validation (F1 0.649) |
| 20,000 / 131,767 sentences | §1 | the subsample, seed 42 |
| 91 seconds to fit | §3 | wall time of the final fit |
| `O → I-organization` = −7.58 | §4 | a learned transition weight |
| 30.7% tokenizer disagreement | `reports/05_spacy.txt` §2 | spaCy's tokens vs Few-NERD's, 3,000 sentences |
| FAC → location | §3 | learned from validation; 47.7% purity |
| building = 0.000 | §5 | no OntoNotes label maps to it — 5,007 test entities |

Everything regenerates by re-running the scripts. Nothing is typed by hand.

---

# PART G — what is NOT mine

Do not present these. If asked, name the person who owns them.

- **The scores.** micro/macro F1 for the three approaches, the per-type table,
  the confusion matrix, the error breakdown — **all of that is the evaluation
  part**. I finish by handing over, not by announcing a winner.
- **`figures/fig1` to `fig4`** belong to the evaluation slides. **My picture is
  `fig5` only.** If I show fig1 too, the audience sees the same chart twice.
- **Preprocessing** — why we do not lowercase or remove stopwords, and the
  augmentation result — belongs to the preprocessing person.
- **The annotation process and Cohen's Kappa** belong to the annotation person.

If I want one sentence of result to hand over on, it is this and nothing more:

> "The trained CRF came out ahead of both shortcuts — [name] has the numbers."

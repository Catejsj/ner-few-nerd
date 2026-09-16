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

**My job in one line:** build the three ways of finding names that the rubric
asks for, and explain how each one works.

**The one idea that holds it together — who did the learning:**

| method | who learned it |
|---|---|
| **1. Gazetteer** (name list) | nobody — we just copied names into a list |
| **2. CRF** | **us**, on our data |
| **3. spaCy** | a company, on different text, before our project |

Say that at the start and everything after it has somewhere to sit.

**The scores belong to the evaluation person, not me.** I set the models up and
hand over. See Part G.

---

# PART A — for the slide maker

Type exactly what is in the bullets. Nothing else goes on the slide.

---

## Slide 1 — TWO COLUMNS

**Slide title:** `Method 1 and Method 3: a name list, and a ready-made model`

### LEFT COLUMN

**Heading:** `Method 1 — Gazetteer (a name list)`

- We collect every name in the training data → **185,158 names**
- For a new sentence, look up the **longest match** first
- Problem: **4.4%** of names have two meanings — *Washington* is a person and a place
- Problem: only **57.8%** of test names appear in the training data at all

### RIGHT COLUMN

**Heading:** `Method 3 — spaCy (trained by someone else)`

- A ready-made model, trained on **news text**, before our project
- It never saw our data. We only ran it
- It cuts sentences into words differently from ours — **30.7%** disagree, so we hand it our words
- It has **18** name-types, we have **8** → we matched them using set-aside data
- Nothing matched our **building** type → spaCy scores **0.000** there

No picture on this slide.

---

## Slide 2 — FULL WIDTH

**Slide title:** `Method 2: the model we trained ourselves`

- Decides the labels for the **whole sentence together**, not word by word
- So it never produces an impossible combination
- For each word it looks at **that word + 2 before + 2 after**
- **34 clues per word**: capital letters, word shape, word endings, word type
- Trained on **20,000 sentences**; settings picked on set-aside data

**Picture:** `figures/fig5_crf_features.png`, full width, under the bullets.

---

# PART B — what I say out loud

About 4 minutes. Say it in my own words; this is the shape.

## Opening — the map  (15 seconds)

> There are three ways to find names in text, and the real difference between
> them is simply **who did the learning**.
>
> The first one, nobody learned anything — we just wrote out a list.
> The second one, we trained ourselves, on our data.
> The third one was trained by a company, on different text, before this project
> started.
>
> This slide has the list and the ready-made model. The next slide has the one we
> trained.

## Slide 1, left — the name list  (60 seconds)

> A gazetteer is just a list of names.
>
> We went through our training sentences, and every time a name was marked, we
> wrote it down with its type. "Barack Obama — person." "New York — location."
> That gave us 185,000 names.
>
> Then for a new sentence we slide along it and ask: is this phrase in my list?
> We always try the longest phrase first. If we did not, "New York Times" would
> come out as "New York" — a place instead of a newspaper.
>
> One thing matters a lot here: we only used the **training** sentences to build
> the list. If we had used the test sentences too, the answers would be sitting
> inside the list, and our score would be fake.
>
> It has two problems it cannot fix. Some names mean two things — "Washington" is
> a person and a place — and a list has no way of telling which one you meant.
> And most test names were never in the training data: only 58% of them appear
> there. The other 42% are names it has simply never seen, and a list cannot
> invent a name.

## Slide 1, right — the ready-made model  (75 seconds)

> The third method is spaCy. spaCy is a free language-processing library, and it
> comes with a model that already knows how to find names. The spaCy team trained
> it on news articles, long before this project. We did not train it at all — we
> just ran it on our sentences and saw what it found.
>
> Two things had to be fixed before that was a fair test.
>
> First: spaCy cuts a sentence into words its own way, and our data was already
> cut a different way. We checked, and they disagree on 30% of sentences. If we
> let spaCy do the cutting, the positions it reports would not line up with the
> positions in our answer key, and we would be measuring the cutting rather than
> the name-finding. So we hand spaCy our words directly.
>
> Second: spaCy uses 18 name-types and our data uses 8, and they do not line up
> one to one. So instead of guessing which goes with which, we used a set-aside
> part of our data to check. When spaCy says "GPE", what does our data actually
> call that thing? It calls it a location. So GPE becomes location.
>
> Doing it that way caught something we would have got wrong. spaCy has a type
> called FAC, for facilities — airports, bridges, stadiums. We assumed that was
> our "building" type. The data said no: our annotators call those things
> locations. Which means nothing at all maps to "building", so spaCy scores zero
> there — five thousand names it cannot get right, no matter how good it is.
>
> If we had written that matching by hand, we would never have found out.

## Slide 2 — the one we trained  (90 seconds)

> This is the model we trained ourselves. It is called a CRF.
>
> The simplest way to say what it does: instead of deciding each word on its own,
> it decides the labels for the **whole sentence together** and picks the best
> combination. That matters, because some combinations are impossible — you
> cannot have "middle of a person's name" right after "start of a place name". A
> word-by-word method can produce that. This one learns not to.
>
> [Point at the picture.] This is what it looks at for one word. We are labelling
> "New". The blue band is what it is allowed to see: two words before and two
> words after.
>
> For that one word it collects 34 small clues. The word itself. Whether it
> starts with a capital. The shape of it — capitals become X, so "New" is X-x-x.
> The last three letters. What kind of word it is. And then the same set of clues
> for each of the four neighbours.
>
> The neighbours are the whole point. "New" on its own tells you nothing — "new
> car", "New York". What makes it a place is that "visited" comes before it and
> "York" comes after. Those are two of the thirty-four clues, right there in the
> list.
>
> Training means going through 20,000 sentences and working out how much each
> clue is worth. We used 20,000 rather than all 130,000 because the full set did
> not fit in our computer's memory — so our result is a floor, not the best this
> model could do.
>
> [Name] will now tell you how the three actually scored.

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

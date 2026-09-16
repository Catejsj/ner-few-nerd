# Annotation and inter-annotator agreement (rubric criterion 3, 10 points)

## The decision: we use an existing labelled dataset

The rubric offers two routes and is explicit about which it prefers:

> "For annotation done by the group (**not recommended in this class**) …
> **if using an existing labeled dataset**: please report on the annotation
> guideline and the inter-annotator agreement of the original authors."

We take the second route and use **Few-NERD**. What the rubric then asks for is
not our own kappa — it is a report on the original authors' annotation process
and agreement. That is what this document is.

### Why annotating NER ourselves would have been the wrong choice

Worth being able to say out loud, because it is a fair question:

- **Span annotation is far harder than document annotation.** For sentiment you
  pick one label per document. For NER you must decide, for every token, whether
  it is inside an entity, where the entity starts, where it ends, and which of 8
  types it is. A 25-token sentence is 25 decisions, not one.
- **The boundary is where annotators disagree.** Is it *the University of
  Toronto* or *University of Toronto*? *New York Times* or *The New York Times*?
  Without a guideline that answers these in advance, agreement collapses, and
  writing that guideline is itself a large piece of work.
- **The type ambiguity is real.** Few-NERD's own paper uses this example:
  *"London is the fifth album by the British rock band Jesus Jones"* — here
  **London is art (a music album), not a location.** Getting that right requires
  reading the context properly, every time.
- **Scale.** Few-NERD is 188,238 sentences annotated by 80 people. Six students
  in a semester could produce a few hundred sentences — enough to measure
  agreement on, not enough to train or fairly evaluate three approaches.

The honest summary: we would have produced a small, noisy corpus and then spent
the project explaining its noise, instead of comparing three NER methods, which
is what the rubric actually weights (25 + 25 points).

---

## The original authors' annotation process

Source: Ding et al., *Few-NERD: A Few-shot Named Entity Recognition Dataset*,
ACL-IJCNLP 2021, section 4.

### Who annotated

| | |
|---|---|
| Annotators | **70** |
| Experts | **10** experienced experts, on top of the 70 |
| Requirement | all annotators have **linguistic knowledge** and were instructed with detailed and formal annotation principles |
| Effort | each annotator spent an average of **32 hours** |
| Payment | annotators were paid at market rate by workload — the paper states this explicitly |

That last row matters for the ethics part of the rubric: this is not unpaid or
undisclosed crowd labour.

### The procedure

1. **Two independent annotators** label every paragraph. Neither sees the
   other's work.
2. **An experienced expert** then reviews the paragraph for wrong or missing
   annotations and **makes the final decision**. So disagreements are resolved
   by adjudication, not by voting or by coin-flip.
3. Data is annotated and submitted **in batches of 1,000–3,000 sentences**.
4. For each batch, **10% of sentences are randomly selected and double-checked**.
5. **If the accuracy of that sample is below 95% (measured at sentence level),
   the whole batch is re-annotated.** A hard quality gate, not a warning.

### The agreement number

> **Cohen's Kappa = 76.44%** between two annotators.

On the Landis & Koch scale — the same scale we used for our own kappa in the
sentiment project — 0.61 to 0.80 is **"substantial agreement"**, the second
highest band. The authors describe it as "a high degree of consistency".

**Context for the presentation:** 0.76 is genuinely good *for this task*. NER
kappa is depressed by boundary disagreements that are not really disagreements
about meaning — two annotators who both see *University of Toronto* but differ on
whether *the* is included are counted as disagreeing. A document-level sentiment
task can reach 0.9; span annotation over 66 fine-grained types reaching 0.76 is
strong.

### The guideline itself

The schema was not invented from scratch. It is adapted from **FIGER** (Ling &
Weld, 2012), which defines 112 entity tags, and was then refined:

- Types with low frequency were removed, leaving 80, then 66 fine-grained types.
- Types that annotators could not reliably separate were **merged**. *Country*,
  *Province/State*, *City* and *Restrict* all became a single **GPE** type,
  because "it is difficult to distinguish these types only based on context".
- Types the schema was missing were **added**. *Person-Scholar* was created after
  pre-annotation showed many entities (mathematician, physicist, chemist,
  biologist, palaeontologist) that FIGER could not express.
- Several rounds of **pre-annotation** drove those changes before the real
  annotation began.

That refinement loop is the part worth highlighting: the guideline was tested
against annotators and changed where it failed, which is exactly what a good
annotation guideline process looks like.

### Where the text came from

The entire English Wikipedia dump. 66,000 paragraphs were selected through a
distant dictionary built from FIGER annotations, then manually denoised, so that
every fine-grained type had enough examples. Paragraphs average 61.3 tokens, and
annotation was done **at paragraph level** so annotators could see the context —
which is what lets them get *London the album* right.

---

## How entity labels are represented (the rubric asks this directly)

### What Few-NERD ships

**IO tags**: one label per token, no prefix. An entity is a run of consecutive
tokens with the same tag.

### What we use: BIO

```
tokens     Barack    Obama     visited   New         York
Few-NERD   person    person    O         location    location
BIO        B-person  I-person  O         B-location  I-location
```

- **B-** marks the **first** token of an entity
- **I-** marks a **continuation** token
- **O** means the token is not part of any entity

Converted in `scripts/build_corpus.py`, function `to_bio()`.

### Why BIO and not IO

Because IO cannot represent two entities of the same type standing next to each
other — *[New York] [Los Angeles]* looks identical to one four-token location.
BIO can. It is also what sklearn-crfsuite, seqeval and spaCy all expect, so all
three of our approaches stay directly comparable.

### Why not BIOES

BIOES adds **E-** for the last token of an entity and **S-** for single-token
entities. It gives a CRF more signal, and it is a defensible choice. We did not
use it because it doubles the number of labels (8 types × 4 prefixes + O = 33
labels instead of 17), which slows training on a machine that is already memory
constrained, and because the published Few-NERD baselines use BIO — staying with
BIO keeps our numbers comparable to theirs.

### The one cost of the conversion — say this before you are asked

Few-NERD's released data is IO, so **the information BIO needs is not always
there**. Two same-type entities that touch with no token between them have
already been merged in the source file, and no conversion can recover the
boundary. We cannot measure how often this happens, because measuring it would
require the information that is missing. It is in our limitations.

**43.2% of entities are a single token and 33.1% are two tokens**
(`reports/01_corpus.txt` §5), so most entities are short and the issue is rare —
but "rare" is not "never", and we do not claim otherwise.

---

## Our corpus in numbers

From `reports/01_corpus.txt` — quote these, do not re-derive them.

| | sentences | tokens | entities |
|---|---|---|---|
| train | 131,767 | 3,227,563 | 340,180 |
| validation | 18,824 | 463,214 | 48,741 |
| test | 37,648 | 921,118 | 96,842 |
| **total** | **188,239** | **4,611,895** | **485,763** |

**Class balance** (training entities): location 28.0%, person 22.3%,
organization 19.7%, other 9.9%, product 6.4%, building 5.2%, art 4.4%,
event 4.1%. The imbalance ratio between the largest and smallest type is
**6.8 to 1**, which is mild for a NER corpus, and every type has over 14,000
training examples.

**The imbalance that actually matters** is not between types — it is between
entity and non-entity. **21.1% of training tokens are inside an entity.** A model
that predicted O everywhere would be 78.9% accurate at the token level and find
zero entities. That is why every number we report is entity level.

---

## If you are asked "so you did no annotation at all?"

> "Correct, and deliberately — the rubric recommends against group annotation
> for this task, and we agree with the reason. What we did instead is report the
> original process in detail: 70 annotators plus 10 expert adjudicators, two
> independent passes per paragraph, expert adjudication of disagreements, a 10%
> audit of every batch with re-annotation below 95% accuracy, and Cohen's Kappa
> of 0.7644 — substantial agreement. We also documented the label scheme we
> convert to, BIO, and the one thing that conversion cannot recover."

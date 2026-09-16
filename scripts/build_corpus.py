"""Turn the Few-NERD parquet files into BIO-tagged sentences, and describe them.

Few-NERD ships one tag per token with no B-/I- prefix (an "IO" scheme). An
entity is therefore a run of consecutive tokens carrying the same tag. This
script converts those runs to the BIO scheme the rest of the pipeline uses,
because BIO is what sklearn-crfsuite, seqeval and spaCy all expect.

    tokens   Barack   Obama    visited   New    York
    Few-NERD person   person   O         location location
    BIO      B-person I-person O         B-location I-location

The known cost of IO -> BIO: two entities of the same type standing next to
each other with no token between them merge into one. Few-NERD's released data
cannot distinguish that case, so neither can we. It is reported in the
limitations rather than silently ignored.

We model the 8 coarse types, not the 66 fine types - see [4] for why.

Output: data/processed/{train,validation,test}.jsonl, reports/01_corpus.txt
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "01_corpus.txt"

SPLITS = ["train", "validation", "test"]


def coarse_names() -> list[str]:
    schema = pq.read_schema(RAW / "supervised_train.parquet")
    info = json.loads(schema.metadata[b"huggingface"].decode())
    return info["info"]["features"]["ner_tags"]["feature"]["names"]


def fine_names() -> list[str]:
    schema = pq.read_schema(RAW / "supervised_train.parquet")
    info = json.loads(schema.metadata[b"huggingface"].decode())
    return info["info"]["features"]["fine_ner_tags"]["feature"]["names"]


def to_bio(tag_ids, names: list[str]) -> list[str]:
    """IO tag ids -> BIO strings. A new entity starts wherever the tag changes."""
    out, previous = [], "O"
    for tid in tag_ids:
        name = names[int(tid)]
        if name == "O":
            out.append("O")
        elif name != previous:
            out.append(f"B-{name}")
        else:
            out.append(f"I-{name}")
        previous = name
    return out


def spans(bio: list[str]) -> list[tuple[str, int, int]]:
    """BIO -> list of (type, start, end_exclusive)."""
    found, start, label = [], None, None
    for i, tag in enumerate(bio + ["O"]):
        if tag.startswith("B-") or tag == "O" or (
                tag.startswith("I-") and label != tag[2:]):
            if label is not None:
                found.append((label, start, i))
                start, label = None, None
        if tag.startswith("B-"):
            start, label = i, tag[2:]
    return found


def main() -> None:
    missing = [s for s in SPLITS if not (RAW / f"supervised_{s}.parquet").exists()]
    if missing:
        raise SystemExit(f"missing {missing} in data/raw - see README for the "
                         "download command")

    names = coarse_names()
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("[1] SOURCE")
    say("    dataset     Few-NERD (Ding et al., ACL-IJCNLP 2021)")
    say("    paper       'Few-NERD: A Few-shot Named Entity Recognition Dataset'")
    say("    setting     SUPERVISED  (all sentences, standard train/dev/test)")
    say("    licence     CC BY-SA 4.0 - reuse and redistribution with attribution")
    say("    built from  English Wikipedia, which is CC BY-SA 4.0 itself")
    say("    obtained    HuggingFace mirror DFKI-SLT/few-nerd, parquet format")
    say()
    say("    Few-NERD also ships INTRA and INTER splits, which hold out whole")
    say("    entity types for few-shot experiments. We use SUPERVISED because our")
    say("    question is ordinary supervised NER, not few-shot transfer.")
    say()

    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for split in SPLITS:
        df = pd.read_parquet(RAW / f"supervised_{split}.parquet")
        n_tokens = 0
        ent_counter: collections.Counter[str] = collections.Counter()
        len_counter: collections.Counter[int] = collections.Counter()
        sent_lengths = []
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for row in df.itertuples():
                bio = to_bio(row.ner_tags, names)
                ents = spans(bio)
                n_tokens += len(row.tokens)
                sent_lengths.append(len(row.tokens))
                for label, a, b in ents:
                    ent_counter[label] += 1
                    len_counter[b - a] += 1
                fh.write(json.dumps({
                    "id": int(row.id),
                    "tokens": list(row.tokens),
                    "bio": bio,
                }) + "\n")
        stats[split] = dict(
            sentences=len(df), tokens=n_tokens, entities=sum(ent_counter.values()),
            per_type=ent_counter, lengths=len_counter,
            mean_len=sum(sent_lengths) / len(sent_lengths))

    say("[2] SIZE")
    say(f"    {'split':<12} {'sentences':>11} {'tokens':>12} {'entities':>10} "
        f"{'mean sent len':>14}")
    for split in SPLITS:
        s = stats[split]
        say(f"    {split:<12} {s['sentences']:>11,} {s['tokens']:>12,} "
            f"{s['entities']:>10,} {s['mean_len']:>14.1f}")
    total_sent = sum(stats[s]["sentences"] for s in SPLITS)
    total_ent = sum(stats[s]["entities"] for s in SPLITS)
    say(f"    {'TOTAL':<12} {total_sent:>11,} "
        f"{sum(stats[s]['tokens'] for s in SPLITS):>12,} {total_ent:>10,}")
    say()
    say("    The split is the one published with the dataset. We did not re-split:")
    say("    re-splitting a corpus built from Wikipedia risks putting sentences")
    say("    from the same article on both sides, which leaks entities into test.")
    say()

    say("[3] CLASS BALANCE  (entity counts, training split)")
    train = stats["train"]
    say(f"    {'type':<16} {'entities':>10} {'share':>8} {'test':>10}")
    for label, count in train["per_type"].most_common():
        share = count / train["entities"]
        say(f"    {label:<16} {count:>10,} {share:>7.1%} "
            f"{stats['test']['per_type'][label]:>10,}")
    say()
    biggest = train["per_type"].most_common(1)[0]
    smallest = train["per_type"].most_common()[-1]
    say(f"    Most common type is {biggest[0]} ({biggest[1]:,}), least common is "
        f"{smallest[0]} ({smallest[1]:,}).")
    say(f"    The imbalance ratio is {biggest[1] / smallest[1]:.1f} to 1, which is "
        "mild as NER corpora go.")
    say("    Every type has tens of thousands of examples, so no type is starved.")
    say()
    entity_tokens = sum(length * count for length, count in train["lengths"].items())
    say("    The real imbalance is not between the types, it is between entity and")
    say(f"    non-entity: {entity_tokens:,} of {train['tokens']:,} training tokens "
        f"are inside an entity, or {entity_tokens / train['tokens']:.1%}.")
    say(f"    A model that predicted O everywhere would be "
        f"{1 - entity_tokens / train['tokens']:.1%} accurate at the token")
    say("    level and would find zero entities. That is why we report entity-level")
    say("    precision, recall and F1 and never token accuracy.")
    say()

    say("[4] WHY 8 COARSE TYPES AND NOT 66 FINE TYPES")
    say(f"    Few-NERD labels {len(fine_names()) - 1} fine-grained types nested "
        "under the 8 coarse ones")
    say("    (person-actor, person-politician, building-airport, and so on).")
    say("    We model the 8 coarse types for three reasons:")
    say("      - The rubric requires a pre-trained spaCy model as one of the three")
    say("        approaches. spaCy's English model predicts 18 OntoNotes types; it")
    say("        cannot produce 66 Few-NERD types, so a 66-way comparison would be")
    say("        impossible for one of the three required approaches.")
    say("      - A gazetteer over 66 types would be 66 lists competing for the same")
    say("        surface forms, and the ambiguity would dominate the result.")
    say("      - Entity-level error analysis over 66 types is a 66x66 confusion")
    say("        matrix that nobody can read in a presentation.")
    say("    The fine labels are kept in the raw files, so the choice is reversible.")
    say()

    say("[5] ENTITY LENGTH  (how many tokens an entity spans, training split)")
    total = sum(train["lengths"].values())
    for length in sorted(train["lengths"])[:6]:
        count = train["lengths"][length]
        say(f"    {length} token(s)   {count:>9,}  {count / total:>6.1%}")
        longer = sum(c for l, c in train["lengths"].items() if l > 6)
    say(f"    7 or more    {longer:>9,}  {longer / total:>6.1%}")
    say("    Multi-token entities are the majority, which is why the evaluation")
    say("    has to be entity-level with exact span matching, not token-level.")
    say()

    say("[6] LABEL REPRESENTATION")
    say("    Few-NERD releases IO tags: one label per token, no B- or I- prefix.")
    say("    We convert to BIO, where B- marks the first token of an entity and")
    say("    I- marks continuation. BIO is what sklearn-crfsuite, seqeval and")
    say("    spaCy all use, so the three approaches stay comparable.")
    say("    We did not use BIOES (which adds E- for the last token and S- for")
    say("    single-token entities). BIOES gives a CRF more signal but doubles the")
    say("    label count, and the published Few-NERD baselines use BIO.")
    say()
    say("    Cost of the conversion: two entities of the same type that touch with")
    say("    no token in between merge into one. The released data does not mark")
    say("    that boundary, so it is unrecoverable. See the limitations.")
    say()

    say("[7] LEGAL AND ETHICAL NOTES")
    say("    - Few-NERD is released under CC BY-SA 4.0. We may use, modify and")
    say("      redistribute it as long as we credit the authors and share alike.")
    say("      The paper is cited in the references slide.")
    say("    - The underlying text is English Wikipedia, which is publicly")
    say("      available and licensed CC BY-SA 4.0, so there is no scraping and no")
    say("      terms of service issue.")
    say("    - Personal data: the corpus contains person names by design - it is a")
    say("      named entity dataset. Those names are public figures already")
    say("      documented on Wikipedia, not private individuals, and they appear in")
    say("      exactly the form Wikipedia published them.")
    say("    - We therefore do not anonymise. Anonymising a NER corpus would")
    say("      destroy the labels we are trying to predict. We do not link entities")
    say("      across sentences, build profiles of any person, or add any")
    say("      information that was not already in the source text.")
    say("    - Results are reported at the type level. No claim is made about any")
    say("      individual named in the data.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {(OUT / 'train.jsonl').relative_to(ROOT)} and 2 more")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

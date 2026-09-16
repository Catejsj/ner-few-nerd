"""Shared helpers: loading, BIO <-> spans, and entity-level scoring.

Kept in one place so all three approaches are scored by identical code. If the
gazetteer and the CRF were scored by two different functions, any difference
between them could be the scorer rather than the model.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

TYPES = ["art", "building", "event", "location", "organization", "other",
         "person", "product"]


def load(name: str) -> list[dict]:
    path = PROC / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(ROOT)} - run the earlier "
                         "scripts first")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def to_spans(bio: list[str]) -> set[tuple[str, int, int]]:
    """BIO tags -> {(type, start, end_exclusive)}.

    An I- tag whose type does not match the open entity starts a new entity
    rather than being dropped; predictions from a model are not guaranteed to
    be well-formed BIO and silently discarding them would flatter the model.
    """
    out: set[tuple[str, int, int]] = set()
    start, label = None, None
    for i, tag in enumerate(list(bio) + ["O"]):
        kind, _, this = tag.partition("-")
        if label is not None and (kind in {"B", "O"} or this != label):
            out.add((label, start, i))
            start, label = None, None
        if kind == "B" or (kind == "I" and label is None):
            start, label = i, this
    return out


def score(gold_bio: list[list[str]], pred_bio: list[list[str]]) -> dict:
    """Entity-level precision / recall / F1 with EXACT span matching.

    A prediction counts as correct only if the type, the start token and the
    end token all match the gold entity. A predicted span that overlaps a gold
    entity but starts one token early is a false positive AND leaves the gold
    entity as a false negative - it is punished twice, which is the standard
    CoNLL convention and the strictest reasonable reading.
    """
    per_type: dict[str, dict[str, int]] = {
        t: {"tp": 0, "fp": 0, "fn": 0} for t in TYPES}

    for gold, pred in zip(gold_bio, pred_bio):
        g, p = to_spans(gold), to_spans(pred)
        for label, _, _ in g & p:
            per_type.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        for label, _, _ in p - g:
            per_type.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        for label, _, _ in g - p:
            per_type.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

    def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return precision, recall, f1

    result = {"per_type": {}, "support": {}}
    for label, counts in per_type.items():
        precision, recall, f1 = prf(**counts)
        result["per_type"][label] = dict(
            precision=precision, recall=recall, f1=f1, **counts)
        result["support"][label] = counts["tp"] + counts["fn"]

    tp = sum(c["tp"] for c in per_type.values())
    fp = sum(c["fp"] for c in per_type.values())
    fn = sum(c["fn"] for c in per_type.values())
    precision, recall, f1 = prf(tp, fp, fn)
    result["micro"] = dict(precision=precision, recall=recall, f1=f1,
                           tp=tp, fp=fp, fn=fn)
    present = [t for t in per_type if result["support"][t] > 0]
    result["macro"] = dict(
        precision=sum(result["per_type"][t]["precision"] for t in present) / len(present),
        recall=sum(result["per_type"][t]["recall"] for t in present) / len(present),
        f1=sum(result["per_type"][t]["f1"] for t in present) / len(present),
    )
    return result


def report_table(result: dict, say) -> None:
    say(f"    {'type':<14} {'P':>7} {'R':>7} {'F1':>7} {'support':>9} "
        f"{'TP':>7} {'FP':>7} {'FN':>7}")
    for label in sorted(result["per_type"],
                        key=lambda t: -result["support"].get(t, 0)):
        row = result["per_type"][label]
        if result["support"][label] == 0 and row["fp"] == 0:
            continue
        say(f"    {label:<14} {row['precision']:>7.3f} {row['recall']:>7.3f} "
            f"{row['f1']:>7.3f} {result['support'][label]:>9,} "
            f"{row['tp']:>7,} {row['fp']:>7,} {row['fn']:>7,}")
    m, M = result["micro"], result["macro"]
    say(f"    {'micro avg':<14} {m['precision']:>7.3f} {m['recall']:>7.3f} "
        f"{m['f1']:>7.3f} {sum(result['support'].values()):>9,}")
    say(f"    {'macro avg':<14} {M['precision']:>7.3f} {M['recall']:>7.3f} "
        f"{M['f1']:>7.3f}")


def save_predictions(name: str, pred_bio: list[list[str]]) -> Path:
    path = PROC / f"pred_{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for bio in pred_bio:
            fh.write(json.dumps(bio) + "\n")
    return path


def load_predictions(name: str) -> list[list[str]]:
    path = PROC / f"pred_{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing predictions for {name} - run "
                         f"scripts/approach_{name}.py first")
    return [json.loads(line) for line in path.open(encoding="utf-8")]

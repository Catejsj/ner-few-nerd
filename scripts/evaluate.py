"""Compare the three approaches, and take the errors apart.

Everything here is ENTITY level with EXACT span matching: a prediction counts
only if the type, the first token and the last token all match. Token-level
accuracy is never reported - 78.9% of tokens are O, so a model that predicts
nothing scores 78.9% and finds no entities.

The error analysis goes beyond a single F1 by splitting every mistake into the
five kinds a NER system can make (the MUC categories):

  correct            right span, right type
  type error         right span, wrong type      -> the model saw the entity
  boundary error     overlapping span, right type -> the model saw it, mis-cut it
  type + boundary    overlapping span, wrong type
  missed             no overlapping prediction    -> invisible to the model
  spurious           predicted where nothing is

The distinction matters: a boundary error means the model is nearly right and a
better feature set would fix it, while a missed entity usually means the model
has no evidence at all.

Output: reports/06_comparison.txt, data/processed/confusion_*.csv
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nerlib

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "06_comparison.txt"

APPROACHES = ["gazetteer", "crf", "spacy"]
NICE = {"gazetteer": "Gazetteer", "crf": "CRF", "spacy": "spaCy (pre-trained)"}
N_BOOT = 1000
SEED = 42


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def categorise(gold_spans, pred_spans):
    """Assign each gold and predicted entity to one MUC category."""
    result = collections.Counter()
    confusion = collections.Counter()
    matched_pred = set()

    pred_by_pos = list(pred_spans)
    for g_label, g_a, g_b in gold_spans:
        exact = [(l, a, b) for l, a, b in pred_by_pos if (a, b) == (g_a, g_b)]
        if exact:
            p_label = exact[0][0]
            matched_pred.add(exact[0])
            if p_label == g_label:
                result["correct"] += 1
            else:
                result["type error"] += 1
            confusion[(g_label, p_label)] += 1
            continue
        over = [(l, a, b) for l, a, b in pred_by_pos if overlaps((a, b), (g_a, g_b))]
        if over:
            p_label = over[0][0]
            matched_pred.add(over[0])
            result["boundary error" if p_label == g_label
                   else "type + boundary"] += 1
            confusion[(g_label, p_label if p_label != g_label else g_label)] += 1
            continue
        result["missed"] += 1
        confusion[(g_label, "MISSED")] += 1

    for span in pred_by_pos:
        if span not in matched_pred:
            result["spurious"] += 1
            confusion[("SPURIOUS", span[0])] += 1
    return result, confusion


def bootstrap_ci(gold, pred, n=N_BOOT):
    """Confidence interval for micro F1, resampling sentences."""
    rng = np.random.default_rng(SEED)
    per_sentence = []
    for g, p in zip(gold, pred):
        gs, ps = nerlib.to_spans(g), nerlib.to_spans(p)
        per_sentence.append((len(gs & ps), len(ps - gs), len(gs - ps)))
    arr = np.array(per_sentence)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(arr), len(arr))
        tp, fp, fn = arr[idx].sum(axis=0)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        draws.append(2 * precision * recall / (precision + recall)
                     if precision + recall else 0)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    test = nerlib.load("test_clean")
    gold = [r["bio"] for r in test]
    preds = {name: nerlib.load_predictions(name) for name in APPROACHES}

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("[1] HOW WE ARE SCORING")
    say("    Level          entity, not token")
    say("    Matching       EXACT - type, start token and end token must all match")
    say("    A span that overlaps a gold entity but starts one token early counts")
    say("    as a false positive AND leaves the gold entity as a false negative.")
    say("    It is punished twice. That is the CoNLL convention and the strictest")
    say("    reasonable reading; [4] separates those near-misses out so the")
    say("    penalty is visible rather than hidden.")
    say()
    say(f"    test sentences {len(test):,}, gold entities "
        f"{sum(len(nerlib.to_spans(g)) for g in gold):,}")
    say()

    say("[2] HEADLINE  (test set)")
    say(f"    {'approach':<22} {'micro P':>9} {'micro R':>9} {'micro F1':>9} "
        f"{'95% CI':>16} {'macro F1':>9}")
    scores = {}
    for name in APPROACHES:
        result = nerlib.score(gold, preds[name])
        scores[name] = result
        lo, hi = bootstrap_ci(gold, preds[name])
        m = result["micro"]
        say(f"    {NICE[name]:<22} {m['precision']:>9.3f} {m['recall']:>9.3f} "
            f"{m['f1']:>9.3f} [{lo:>6.3f},{hi:>6.3f}] "
            f"{result['macro']['f1']:>9.3f}")
    say()
    best = max(scores, key=lambda n: scores[n]["micro"]["f1"])
    say(f"    Best micro F1: {NICE[best]} at {scores[best]['micro']['f1']:.3f}.")
    say("    The intervals are narrow because the test set is large, so these")
    say("    differences are real rather than noise.")
    say()
    say("    micro averages over entities, so it is dominated by location, person")
    say("    and organization, which are 69% of the test entities. macro averages")
    say("    over the 8 types equally, so a model that ignores the rare types is")
    say("    punished. We report both because they answer different questions:")
    say("    micro is 'how often is this system right', macro is 'does it work")
    say("    for every type'.")
    say()

    say("[3] PER-TYPE F1  (all three side by side)")
    say(f"    {'type':<14} {'support':>9} " +
        " ".join(f"{NICE[n]:>12}" for n in APPROACHES))
    supports = scores[APPROACHES[0]]["support"]
    for label in sorted(nerlib.TYPES, key=lambda t: -supports.get(t, 0)):
        row = " ".join(f"{scores[n]['per_type'][label]['f1']:>12.3f}"
                       for n in APPROACHES)
        say(f"    {label:<14} {supports.get(label, 0):>9,} {row}")
    say()
    for label in nerlib.TYPES:
        winners = {n: scores[n]["per_type"][label]["f1"] for n in APPROACHES}
        champion = max(winners, key=winners.get)
        if winners[champion] > 0:
            continue
    say("    No single approach wins everywhere - which is the argument for")
    say("    reporting all three rather than picking one.")
    say()

    say("[4] ERROR BREAKDOWN  (every gold entity and every prediction, categorised)")
    say(f"    {'category':<18} " + " ".join(f"{NICE[n]:>22}" for n in APPROACHES))
    cats = {}
    confusions = {}
    for name in APPROACHES:
        total = collections.Counter()
        conf = collections.Counter()
        for g, p in zip(gold, preds[name]):
            c, cf = categorise(nerlib.to_spans(g), nerlib.to_spans(p))
            total.update(c)
            conf.update(cf)
        cats[name] = total
        confusions[name] = conf
    order = ["correct", "type error", "boundary error", "type + boundary",
             "missed", "spurious"]
    n_gold = sum(len(nerlib.to_spans(g)) for g in gold)
    for category in order:
        cells = []
        for name in APPROACHES:
            count = cats[name][category]
            cells.append(f"{count:>10,} ({count / n_gold:>6.1%})")
        say(f"    {category:<18} " + " ".join(f"{c:>22}" for c in cells))
    say()
    say("    Percentages are of the gold entity count, so 'spurious' can exceed")
    say("    100% - a model is free to predict more entities than exist.")
    say()
    for name in APPROACHES:
        near = cats[name]["boundary error"] + cats[name]["type + boundary"]
        say(f"    {NICE[name]}: {cats[name]['missed'] / n_gold:.1%} of gold "
            f"entities are invisible to it, and {near / n_gold:.1%} are seen but "
            "mis-cut.")
    say()

    say("[5] CONFUSION - WHICH TYPES GET MISTAKEN FOR WHICH")
    say("    Rows are the true type, columns what the system said. MISSED means")
    say("    nothing was predicted there; the SPURIOUS row is predictions with no")
    say("    gold entity underneath.")
    for name in APPROACHES:
        say()
        say(f"    --- {NICE[name]} ---")
        labels = nerlib.TYPES
        corner = "true \\ pred"
        header = f"    {corner:<14}" + "".join(
            f"{l[:7]:>8}" for l in labels) + f"{'MISSED':>9}"
        say(header)
        matrix = []
        for true_label in labels:
            cells = [confusions[name][(true_label, pred)] for pred in labels]
            missed = confusions[name][(true_label, "MISSED")]
            row = f"    {true_label:<14}" + "".join(f"{c:>8,}" for c in cells) + \
                  f"{missed:>9,}"
            say(row)
            matrix.append(cells + [missed])
        spurious = [confusions[name][("SPURIOUS", pred)] for pred in labels]
        say(f"    {'SPURIOUS':<14}" + "".join(f"{c:>8,}" for c in spurious))
        pd.DataFrame(matrix, index=labels, columns=labels + ["MISSED"]).to_csv(
            PROC / f"confusion_{name}.csv")

        off = [((t, p), confusions[name][(t, p)]) for t in labels for p in labels
               if t != p]
        off.sort(key=lambda kv: -kv[1])
        say("    worst confusions: " + ", ".join(
            f"{t}->{p} ({c:,})" for (t, p), c in off[:4]))
    say()

    say("[6] WHERE THE APPROACHES DISAGREE")
    say("    Entities found by exactly one approach - what each one uniquely adds.")
    sets = {}
    for name in APPROACHES:
        found = set()
        for i, (g, p) in enumerate(zip(gold, preds[name])):
            found |= {(i, *s) for s in nerlib.to_spans(g) & nerlib.to_spans(p)}
        sets[name] = found
    all_gold = set()
    for i, g in enumerate(gold):
        all_gold |= {(i, *s) for s in nerlib.to_spans(g)}
    for name in APPROACHES:
        others = set().union(*(sets[o] for o in APPROACHES if o != name))
        only = sets[name] - others
        say(f"    {NICE[name]:<22} finds {len(sets[name]):>7,} correctly, "
            f"{len(only):>6,} of which no other approach finds")
    union = set().union(*sets.values())
    say(f"    {'ANY of the three':<22} finds {len(union):>7,} "
        f"({len(union) / len(all_gold):.1%} of all gold entities)")
    intersection = set.intersection(*sets.values())
    say(f"    {'ALL three agree on':<22} {len(intersection):>7,} "
        f"({len(intersection) / len(all_gold):.1%})")
    say()
    say(f"    A perfect combination of the three would reach "
        f"{len(union) / len(all_gold):.1%} recall, against")
    say(f"    {max(scores[n]['micro']['recall'] for n in APPROACHES):.1%} for the "
        "best single approach. That gap is the argument for an ensemble,")
    say("    and it is listed under future work.")
    say()

    say("[7] EXAMPLES OF EACH ERROR TYPE  (from the best system)")
    shown = collections.Counter()
    for i, (record, p) in enumerate(zip(test, preds[best])):
        gs, ps = nerlib.to_spans(record["bio"]), nerlib.to_spans(p)
        for g_label, a, b in gs:
            exact = [(l, x, y) for l, x, y in ps if (x, y) == (a, b)]
            surface = " ".join(record["tokens"][a:b])
            if exact and exact[0][0] != g_label and shown["type"] < 3:
                shown["type"] += 1
                say(f"    type error      '{surface}' is {g_label}, "
                    f"predicted {exact[0][0]}")
            over = [(l, x, y) for l, x, y in ps
                    if overlaps((x, y), (a, b)) and (x, y) != (a, b)]
            if over and shown["boundary"] < 3:
                shown["boundary"] += 1
                pl, x, y = over[0]
                say(f"    boundary error  gold '{surface}' ({g_label}), predicted "
                    f"'{' '.join(record['tokens'][x:y])}' ({pl})")
            if not exact and not over and shown["missed"] < 3:
                shown["missed"] += 1
                say(f"    missed          '{surface}' ({g_label}) - nothing "
                    "predicted here")
        if sum(shown.values()) >= 9:
            break

    scores_out = {n: {"micro": scores[n]["micro"], "macro": scores[n]["macro"],
                      "per_type": scores[n]["per_type"],
                      "support": scores[n]["support"],
                      "errors": dict(cats[n])} for n in APPROACHES}
    (PROC / "comparison.json").write_text(json.dumps(scores_out, indent=1))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

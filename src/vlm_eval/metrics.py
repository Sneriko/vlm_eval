from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass
class BowScores:
    precision: float
    recall: float
    f1: float


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def bow_scores(reference: str, prediction: str) -> BowScores:
    """Bag-of-words precision, recall, and F1 with multiset token counts."""
    ref_tokens = Counter(tokenize(reference))
    pred_tokens = Counter(tokenize(prediction))

    if not pred_tokens and not ref_tokens:
        return BowScores(precision=1.0, recall=1.0, f1=1.0)
    if not pred_tokens:
        return BowScores(precision=0.0, recall=0.0, f1=0.0)

    overlap = sum(min(pred_tokens[token], ref_tokens[token]) for token in pred_tokens)
    pred_total = sum(pred_tokens.values())
    ref_total = sum(ref_tokens.values())

    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return BowScores(precision=precision, recall=recall, f1=f1)

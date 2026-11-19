"""Run BEFORE releasing weights: prompt with the start of training texts, check for verbatim continuation."""
from __future__ import annotations

import re


def _ngrams(text: str, n: int) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def verbatim_overlap(generated: str, reference: str, n: int = 8) -> float:
    """Fraction of the reference's n-grams that appear in the generation."""
    ref = _ngrams(reference, n)
    return len(ref & _ngrams(generated, n)) / len(ref) if ref else 0.0


def prefix_probe(texts: list[str], generate, prefix_words: int = 15, n: int = 8) -> dict:
    """`generate(prefix) -> continuation`. Returns the share of texts regurgitated and the worst offenders."""
    scores = []
    for t in texts:
        w = t.split()
        if len(w) <= prefix_words + n:
            continue
        cont = generate(" ".join(w[:prefix_words]))
        scores.append((verbatim_overlap(cont, " ".join(w[prefix_words:]), n), t[:60]))
    flagged = [s for s in scores if s[0] > 0.2]
    return {"tested": len(scores), "flagged": len(flagged), "worst": sorted(scores, reverse=True)[:5]}

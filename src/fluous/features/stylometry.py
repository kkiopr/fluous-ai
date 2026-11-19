"""Measurable voice features. Used for diagnostics and candidate filtering, NEVER as a training reward."""
from __future__ import annotations

import re
from statistics import mean, pstdev

import numpy as np

FUNCTION_WORDS = (
    "the of and to a in that is was it for on with as i you he she we they but at by this from "
    "or an be have not are so if just like really maybe actually would could will can do my me "
    "your out up about what when there their then than because too very"
).split()
HEDGES = ("maybe", "perhaps", "kind of", "sort of", "i think", "i guess", "probably", "might", "somewhat")
APOLOGIES = ("sorry", "apologize", "apologies", "my bad", "forgive")
_TOKEN = re.compile(r"[A-Za-z']+")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s.strip()]


def mattr(tokens: list[str], window: int = 50) -> float:
    if len(tokens) < window:
        return len(set(tokens)) / max(1, len(tokens))
    return float(np.mean([len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]))


def lag1_autocorr(xs: list[float]) -> float:
    if len(xs) < 3 or np.std(xs) == 0:
        return 0.0
    return float(np.corrcoef(xs[:-1], xs[1:])[0, 1])


def featurize(text: str) -> dict[str, float]:
    low = text.lower()
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    n = max(1, len(tokens))
    lens = [len(_TOKEN.findall(s)) for s in _sentences(text)] or [0]
    letters = [c for c in text if c.isalpha()]
    f = {
        "sent_len_mean": mean(lens),
        "sent_len_std": pstdev(lens) if len(lens) > 1 else 0.0,
        "sent_len_lag1": lag1_autocorr(lens),
        "mattr": mattr(tokens),
        "words": float(len(tokens)),
        "comma_rate": text.count(",") / n,
        "period_rate": text.count(".") / n,
        "excl_rate": text.count("!") / n,
        "question_rate": text.count("?") / n,
        "ellipsis_rate": text.count("...") / n,
        "emoji_rate": len(re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", text)) / n,
        "lower_start_rate": float(np.mean([s.strip()[0].islower() for s in _sentences(text) if s.strip()] or [0])),
        "caps_ratio": sum(c.isupper() for c in letters) / max(1, len(letters)),
        "elongation_rate": len(re.findall(r"(\w)\1{2,}", low)) / n,
        "hedge_rate": sum(low.count(h) for h in HEDGES) / n,
        "apology_rate": sum(low.count(a) for a in APOLOGIES) / n,
    }
    for w in FUNCTION_WORDS:
        f[f"fw_{w}"] = tokens.count(w) / n
    return f


def fit_target(texts: list[str]) -> dict[str, tuple[float, float]]:
    """Per-feature (mean, std) over an author's texts, with a std floor."""
    feats = [featurize(t) for t in texts]
    keys = feats[0].keys()
    return {k: (float(np.mean([f[k] for f in feats])), max(float(np.std([f[k] for f in feats])), 1e-3)) for k in keys}


def style_distance(text: str, target: dict[str, tuple[float, float]], weights: dict[str, float] | None = None) -> float:
    """D(y) = sum_j w_j * (how many stds feature j sits OUTSIDE the target range)^2. Diagnostic only."""
    f = featurize(text)
    d = 0.0
    for k, (mu, sd) in target.items():
        z = max(0.0, abs(f[k] - mu) / sd - 1.0)
        d += (weights or {}).get(k, 1.0) * z * z
    return d

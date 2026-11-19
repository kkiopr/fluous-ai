"""Blind A/B sheets + analysis. Judges should be strangers AND people who know the writer.
Question to ask: "Which of these was written by <writer>?" or "Which sounds more like them?"
Report sample sizes and confidence intervals, not just win rates."""
from __future__ import annotations

import csv
import json
import math
import random


def make_sheet(items: list[dict], sheet_path: str, key_path: str, seed: int = 0):
    """items: {"id", "prompt", "model_text", "other_text"}. Sides are randomized."""
    rng = random.Random(seed)
    key = {}
    with open(sheet_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "A", "B", "choice"])
        for it in items:
            model_is_a = rng.random() < 0.5
            a, b = (it["model_text"], it["other_text"]) if model_is_a else (it["other_text"], it["model_text"])
            key[it["id"]] = "A" if model_is_a else "B"
            w.writerow([it["id"], it["prompt"], a, b, ""])
    json.dump(key, open(key_path, "w"))


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def analyze(responses_csv: str, key_path: str) -> dict:
    key = json.load(open(key_path))
    wins = n = 0
    for row in csv.DictReader(open(responses_csv)):
        if row["choice"] in ("A", "B") and row["id"] in key:
            n += 1
            wins += row["choice"] == key[row["id"]]
    lo, hi = wilson(wins, n)
    return {"n": n, "model_win_rate": wins / n if n else None, "ci95": (lo, hi),
            "significant_vs_50pct": lo > 0.5 or hi < 0.5}

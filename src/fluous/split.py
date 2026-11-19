"""Eval split. Run this BEFORE cards, augmentation, or training touch the data.
Writes data/processed/EVAL_LOCKED.json and refuses to re-split unless --force."""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .clean import _shingles
from .schema import Record


def split_records(records: list[Record], eval_frac=0.12, eval_max_per_author=200,
                  heldout_authors=(), seed=13) -> list[Record]:
    rng = random.Random(seed)
    by_author = defaultdict(list)
    for r in records:
        by_author[r.author_id].append(r)
    out = []
    for author, recs in sorted(by_author.items()):
        recs = sorted(recs, key=lambda r: r.id)
        if author in heldout_authors:
            for r in recs:
                r.split = "heldout_author"
            out += recs
            continue
        rng.shuffle(recs)
        n_eval = min(eval_max_per_author, max(1, math.ceil(eval_frac * len(recs))))
        for i, r in enumerate(recs):
            r.split = "eval" if i < n_eval else "train"
        out += recs
    return out


def leak_report(records: list[Record], threshold=0.6) -> list[tuple[str, str, float]]:
    """Near-duplicates between eval/heldout and train. Should be empty."""
    train = [(r.id, _shingles(r.text)) for r in records if r.split == "train"]
    leaks = []
    for r in records:
        if r.split in ("eval", "heldout_author"):
            s = _shingles(r.text)
            for tid, t in train:
                j = len(s & t) / max(1, len(s | t))
                if j >= threshold:
                    leaks.append((r.id, tid, round(j, 2)))
    return leaks


def lock(records: list[Record], path="data/processed/EVAL_LOCKED.json", force=False):
    p = Path(path)
    if p.exists() and not force:
        raise SystemExit(f"{p} exists: eval split already locked. Use --force only if you accept re-splitting.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"eval_ids": [r.id for r in records if r.split != "train"]}))

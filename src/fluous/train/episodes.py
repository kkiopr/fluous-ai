"""Episodic training examples: k samples from an author -> a DIFFERENT text from that author."""
from __future__ import annotations

import random
from collections import defaultdict

from ..schema import iter_jsonl
from .prompt import build_prompt, sample_voice


def load_pools(path: str, split: str = "train", per_author_cap: int | None = None, seed: int = 13):
    rng = random.Random(seed)
    pools = defaultdict(list)
    for d in iter_jsonl(path):
        if d.get("split") == split and d.get("card"):
            pools[d["author_id"]].append(d)
    if per_author_cap:
        for a in pools:
            rng.shuffle(pools[a])
            pools[a] = pools[a][:per_author_cap]
    return pools


def make_episodes(pools: dict, ks: list[int], episodes_per_record: int, seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    eps = []
    for author, pool in pools.items():
        if len(pool) < 2:
            continue
        for target in pool:
            for _ in range(episodes_per_record):
                k = min(rng.choice(ks), len(pool) - 1)
                prompt = build_prompt(sample_voice(pool, target, k, rng), target["card"])
                eps.append({"author_id": author, "id": target["id"], "prompt": prompt, "target": target["text"]})
    rng.shuffle(eps)
    return eps

"""Build DPO preference pairs. chosen = the author's real text. Rejected comes from three kinds:
  fact_corrupt : local, made by corrupting numbers/negations in the real text (also grows the stress set)
  baseline     : frontier-model output with the author's samples in the prompt (supply as jsonl)
  wrong_voice  : same card written in ANOTHER author's voice (supply as jsonl)
External rejects jsonl format: {"id": <record id>, "text": "..."}   one file per kind."""
from __future__ import annotations

import argparse
import json
import random
import re

from ..schema import iter_jsonl
from .episodes import load_pools
from .prompt import build_prompt, sample_voice

_NUM = re.compile(r"\d+(?:\.\d+)?")


def corrupt_facts(text: str, rng: random.Random) -> str | None:
    nums = list(_NUM.finditer(text))
    if nums:
        m = rng.choice(nums)
        v = float(m.group(0))
        new = v + rng.choice([-2, -1, 1, 2, 10]) if v == int(v) else v * 1.1
        new = max(0, new)
        s = str(int(new)) if v == int(v) else f"{new:.1f}"
        if s != m.group(0):
            return text[:m.start()] + s + text[m.end():]
    if re.search(r"\bnot\b", text):
        return re.sub(r"\bnot\b ?", "", text, count=1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/carded.jsonl")
    ap.add_argument("--out", default="data/processed/prefs.jsonl")
    ap.add_argument("--rejects", nargs="*", default=[], help="kind=path.jsonl, e.g. baseline=data/rejects/baseline.jsonl")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    external = {}
    for spec in a.rejects:
        kind, path = spec.split("=", 1)
        for d in iter_jsonl(path):
            external.setdefault(d["id"], []).append((kind, d["text"]))

    pools = load_pools(a.data, "train")
    n = 0
    with open(a.out, "w") as f:
        for pool in pools.values():
            for target in pool:
                prompt = build_prompt(sample_voice(pool, target, min(a.k, len(pool) - 1), rng), target["card"]) if len(pool) > 1 else None
                if prompt is None:
                    continue
                rejects = list(external.get(target["id"], []))
                bad = corrupt_facts(target["text"], rng)
                if bad:
                    rejects.append(("fact_corrupt", bad))
                for kind, rej in rejects:
                    f.write(json.dumps({"prompt": prompt, "chosen": target["text"], "rejected": rej, "kind": kind}) + "\n")
                    n += 1
    print(f"wrote {n} preference pairs to {a.out}")


if __name__ == "__main__":
    main()

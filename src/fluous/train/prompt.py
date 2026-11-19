"""One prompt format shared by SFT, DPO, baselines, and inference.
Voice conditioning v0 = a few of the author's own texts in context. Compare against a learned
voice embedding/soft prefix later (see docs/PLAN.md)."""
from __future__ import annotations

import random

TEMPLATE = """### Voice samples (same writer)
{samples}

### Situation
{situation}

### Content card (keep every fact exactly)
Intent: {intent}
Facts:
{facts}

### Write the message in this writer's voice
"""


def build_prompt(samples: list[str], card: dict) -> str:
    s = "\n---\n".join(samples) if samples else "(none)"
    facts = "\n".join(f"- {f}" for f in card.get("facts", []))
    return TEMPLATE.format(samples=s, situation=card.get("situation", "other"),
                           intent=card.get("intent", ""), facts=facts)


def sample_voice(pool: list[dict], target: dict, k: int, rng: random.Random) -> list[str]:
    """Pick k texts from the same author, excluding the target, preferring a different situation
    so the model must extract author-level traits instead of copying situation."""
    others = [r for r in pool if r["id"] != target["id"]]
    diff = [r for r in others if (r.get("card") or {}).get("situation") != (target.get("card") or {}).get("situation")]
    rng.shuffle(diff)
    rng.shuffle(others)
    chosen = (diff + [r for r in others if r not in diff])[:k]
    return [r["text"] for r in chosen]

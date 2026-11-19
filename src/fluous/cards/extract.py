"""Meaning-card extraction: text -> {situation, intent, facts, tone_notes}.

PRIVACY: only send cleaned, scrubbed, sent-only text to any hosted LLM. Never raw threads.
`complete` is any callable str -> str, so you can swap providers or use a local model.
Hand-check ~100 cards: bad cards poison training.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable

SITUATIONS = [
    "running_late", "canceling", "apologizing", "excited", "formal_request",
    "follow_up", "declining", "thanking", "quote_or_pricing", "review_reply",
    "scheduling", "casual_chat", "other",
]

PROMPT = """You extract a neutral "meaning card" from a message. Do NOT imitate its style.
Return ONLY JSON with keys:
  "situation": one of {situations}
  "intent": one short neutral sentence (what the writer wants to achieve)
  "facts": list of short neutral strings; EVERY number, name, date, time, price, commitment,
           negation and condition in the message must appear here, verbatim where possible
  "tone_notes": list of up to 3 words describing emotional stance (not style)

Message:
\"\"\"{text}\"\"\"
"""


def make_anthropic_complete(model: str, max_tokens: int = 600) -> Callable[[str], str]:
    import anthropic  # pip install anthropic; reads ANTHROPIC_API_KEY
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(prompt: str) -> str:
        r = client.messages.create(model=model, max_tokens=max_tokens,
                                   messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    return complete


def parse_card(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        card = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(card.get("facts"), list):
        return None
    if card.get("situation") not in SITUATIONS:
        card["situation"] = "other"
    return card


def extract_card(text: str, complete: Callable[[str], str]) -> dict | None:
    return parse_card(complete(PROMPT.format(situations=SITUATIONS, text=text)))

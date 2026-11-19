"""Kept / dropped / changed-or-invented. Target: retention 1.0, injection 0.

kept     -> fact present in both
dropped  -> in source, missing in output  (flag for the user)
changed  -> a number-like fact swapped for another of the same kind (FAIL)
invented -> fact in output not in source   (FAIL)
Negations and conditionals are counted, and any mismatch is flagged for human review.
"""
from __future__ import annotations

from .extract import _NUMERIC_KINDS, extract_facts, logic_counts


def check(source: str, output: str) -> dict:
    src, out = extract_facts(source), extract_facts(output)
    kept, dropped, invented, changed = [], [], [], []
    for kind in src:
        s, o = src[kind], out[kind]
        kept += [(kind, x) for x in sorted(s & o)]
        miss, extra = sorted(s - o), sorted(o - s)
        if kind in _NUMERIC_KINDS:
            pairs = min(len(miss), len(extra))
            changed += [(kind, a, b) for a, b in zip(miss[:pairs], extra[:pairs])]
            miss, extra = miss[pairs:], extra[pairs:]
        dropped += [(kind, x) for x in miss]
        invented += [(kind, x) for x in extra]
    ls, lo = logic_counts(source), logic_counts(output)
    review = [k for k in ls if ls[k] != lo[k]]
    return {
        "kept": kept, "dropped": dropped, "changed": changed, "invented": invented,
        "needs_review": review,
        "passed": not changed and not invented and not review,
        "retention_full": not dropped and not changed,
    }

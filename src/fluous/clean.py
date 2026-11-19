"""Cleaning pipeline. Order: cutoff -> strip quotes/signatures -> scrub PII -> merge chat bursts
-> min words -> near-dedupe. Eval split is cut AFTER this (see split.py) but BEFORE cards/training."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import timedelta

from .schema import Record, parse_iso

CHAT_SOURCES = {"imessage", "whatsapp", "discord"}

_MARKERS = [re.compile(p, re.I) for p in (
    r"^>", r"^on .{5,200} wrote:\s*$", r"^-{2,}\s*original message", r"^-{5,}\s*forwarded message",
    r"^from:\s.+", r"^-- ?$", r"^sent from my ", r"^get outlook for ",
)]
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")


def strip_quoted(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for i, line in enumerate(lines):
        s = line.strip()
        wrapped_on = s.lower().startswith("on ") and i + 1 < len(lines) and lines[i + 1].strip().lower().endswith("wrote:")
        if wrapped_on or any(p.match(s) for p in _MARKERS):
            break
        kept.append(line)
    return "\n".join(kept).strip()


_nlp = None


def scrub(text: str, use_spacy: bool = False, model: str = "en_core_web_sm") -> str:
    text = _URL.sub("[URL]", text)
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    if use_spacy:
        global _nlp
        if _nlp is None:
            import spacy
            _nlp = spacy.load(model)
        doc = _nlp(text)
        for ent in reversed(doc.ents):
            if ent.label_ in {"PERSON"}:
                text = text[:ent.start_char] + "[NAME]" + text[ent.end_char:]
    return text


def merge_bursts(records: list[Record], gap_minutes: int) -> list[Record]:
    groups: dict[tuple, list[Record]] = defaultdict(list)
    passthrough = []
    for r in records:
        if r.source in CHAT_SOURCES and parse_iso(r.date):
            groups[(r.author_id, r.source)].append(r)
        else:
            passthrough.append(r)
    merged = []
    gap = timedelta(minutes=gap_minutes)
    for recs in groups.values():
        recs.sort(key=lambda r: parse_iso(r.date))
        cur = recs[0]
        for r in recs[1:]:
            if parse_iso(r.date) - parse_iso(cur.date) <= gap:
                cur = Record(cur.author_id, cur.text + "\n" + r.text, cur.source, cur.consent, date=cur.date)
            else:
                merged.append(cur)
                cur = r
        merged.append(cur)
    return passthrough + merged


def _shingles(text: str, n: int = 5) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def near_dedupe(records: list[Record], threshold: float) -> list[Record]:
    """O(n^2): fine for thousands of texts. Swap for MinHash if you scale up."""
    kept, sigs = [], []
    for r in records:
        s = _shingles(r.text)
        if any(len(s & t) / max(1, len(s | t)) >= threshold for t in sigs):
            continue
        kept.append(r)
        sigs.append(s)
    return kept


def clean_records(records: list[Record], cfg: dict) -> tuple[list[Record], dict]:
    stats = {"input": len(records)}
    cutoff = parse_iso(cfg.get("cutoff_date"))
    out = []
    for r in records:
        dt = parse_iso(r.date)
        if cutoff and dt and dt > cutoff:
            continue
        if cutoff and dt is None and cfg.get("drop_unknown_dates"):
            continue
        text = scrub(strip_quoted(r.text), cfg.get("scrub_names_with_spacy", False), cfg.get("spacy_model", "en_core_web_sm"))
        if text.strip():
            out.append(Record(r.author_id, text.strip(), r.source, r.consent, date=r.date, situation=r.situation))
    stats["after_cutoff_and_scrub"] = len(out)
    out = merge_bursts(out, cfg.get("burst_gap_minutes", 10))
    stats["after_merge"] = len(out)
    out = [r for r in out if len(r.text.split()) >= cfg.get("min_words", 20)]
    stats["after_min_words"] = len(out)
    out = near_dedupe(out, cfg.get("dedupe_jaccard", 0.8))
    stats["after_dedupe"] = len(out)
    return out, stats

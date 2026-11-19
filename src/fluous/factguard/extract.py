"""Pull checkable facts out of text. Regex-first; add an LLM pass for negations/conditionals later."""
from __future__ import annotations

import re

_TIME = re.compile(r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b|\b\d{1,2}\s?(?:am|pm)\b", re.I)
_MONEY = re.compile(r"[$£€]\s?\d[\d,]*(?:\.\d+)?(?:\s?[kKmM])?")
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\b|\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b", re.I)
_DAY = re.compile(r"\b(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?\b", re.I)
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")
_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b")

NEGATIONS = {"not", "no", "never", "cannot", "without", "neither", "nor", "none"}
CONDITIONALS = {"if", "unless", "provided", "otherwise", "whether"}
_NUMERIC_KINDS = ("time", "money", "percent", "date", "number")


def _norm(kind: str, s: str) -> str:
    s = s.lower().strip()
    if kind in ("time",):
        return re.sub(r"\s", "", s)
    if kind == "date":
        return re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s).rstrip(".")
    if kind in ("money", "number", "percent"):
        return re.sub(r"[,\s]", "", s)
    return s


def extract_facts(text: str) -> dict[str, set[str]]:
    facts: dict[str, set[str]] = {k: set() for k in (*_NUMERIC_KINDS, "day", "name")}
    work = text
    for kind, pat in (("time", _TIME), ("money", _MONEY), ("percent", _PERCENT), ("date", _DATE)):
        for m in pat.finditer(work):
            facts[kind].add(_norm(kind, m.group(0)))
        work = pat.sub(" ", work)
    for m in _NUMBER.finditer(work):
        facts["number"].add(_norm("number", m.group(0)))
    for m in _DAY.finditer(text):
        facts["day"].add(m.group(0).lower()[:3])
    for m in _NAME.finditer(text):
        w = m.group(1)
        if not _DAY.fullmatch(w) and not re.fullmatch(_MONTHS, w, re.I):
            facts["name"].add(w.lower())
    return facts


def logic_counts(text: str) -> dict[str, int]:
    toks = re.findall(r"[a-z']+", text.lower())
    neg = sum(t in NEGATIONS or t.endswith("n't") for t in toks)
    cond = sum(t in CONDITIONALS for t in toks)
    return {"negations": neg, "conditionals": cond}

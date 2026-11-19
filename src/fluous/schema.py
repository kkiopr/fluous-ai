"""Record schema. Every text carries lineage so any source can be stripped later."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

# consent: how we're allowed to use the text. Never train the generator on "unconsented_public".
CONSENT_VALUES = {"self", "friend_export", "game_optin", "public_licensed", "unconsented_public"}


@dataclass
class Record:
    author_id: str
    text: str
    source: str            # gmail | imessage | whatsapp | discord | ...
    consent: str
    date: str | None = None  # ISO 8601
    situation: str | None = None
    card: dict | None = None
    split: str | None = None  # train | eval | heldout_author
    extra: dict | None = None

    def __post_init__(self):
        if self.consent not in CONSENT_VALUES:
            raise ValueError(f"unknown consent value: {self.consent}")

    @property
    def id(self) -> str:
        key = f"{self.author_id}|{self.source}|{self.date}|{self.text}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        return cls(**{k: v for k, v in d.items() if k != "id"})


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def read_jsonl(path: str | Path) -> list[Record]:
    with open(path, encoding="utf-8") as f:
        return [Record.from_dict(json.loads(line)) for line in f if line.strip()]


def write_jsonl(records: Iterable[Record], path: str | Path) -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

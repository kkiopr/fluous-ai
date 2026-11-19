"""Discord data package (messages/c*/messages.csv). The package only contains YOUR messages."""
from __future__ import annotations

import csv
import glob
import os

from ..schema import Record


def parse(package_dir: str, author_id: str, consent: str) -> list[Record]:
    out = []
    for path in glob.glob(os.path.join(package_dir, "messages", "*", "messages.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                text = (row.get("Contents") or "").strip()
                if not text:
                    continue
                ts = (row.get("Timestamp") or "").replace(" ", "T", 1) or None
                out.append(Record(author_id, text, "discord", consent, date=ts))
    return out

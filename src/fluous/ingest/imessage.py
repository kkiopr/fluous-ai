"""iMessage chat.db -> messages you sent. Copy the db first; needs Full Disk Access.

Newer macOS versions store text in `attributedBody`; the decoder below is best-effort.
If you get far fewer rows than expected, inspect that column.
"""
from __future__ import annotations

import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

from ..schema import Record

_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


def _decode_attributed(blob: bytes | None) -> str | None:
    if not blob:
        return None
    try:
        i = blob.index(b"NSString") + len(b"NSString")
        blob = blob[i + 5:]
        if blob[0] == 0x81:
            n, start = int.from_bytes(blob[1:3], "little"), 3
        else:
            n, start = blob[0], 1
        return blob[start:start + n].decode("utf-8", errors="ignore")
    except Exception:
        return None


def _to_iso(raw: int | None) -> str | None:
    if raw is None:
        return None
    seconds = raw / 1e9 if raw > 1e11 else raw  # newer versions use nanoseconds
    return (_APPLE_EPOCH + timedelta(seconds=seconds)).isoformat()


def parse(db_path: str, author_id: str, consent: str) -> list[Record]:
    with tempfile.TemporaryDirectory() as tmp:
        copy = f"{tmp}/chat.db"
        shutil.copy(db_path, copy)
        db = sqlite3.connect(copy)
        rows = db.execute(
            "SELECT date, text, attributedBody FROM message WHERE is_from_me = 1"
        ).fetchall()
        db.close()
    out = []
    for date, text, blob in rows:
        text = text or _decode_attributed(blob)
        if text and text.strip():
            out.append(Record(author_id, text.strip(), "imessage", consent, date=_to_iso(date)))
    return out

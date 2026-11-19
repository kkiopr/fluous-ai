"""Gmail Takeout mbox -> sent messages only. Extraction only; cleaning happens in clean.py."""
from __future__ import annotations

import mailbox
from email.utils import parsedate_to_datetime

from ..schema import Record


def _body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
        return ""
    payload = msg.get_payload(decode=True)
    return payload.decode(msg.get_content_charset() or "utf-8", errors="ignore") if payload else ""


def parse(path: str, author_id: str, consent: str) -> list[Record]:
    out = []
    for m in mailbox.mbox(path):
        labels = m.get("X-Gmail-Labels") or ""
        if "Sent" not in labels:
            continue
        text = _body(m).strip()
        if not text:
            continue
        try:
            date = parsedate_to_datetime(m["Date"]).isoformat()
        except Exception:
            date = None
        out.append(Record(author_id, text, "gmail", consent, date=date))
    return out

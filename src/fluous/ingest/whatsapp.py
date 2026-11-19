"""WhatsApp 'Export chat' .txt -> your messages. Handles Android and iOS line formats."""
from __future__ import annotations

import re
from datetime import datetime

from ..schema import Record

LINE = re.compile(
    r"^\[?(\d{1,2}[/.]\d{1,2}[/.]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s*(?:-\s*)?([^:]+?):\s(.*)$"
)
SKIP = {"<media omitted>", "this message was deleted", "you deleted this message"}


def _iso(d: str, t: str, dayfirst: bool) -> str | None:
    sep = "/" if "/" in d else "."
    parts = d.split(sep)
    if len(parts[2]) == 2:
        parts[2] = "20" + parts[2]
    day, month = (parts[0], parts[1]) if dayfirst else (parts[1], parts[0])
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p", "%I:%M%p"):
        try:
            tm = datetime.strptime(t.strip().upper(), fmt)
            return datetime(int(parts[2]), int(month), int(day), tm.hour, tm.minute, tm.second).isoformat()
        except ValueError:
            continue
    return None


def parse(path: str, author_id: str, consent: str, me: str, dayfirst: bool = True) -> list[Record]:
    """`me` must match your display name in the export exactly. Check `dayfirst` for your locale."""
    msgs: list[dict] = []
    last_sender = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = LINE.match(line)
            if m:
                d, t, sender, text = m.groups()
                last_sender = sender
                if sender == me and text.strip().lower() not in SKIP:
                    msgs.append({"date": _iso(d, t, dayfirst), "text": text})
            elif last_sender == me and msgs:
                msgs[-1]["text"] += "\n" + line  # continuation of a multi-line message
    return [Record(author_id, x["text"].strip(), "whatsapp", consent, date=x["date"]) for x in msgs if x["text"].strip()]

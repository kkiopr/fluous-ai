"""python -m fluous.cli {ingest,clean,split,cards} ..."""
from __future__ import annotations

import argparse
import glob
import json

import yaml

from .schema import Record, iter_jsonl, read_jsonl, write_jsonl


def cmd_ingest(a):
    if a.source == "gmail":
        from .ingest import gmail; recs = gmail.parse(a.path, a.author_id, a.consent)
    elif a.source == "imessage":
        from .ingest import imessage; recs = imessage.parse(a.path, a.author_id, a.consent)
    elif a.source == "whatsapp":
        from .ingest import whatsapp; recs = whatsapp.parse(a.path, a.author_id, a.consent, a.me, not a.monthfirst)
    elif a.source == "discord":
        from .ingest import discord; recs = discord.parse(a.path, a.author_id, a.consent)
    print(f"{len(recs)} records -> {a.out}")
    write_jsonl(recs, a.out)


def cmd_clean(a):
    from .clean import clean_records
    cfg = yaml.safe_load(open(a.config))
    recs = [r for p in a.inp for f in glob.glob(p) for r in read_jsonl(f)]
    out, stats = clean_records(recs, cfg)
    print(json.dumps(stats, indent=2))
    write_jsonl(out, a.out)


def cmd_split(a):
    from .split import leak_report, lock, split_records
    cfg = yaml.safe_load(open(a.config))["split"]
    recs = split_records(read_jsonl(a.inp), cfg["eval_frac"], cfg["eval_max_per_author"],
                         cfg["heldout_authors"], cfg["seed"])
    leaks = leak_report(recs)
    if leaks:
        raise SystemExit(f"{len(leaks)} near-duplicate eval/train leaks, e.g. {leaks[:3]}. Fix before continuing.")
    lock(recs, force=a.force)
    write_jsonl(recs, a.out)
    print({s: sum(r.split == s for r in recs) for s in ("train", "eval", "heldout_author")})


def cmd_cards(a):
    from .cards.extract import extract_card, make_anthropic_complete
    complete = make_anthropic_complete(a.model)
    out, failed = [], 0
    for r in read_jsonl(a.inp):
        if r.split == "eval" and not a.include_eval:
            out.append(r); continue  # eval texts don't need cards unless you're evaluating card->text
        card = extract_card(r.text, complete)
        failed += card is None
        r.card, r.situation = card, (card or {}).get("situation")
        out.append(r)
    write_jsonl(out, a.out)
    print(f"{failed} cards failed to parse. Hand-check ~100 before training.")


def main():
    ap = argparse.ArgumentParser(prog="fluous")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("ingest"); p.set_defaults(fn=cmd_ingest)
    p.add_argument("--source", required=True, choices=["gmail", "imessage", "whatsapp", "discord"])
    p.add_argument("--path", required=True); p.add_argument("--author-id", required=True)
    p.add_argument("--consent", default="self"); p.add_argument("--out", required=True)
    p.add_argument("--me", help="your display name in a WhatsApp export")
    p.add_argument("--monthfirst", action="store_true", help="WhatsApp dates are MM/DD (US locale)")

    p = sub.add_parser("clean"); p.set_defaults(fn=cmd_clean)
    p.add_argument("--inp", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/data.yaml")

    p = sub.add_parser("split"); p.set_defaults(fn=cmd_split)
    p.add_argument("--inp", required=True); p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/data.yaml"); p.add_argument("--force", action="store_true")

    p = sub.add_parser("cards"); p.set_defaults(fn=cmd_cards)
    p.add_argument("--inp", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", required=True, help="model name for your LLM provider")
    p.add_argument("--include-eval", action="store_true")

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

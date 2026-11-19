# fluous

Voice-conditioned rewriting with a fact guard. Small open model (3B-8B) + LoRA, trained on consented text.

## Status
Scaffold. Data pipeline, fact guard, features, and tests run. Training scripts (`train/sft.py`, `train/dpo.py`) are
written but **untested against your transformers/peft/trl versions**: expect small API fixes. Pin your versions.

## Pipeline
```
export (Takeout, WhatsApp, Discord, iMessage)      # official exports only, never scrape logged-in apps
  -> ingest   python -m fluous.cli ingest --source gmail --path X.mbox --author-id me --out data/interim/me_gmail.jsonl
  -> clean    make clean-data        # cutoff, strip quotes, scrub PII, merge chat bursts, min words, dedupe
  -> split    make split             # eval carved out FIRST; locks EVAL_LOCKED.json; fails on near-dup leaks
  -> cards    make cards             # LLM meaning cards (scrubbed text only). Hand-check ~100.
  -> sft      make sft               # episodic: k voice samples + card -> a different text by the same author
  -> prefs    python -m fluous.train.prefs --rejects baseline=data/rejects/baseline.jsonl wrong_voice=...
  -> dpo      make dpo
  -> eval     eval/blind.py (blind A/B + Wilson CI), eval/memorization.py, factguard stress tests
```

## Layout
- `src/fluous/ingest/` export parsers (sent messages only)
- `src/fluous/clean.py`, `split.py` cleaning and leak-safe splitting
- `src/fluous/cards/` meaning-card extraction (provider-agnostic `complete` callable)
- `src/fluous/features/stylometry.py` voice features and D(y), a diagnostic and never a reward
- `src/fluous/factguard/` extract + kept/dropped/changed/invented checker
- `src/fluous/train/` prompt format, episodes, SFT, preference pairs, DPO
- `src/fluous/eval/` blind tests, memorization probe

## Rules this repo enforces
1. Every record carries `author_id, source, consent, date`. Strip a source or person with one filter.
2. Eval split happens before cards/training and is locked.
3. `data/` contents are gitignored. Do not commit raw or processed text.
4. Never train the generator on `consent="unconsented_public"`.
5. Run `eval/memorization.py` before releasing any weights.

## Next
See `docs/PLAN.md`.

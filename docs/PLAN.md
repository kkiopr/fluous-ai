# Plan and gates

0. Rerun the prompt baseline at k=1,3,7. Write down the tells (what gave the baseline away) per draft.
1. Data: you + consenting friends. Hold one friend out entirely as an unseen-voice test.
2. SFT (episodic, k mixed) -> compare vs best prompt baseline, blind, with CIs. Gate: wins at k=1 and k=3.
3. DPO with rejects: baseline / wrong_voice / fact_corrupt.
4. Fact guard gate: retention 1.0, injection 0, zero flipped conditions on the stress set (grow `tests/test_factguard.py`).
5. Voice conditioning variants to ablate: text samples in context (v0) vs learned embedding/soft prefix.
6. Not yet possible with ~4 authors: voice-space blending, "find your voice" matching. Needs the game/users.

Open items: base model + license check; situation-delta correction (mu + delta) only if it helps held-out one-shot.

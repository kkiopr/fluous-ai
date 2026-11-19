import random
from fluous.eval.blind import wilson
from fluous.eval.memorization import verbatim_overlap
from fluous.features.stylometry import featurize, fit_target, style_distance
from fluous.train.prefs import corrupt_facts
from fluous.train.prompt import build_prompt, sample_voice

def test_features_and_distance():
    texts = ["omg im sooo late sorry!!! be there soon", "cant make it tonight, sorry. rain check?", "ok sounds good see u there"]
    f = featurize(texts[0])
    assert f["elongation_rate"] > 0 and f["apology_rate"] > 0
    tgt = fit_target(texts)
    formal = "Dear Sir, I regret to inform you that I shall be unable to attend the meeting. Kind regards, Alexander."
    assert style_distance(formal, tgt) > style_distance(texts[1], tgt)

def test_corrupt_changes_a_number():
    out = corrupt_facts("It costs 50 dollars at 7 pm", random.Random(1))
    assert out and out != "It costs 50 dollars at 7 pm"

def test_sample_voice_excludes_target_and_prefers_other_situation():
    pool = [{"id": str(i), "text": f"t{i}", "card": {"situation": "a" if i < 3 else "b"}} for i in range(6)]
    got = sample_voice(pool, pool[0], 2, random.Random(0))
    assert "t0" not in got and all(g in {"t3", "t4", "t5"} for g in got)
    assert "Facts:" in build_prompt(got, {"situation": "a", "intent": "x", "facts": ["f"]})

def test_wilson_and_overlap():
    lo, hi = wilson(20, 30)
    assert 0.48 < lo < 0.5 and hi < 0.85
    assert verbatim_overlap("a b c d e f g h i j", "a b c d e f g h i j", 8) == 1.0

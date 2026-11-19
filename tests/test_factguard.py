import pytest
from fluous.factguard.check import check

SRC = "Hey, I'll be there at 7:40pm. The quote is $1,200 for 3 rooms, valid until March 5th. If it rains, we reschedule. Ask Priya."

def test_identical_passes():
    assert check(SRC, SRC)["passed"]

def test_reworded_keeps_facts():
    out = "Quick note for Priya: arriving 7:40 pm. $1200 covers 3 rooms through March 5. If it rains we'll reschedule."
    r = check(SRC, out)
    assert r["passed"], r

@pytest.mark.parametrize("bad,kind", [
    ("Arriving at 7:45pm. $1,200 for 3 rooms, valid until March 5th. If it rains, we reschedule. Ask Priya.", "changed"),   # lookalike time
    ("Arriving 7:40pm. $1,250 for 3 rooms, valid until March 5th. If it rains, we reschedule. Ask Priya.", "changed"),      # lookalike price
    ("Arriving 7:40pm. $1,200 for 4 rooms, valid until March 5th. If it rains, we reschedule. Ask Priya.", "changed"),
    ("Arriving 7:40pm. $1,200 for 3 rooms, valid until March 5th. If it rains, we reschedule. Ask Priya and Dan.", "invented"),
])
def test_corruptions_fail(bad, kind):
    r = check(SRC, bad)
    assert not r["passed"] and r[kind], r

def test_dropped_is_flagged_not_failed():
    r = check(SRC, "Arriving 7:40pm. $1,200 for 3 rooms, valid until March 5th. Ask Priya.")
    assert r["dropped"] == [] and "conditionals" in r["needs_review"]  # the lost 'if' is a review flag

def test_flipped_negation_flagged():
    r = check("We can not do Friday.", "We can do Friday.")
    assert "negations" in r["needs_review"] and not r["passed"]

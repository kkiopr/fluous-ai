from fluous.clean import clean_records, scrub, strip_quoted
from fluous.schema import Record

def test_strip_quoted():
    t = "Sounds good, see you then.\n\nOn Tue, Mar 4, 2021 at 9:00 AM Bob <b@x.com> wrote:\n> hi"
    assert strip_quoted(t) == "Sounds good, see you then."

def test_scrub():
    s = scrub("email me at a.b@c.com or call +1 (555) 123-4567, see https://x.io/y")
    assert "[EMAIL]" in s and "[PHONE]" in s and "[URL]" in s

def test_price_not_scrubbed_as_phone():
    assert "$1,200" in scrub("The total is $1,200 for 3 rooms")

def test_cutoff_merge_dedupe():
    base = dict(author_id="a", source="whatsapp", consent="self")
    recs = [
        Record(text="word " * 12, date="2021-01-01T10:00:00", **base),
        Record(text="more " * 12, date="2021-01-01T10:03:00", **base),   # merges with previous
        Record(text="late " * 30, date="2023-05-01T10:00:00", **base),   # after cutoff -> dropped
    ]
    out, stats = clean_records(recs, {"min_words": 20, "burst_gap_minutes": 10, "cutoff_date": "2022-11-30", "dedupe_jaccard": 0.8})
    assert len(out) == 1 and len(out[0].text.split()) == 24

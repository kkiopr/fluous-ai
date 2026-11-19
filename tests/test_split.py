from fluous.schema import Record
from fluous.split import leak_report, split_records

def _recs():
    out = []
    for a in ("a", "b", "c"):
        for i in range(30):
            out.append(Record(author_id=a, text=f"unique text number {i} for author {a} " + " ".join(f"w{a}{i}{j}" for j in range(15)), source="gmail", consent="self"))
    return out

def test_split_and_heldout():
    out = split_records(_recs(), eval_frac=0.2, heldout_authors=["c"], seed=1)
    assert {r.split for r in out if r.author_id == "c"} == {"heldout_author"}
    assert sum(r.split == "eval" for r in out if r.author_id == "a") == 6
    assert leak_report(out) == []

def test_deterministic():
    a = [r.id for r in split_records(_recs(), seed=5) if r.split == "eval"]
    b = [r.id for r in split_records(_recs(), seed=5) if r.split == "eval"]
    assert a == b

"""Stage 5: score the filters against the labels in the seed list.

Each seed URL is labelled keep or drop. This prints, for every page, where it ended up (kept, or
dropped and by which stage), then a small confusion table. It's the first eval of the series:
the filters are a classifier, and the seed list is its test set.
"""

from collections import Counter

from common import CLEAN, EXCLUDED, RAW, read_jsonl, read_jsonl_dir
from console import bad, dim, fail, good, heading  # common.py puts the repo root on sys.path

FETCH_FAILURES = {"robots_disallowed": "robots.txt", "http_error": "fetch: HTTP error", "fetch_error": "fetch: error"}


def main():
    if not (RAW / "manifest.jsonl").exists():
        raise fail("No data/raw/manifest.jsonl. Run fetch.py first.")
    manifest = read_jsonl(RAW / "manifest.jsonl")
    outcome = {}
    for rec in manifest:
        if rec["status"] in FETCH_FAILURES:
            outcome[rec["id"]] = FETCH_FAILURES[rec["status"]]
    for folder in sorted(p for p in EXCLUDED.iterdir() if p.is_dir()) if EXCLUDED.exists() else []:
        for doc in read_jsonl_dir(folder):
            outcome.setdefault(doc["id"], folder.name)
    for doc in read_jsonl_dir(CLEAN):
        outcome[doc["id"]] = "kept"

    rows, confusion, by_stage = [], Counter(), Counter()
    for rec in manifest:
        got = outcome.get(rec["id"], "missing")
        actual = "keep" if got == "kept" else "drop"
        correct = actual == rec["expected"]
        confusion[(rec["expected"], actual)] += 1
        if got != "kept":
            by_stage[got] += 1
        rows.append((rec["id"], rec["group"], rec["expected"], got, "" if correct else "WRONG", rec["url"]))

    print(heading(f"{'id':<7} {'group':<10} {'expected':<9} {'outcome':<20} {'':<6} url"))
    for r in rows:
        got = good(f"{r[3]:<20}") if r[3] == "kept" else f"{r[3]:<20}"
        print(f"{dim(f'{r[0]:<7}')} {r[1]:<10} {r[2]:<9} {got} {bad(f'{r[4]:<6}')} {dim(r[5])}")

    tp, fn = confusion[("keep", "keep")], confusion[("keep", "drop")]
    fp, tn = confusion[("drop", "keep")], confusion[("drop", "drop")]
    def miss(n, width):  # wrong calls: red when there are any
        return (bad if n else good)(f"{n:>{width}}")

    print(heading("\n                 kept   dropped"))
    print(f"should keep     {good(f'{tp:>5}')}   {miss(fn, 7)}")
    print(f"should drop     {miss(fp, 5)}   {good(f'{tn:>7}')}")
    print(f"\n{heading(f'Accuracy {(tp + tn) / max(1, len(rows)):.0%}')}  ·  "
          f"good pages lost {miss(fn, 0)}  ·  bad pages let through {miss(fp, 0)}")
    print("Dropped by stage: " + ", ".join(f"{k} {v}" for k, v in by_stage.most_common()))


if __name__ == "__main__":
    main()

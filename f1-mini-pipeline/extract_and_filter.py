"""Stages 3 and 4: extract the main text, filter it, and deduplicate, the way FineWeb does, at toy scale.

Pass 1 (one datatrove pipeline):
  read data/raw  ->  extract main text with trafilatura  ->  language ID (fastText, English only)
  ->  Gopher repetition filter  ->  Gopher quality filter  ->  C4 quality filter  ->  data/extracted/

Pass 2 (datatrove's four MinHash stages):
  signatures  ->  buckets  ->  clusters  ->  drop all but one document per cluster  ->  data/clean/

Every filter writes what it drops to data/excluded/<filter>/, so score_filters.py can say which
filter dropped which page. The FineWeb settings are noted where they differ from datatrove's defaults.

The language filter downloads a fastText model (about 130 MB) the first time it runs.
"""

import argparse
import json
import shutil
from pathlib import Path

import trafilatura
from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
    MinhashDedupSignature,
)
from datatrove.pipeline.filters import C4QualityFilter, GopherQualityFilter, GopherRepetitionFilter, LanguageFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter

from common import CLEAN, EXCLUDED, EXTRACTED, LOGS, MINHASH, RAW, read_jsonl
from console import fail, good  # common.py puts the repo root on sys.path


def excluded(name):
    return JsonlWriter(str(EXCLUDED / name), output_filename="${rank}.jsonl", compression=None)


def read_and_extract(data, rank: int = 0, world_size: int = 1):
    """Turn each fetched page into a Document holding just its main text.

    HTML goes through trafilatura, which drops navigation, menus and footers. Plain-text files
    (the Gutenberg .txt books) are already text and pass straight through.
    """
    dropped = []
    for rec in read_jsonl(RAW / "manifest.jsonl"):
        if rec["status"] != "ok":
            continue  # never fetched: score_filters.py reports these from the manifest
        raw = (RAW / rec["path"]).read_text(encoding="utf-8")
        if rec["path"].endswith(".txt"):
            text = raw
        else:
            text = trafilatura.extract(raw, favor_precision=True, include_comments=False, include_tables=False) or ""
        meta = {"url": rec["url"], "group": rec["group"], "expected": rec["expected"]}
        if not text.strip():
            dropped.append({"id": rec["id"], "metadata": {**meta, "filter_reason": "no main text"}})
            continue
        yield Document(text=text, id=rec["id"], metadata=meta)

    out = EXCLUDED / "extract"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "00000.jsonl", "w", encoding="utf-8") as f:
        for d in dropped:
            f.write(json.dumps(d) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-language-filter", action="store_true", help="skip language ID (no model download)")
    args = ap.parse_args()

    # Start clean each run: datatrove skips work it has already finished.
    for d in (EXTRACTED, EXCLUDED, CLEAN, MINHASH, LOGS):
        shutil.rmtree(d, ignore_errors=True)

    filters = [read_and_extract]
    if not args.skip_language_filter:
        filters.append(LanguageFilter(languages=["en"], language_threshold=0.65, exclusion_writer=excluded("language")))
    filters += [
        GopherRepetitionFilter(exclusion_writer=excluded("gopher_repetition")),
        GopherQualityFilter(exclusion_writer=excluded("gopher_quality")),
        # FineWeb keeps C4's other rules but drops the "every line ends in punctuation" one.
        C4QualityFilter(filter_no_terminal_punct=False, exclusion_writer=excluded("c4_quality")),
        JsonlWriter(str(EXTRACTED), output_filename="${rank}.jsonl", compression=None),
    ]
    LocalPipelineExecutor(pipeline=filters, tasks=1, workers=1, logging_dir=str(LOGS / "filter")).run()

    survivors = sum(1 for p in Path(EXTRACTED).glob("*.jsonl") for line in open(p) if line.strip())
    if survivors == 0:
        raise fail("No documents survived the filters, so there is nothing to deduplicate. "
                         "Run score_filters.py to see which filter dropped them.")

    # MinHash deduplication. With 14 buckets of 8 hashes, pages sharing roughly 75% or more of their
    # 5-word shingles land in the same cluster, and only one page per cluster is kept.
    cfg = MinhashConfig(n_grams=5, num_buckets=14, hashes_per_bucket=8)
    sigs, buckets, remove_ids = (str(MINHASH / n) for n in ("signatures", "buckets", "remove_ids"))
    reader = lambda: JsonlReader(str(EXTRACTED), compression=None)  # noqa: E731

    stage1 = LocalPipelineExecutor(
        pipeline=[reader(), MinhashDedupSignature(output_folder=sigs, config=cfg)],
        tasks=1, workers=1, logging_dir=str(LOGS / "minhash1"),
    )
    stage2 = LocalPipelineExecutor(
        pipeline=[MinhashDedupBuckets(input_folder=sigs, output_folder=buckets, config=cfg)],
        tasks=cfg.num_buckets, workers=1, logging_dir=str(LOGS / "minhash2"), depends=stage1,
    )
    stage3 = LocalPipelineExecutor(
        pipeline=[MinhashDedupCluster(input_folder=buckets, output_folder=remove_ids, config=cfg)],
        tasks=1, workers=1, logging_dir=str(LOGS / "minhash3"), depends=stage2,
    )
    stage4 = LocalPipelineExecutor(
        pipeline=[
            reader(),
            MinhashDedupFilter(input_folder=remove_ids, exclusion_writer=excluded("minhash_dedup")),
            JsonlWriter(str(CLEAN), output_filename="${rank}.jsonl", compression=None),
        ],
        tasks=1, workers=1, logging_dir=str(LOGS / "minhash4"), depends=stage3,
    )
    stage4.run()

    kept = sum(1 for p in Path(CLEAN).glob("*.jsonl") for line in open(p) if line.strip())
    print(f"\n{good(kept)} documents in data/clean/. Run score_filters.py to see what each filter dropped.")


if __name__ == "__main__":
    main()

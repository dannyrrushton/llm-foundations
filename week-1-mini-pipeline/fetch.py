"""Stage 2: fetch every URL in the seed list, politely, and save the raw responses.

  - one request at a time, with a delay between requests
  - a descriptive User-Agent (Wikimedia asks for one with contact details)
  - robots.txt is checked first; disallowed URLs are recorded, not fetched

Output: data/raw/<id>.<html|txt> and data/raw/manifest.jsonl, one record per seed URL.
"""

import argparse
import json
import os
import time
import urllib.robotparser
from urllib.parse import urlsplit

import requests

from common import RAW, read_seed_list

DEFAULT_UA = "llm-foundations-week1/0.1 (learning project; https://danielrushton.com)"


def robots_for(url, ua, cache):
    parts = urlsplit(url)
    root = f"{parts.scheme}://{parts.netloc}"
    if root not in cache:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = requests.get(root + "/robots.txt", headers={"User-Agent": ua}, timeout=20)
            rp.parse(r.text.splitlines() if r.ok else [])
        except requests.RequestException:
            rp.parse([])  # robots.txt unreachable: treat as allowed, as most crawlers do
        cache[root] = rp
    return cache[root]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed_list", nargs="?", default="seed_urls.tsv")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests (default 1.5)")
    ap.add_argument("--user-agent", default=os.environ.get("FETCH_USER_AGENT", DEFAULT_UA))
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    seeds = read_seed_list(args.seed_list)
    robots, records = {}, []

    for i, seed in enumerate(seeds):
        url = seed["url"]
        rec = {"id": f"doc{i:03d}", **seed, "status": None, "path": None, "content_type": None}
        if not robots_for(url, args.user_agent, robots).can_fetch(args.user_agent, url):
            rec["status"] = "robots_disallowed"
        else:
            try:
                r = requests.get(url, headers={"User-Agent": args.user_agent}, timeout=30)
                rec["http_status"] = r.status_code
                rec["final_url"] = r.url
                if r.ok:
                    ctype = r.headers.get("Content-Type", "").split(";")[0].strip()
                    ext = "txt" if ctype == "text/plain" else "html"
                    out = RAW / f"{rec['id']}.{ext}"
                    r.encoding = r.encoding or "utf-8"
                    out.write_text(r.text, encoding="utf-8")
                    rec.update(status="ok", path=str(out.relative_to(RAW)), content_type=ctype)
                else:
                    rec["status"] = "http_error"
            except requests.RequestException as e:
                rec.update(status="fetch_error", error=str(e)[:200])
            time.sleep(args.delay)
        records.append(rec)
        print(f"{rec['id']}  {rec['status']:<18} {url}")

    with open(RAW / "manifest.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    ok = sum(r["status"] == "ok" for r in records)
    print(f"\nFetched {ok} of {len(records)}. Failures are listed above; check them before the next stage.")


if __name__ == "__main__":
    main()

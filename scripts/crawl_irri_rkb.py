from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "http://knowledgebank.irri.org"
USER_AGENT = "CROPSTATE-research-crawler/1.0 (+contact: rkb domain-review pipeline)"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def crawl(registry_path: Path, output_dir: Path, delay: float, force: bool) -> dict[str, str]:
    sources = json.loads(registry_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for source in sources:
        destination = output_dir / f"{source['source_id']}.html"
        if destination.exists() and not force:
            status[source["source_id"]] = "cached"
            continue
        url = BASE_URL + source["url"]
        try:
            html = fetch(url)
        except (urllib.error.URLError, TimeoutError) as error:
            status[source["source_id"]] = f"error: {error}"
            continue
        destination.write_bytes(html)
        status[source["source_id"]] = "fetched"
        time.sleep(delay)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache IRRI Rice Knowledge Bank pages as raw HTML.")
    parser.add_argument("--registry", type=Path, default=Path("configs/knowledge_sources_irri.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between fetches.")
    parser.add_argument("--force", action="store_true", help="Re-fetch pages even if already cached.")
    args = parser.parse_args()
    status = crawl(args.registry, args.output_dir, args.delay, args.force)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

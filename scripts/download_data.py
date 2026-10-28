#!/usr/bin/env python3
"""Download the public UCI archive and preserve source and content hashes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


URL = "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip"
PAGE = "https://archive.ics.uci.edu/dataset/332/online+news+popularity"
EXPECTED_CSV_SHA256 = "b66d9088632308cc27fa35af847650d174a5a50503987c4e511de94a99d1c218"


def download(destination: Path, force: bool = False) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "online_news_popularity.zip"
    source_path = destination / "source.json"
    csv_path = destination / "OnlineNewsPopularity.csv"
    if csv_path.exists() and source_path.exists() and not force:
        record = json.loads(source_path.read_text(encoding="utf-8"))
        if record["csv_sha256"] != EXPECTED_CSV_SHA256:
            raise ValueError("Cached data is not the frozen study snapshot.")
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != record["csv_sha256"]:
            raise ValueError("Existing CSV does not match source.json; inspect or explicitly --force.")
        if archive_path.exists() and hashlib.sha256(archive_path.read_bytes()).hexdigest() != record["archive_sha256"]:
            raise ValueError("Existing ZIP does not match source.json; inspect or explicitly --force.")
        return record
    request = urllib.request.Request(URL, headers={"User-Agent": "COMP6704-Lasso-study/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_members = [m for m in archive.namelist() if m.endswith("/OnlineNewsPopularity.csv")
                       or m == "OnlineNewsPopularity.csv"]
        if len(csv_members) != 1:
            raise ValueError(f"Unexpected archive CSV members: {csv_members}")
        # Read selected entries only: do not extract untrusted archive paths.
        csv = archive.read(csv_members[0])
        descriptions = {Path(m).name: archive.read(m) for m in archive.namelist()
                        if Path(m).name in {"OnlineNewsPopularity.names", "README", "README.txt"}}
    if hashlib.sha256(csv).hexdigest() != EXPECTED_CSV_SHA256:
        raise ValueError("Official CSV differs from the frozen study snapshot; inspect the change before replacing data.")
    record = {
        "dataset": "Online News Popularity", "source_url": URL, "source_page": PAGE,
        "doi": "10.24432/C5NS3V", "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Fernandes, K., Vinagre, P., Cortez, P., and Sernadela, P. (2015). Online News Popularity. UCI Machine Learning Repository.",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": hashlib.sha256(content).hexdigest(),
        "csv_sha256": hashlib.sha256(csv).hexdigest(),
        "archive_bytes": len(content), "csv_bytes": len(csv),
        "csv_member": csv_members[0],
        "observed_data_rows": len(csv.splitlines()) - 1,
    }
    archive_path.write_bytes(content)
    csv_path.write_bytes(csv)
    for name, body in descriptions.items():
        (destination / name).write_bytes(body)
    source_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download(args.output, args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


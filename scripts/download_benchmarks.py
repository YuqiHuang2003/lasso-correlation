#!/usr/bin/env python3
"""Download auditable v2 real benchmarks without changing the v1 snapshot."""
from __future__ import annotations

import argparse
import bz2
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
UJI_URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
UJI_PAGE = "https://archive.ics.uci.edu/dataset/310/ujiindoorloc"
LEUKEMIA_PAGE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#leukemia"
LEUKEMIA_URLS = {
    "leu.bz2": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.bz2",
    "leu.t.bz2": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.t.bz2",
}
EXPECTED_CONTENT_SHA256 = {
    "ujiindoorloc.zip": "893512b82dfd7a7c345d84195b1c8019fbca0fa0d7820ce491ce5aa45ec3782f",
    "trainingData.csv": "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168",
    "validationData.csv": "5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0",
    "leu.bz2": "2f7dc1c8ea4343b17038e27ff777dad0a87d79a5fa29de54765055cc841cb75b",
    "leu.t.bz2": "582161e3d5112f6c1604d262413acca5e33cad1b3f522c681faeb369e8b4a271",
    "leukemia.libsvm": "da6fe1bea490771a47f3913fb44ed42bfda6da5dc9cce8f3531dce6bfb100fc9",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "COMP6704-Lasso-study/2.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def cached(folder: Path, force: bool) -> dict | None:
    path = folder / "source.json"
    if not path.exists() or force:
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in record["files"].items():
        location = folder / name
        if expected["sha256"] != EXPECTED_CONTENT_SHA256[name]:
            raise ValueError(f"Cached {name} is not the recorded v2 source snapshot.")
        if not location.is_file() or digest(location.read_bytes()) != expected["sha256"]:
            raise ValueError(f"Cached benchmark differs from source record: {location}; inspect before --force.")
    return record


def save(folder: Path, files: dict[str, bytes], metadata: dict) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        if digest(body) != EXPECTED_CONTENT_SHA256[name]:
            raise ValueError(f"Official {name} changed from the frozen v2 snapshot; inspect before replacing.")
    record = dict(metadata, retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                  files={name: {"sha256": digest(body), "bytes": len(body)} for name, body in files.items()})
    for name, body in files.items():
        (folder / name).write_bytes(body)
    (folder / "source.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def download_uji(folder: Path, force: bool = False) -> dict:
    old = cached(folder, force)
    if old is not None:
        return old
    archive_bytes = fetch(UJI_URL)
    files = {"ujiindoorloc.zip": archive_bytes}
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for expected_name in ("trainingData.csv", "validationData.csv"):
            names = [name for name in archive.namelist() if Path(name).name == expected_name]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one {expected_name} in official UCI archive.")
            files[expected_name] = archive.read(names[0])
    return save(folder, files, {
        "dataset": "UJIIndoorLoc", "source_url": UJI_URL, "source_page": UJI_PAGE,
        "doi": "10.24432/C5MS59", "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Torres-Sospedra, J., Montoliu, R., Martinez-Us, A., Arnau, T., and Avariento, J. (2014). UJIIndoorLoc. UCI Machine Learning Repository.",
        "original_training_rows": len(files["trainingData.csv"].splitlines()) - 1,
        "original_validation_rows": len(files["validationData.csv"].splitlines()) - 1,
        "study_pool": "Official trainingData.csv only; separate validationData.csv preserved but not used.",
    })


def download_leukemia(folder: Path, force: bool = False) -> dict:
    old = cached(folder, force)
    if old is not None:
        return old
    files = {name: fetch(url) for name, url in LEUKEMIA_URLS.items()}
    text_parts = [bz2.decompress(files[name]).rstrip(b"\r\n") for name in LEUKEMIA_URLS]
    files["leukemia.libsvm"] = b"\n".join(text_parts) + b"\n"
    counts = [len(part.splitlines()) for part in text_parts]
    if counts != [38, 34]:
        raise ValueError(f"Unexpected LIBSVM leukemia split sizes: {counts}")
    return save(folder, files, {
        "dataset": "Leukemia (LIBSVM normalized snapshot)", "source_page": LEUKEMIA_PAGE,
        "source_urls": LEUKEMIA_URLS,
        "license": "Not specified on the checked LIBSVM dataset page",
        "redistribution": "Do not include raw/derived data in distributable submission archives; fetch locally using this script.",
        "attribution": "Golub et al. (1999), distributed in preprocessed LIBSVM format by the LIBSVM dataset collection.",
        "source_preprocessing": "LIBSVM reports instance-wise zero-mean/unit-variance normalization, then feature-wise zero-mean/unit-variance normalization on merged training/testing data.",
        "original_split_rows": counts, "combined_rows": sum(counts), "n_features": 7129,
        "merge_order": list(LEUKEMIA_URLS),
        "evaluation_scope": "All 72 observations form one fixed least-squares Lasso optimization benchmark; binary labels are numeric targets. No generalization or true-support claim.",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "uji", "leukemia"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmarks")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = {}
    if args.dataset in ("all", "uji"):
        report["uji"] = download_uji(args.output / "ujiindoorloc", args.force)
    if args.dataset in ("all", "leukemia"):
        report["leukemia"] = download_leukemia(args.output / "leukemia", args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

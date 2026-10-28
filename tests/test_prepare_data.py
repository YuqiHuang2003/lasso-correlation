"""Publication data bootstrap checks: local fixtures only, never real downloads."""
from pathlib import Path
import hashlib
import importlib.util
import json

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
spec = importlib.util.spec_from_file_location("publication_prepare_data", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def frozen_repo(tmp_path):
    payloads, historical = {}, {}
    for dataset, (manifest, folder, names) in module.DATASETS.items():
        contents = {name: f"fixture {dataset}/{name}\n".encode() for name in sorted(names)}
        payloads[dataset] = contents
        hashes = {name: {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
                  for name, body in contents.items()}
        record = {"dataset": dataset, "retrieved_at_utc": "2001-01-01T00:00:00+00:00"}
        if dataset == "news":
            record.update(archive_sha256=hashes["online_news_popularity.zip"]["sha256"],
                          archive_bytes=hashes["online_news_popularity.zip"]["bytes"],
                          csv_sha256=hashes["OnlineNewsPopularity.csv"]["sha256"],
                          csv_bytes=hashes["OnlineNewsPopularity.csv"]["bytes"])
        else:
            record["files"] = hashes
        # Non-default whitespace and no trailing newline must survive exactly.
        body = json.dumps(record, ensure_ascii=False, indent=3).encode()
        path = tmp_path / manifest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        historical[dataset] = body
    return tmp_path, payloads, historical


def fake_downloader(payloads, calls, *, bad_dataset=None):
    def download(root, dataset, folder):
        calls.append(dataset)
        folder.mkdir(parents=True)
        for name, body in payloads[dataset].items():
            (folder / name).write_bytes(body + (b"changed" if dataset == bad_dataset else b""))
        (folder / "source.json").write_text('{"retrieved_at_utc":"2099-12-31"}')
        return {"retrieved_at_utc": "2099-12-31"}
    return download


@pytest.mark.parametrize("existing_sources", [False, True])
def test_fresh_bootstrap_restores_exact_historical_metadata_and_logs_new_acquisition(
        frozen_repo, monkeypatch, existing_sources):
    root, payloads, historical = frozen_repo
    if existing_sources:
        # Exported frozen metadata may exist while every raw input is absent.
        for dataset, (_, folder, _) in module.DATASETS.items():
            location = root / folder
            location.mkdir(parents=True)
            (location / "source.json").write_bytes(historical[dataset])
    calls = []
    monkeypatch.setattr(module, "_download_to_staging", fake_downloader(payloads, calls))
    report = module.prepare_data(root)
    assert calls == ["news", "uji", "leukemia"] and report["network_used"] is True
    for dataset, (_, folder, _) in module.DATASETS.items():
        assert (root / folder / "source.json").read_bytes() == historical[dataset]
        for name, body in payloads[dataset].items():
            assert (root / folder / name).read_bytes() == body
    log = json.loads((root / "data/acquisition_log.json").read_text())
    assert "not this new acquisition" in log["canonical_source_notice"]
    assert log["events"][0]["network_used"] is True
    assert log["events"][0]["publication_state"] == "completed"
    assert not log["events"][0]["network_acquisition_completed_at_utc"].startswith("2001-")


def test_matching_cache_uses_no_network_or_metadata_rewrites(frozen_repo, monkeypatch):
    root, payloads, _ = frozen_repo
    monkeypatch.setattr(module, "_download_to_staging", fake_downloader(payloads, []))
    module.prepare_data(root)
    before = {str(p.relative_to(root)): p.read_bytes() for p in (root / "data").rglob("*") if p.is_file()}
    def forbidden(*args):
        pytest.fail("Matching input cache must not invoke a download helper")
    monkeypatch.setattr(module, "_download_to_staging", forbidden)
    report = module.prepare_data(root)
    after = {str(p.relative_to(root)): p.read_bytes() for p in (root / "data").rglob("*") if p.is_file()}
    assert report["network_used"] is False and before == after


def test_raw_cache_without_metadata_restores_source_offline(frozen_repo, monkeypatch):
    root, payloads, historical = frozen_repo
    folder = root / module.DATASETS["news"][1]
    folder.mkdir(parents=True)
    for name, body in payloads["news"].items():
        (folder / name).write_bytes(body)
    monkeypatch.setattr(module, "_download_to_staging", lambda *args: pytest.fail("No download needed"))
    report = module.prepare_data(root, ("news",))
    assert report["network_used"] is False
    assert report["datasets"]["news"]["canonical_source_restored"] is True
    assert (folder / "source.json").read_bytes() == historical["news"]


def test_existing_bad_raw_is_rejected_before_any_download_and_not_overwritten(frozen_repo, monkeypatch):
    root, payloads, _ = frozen_repo
    folder = root / module.DATASETS["leukemia"][1]
    folder.mkdir(parents=True)
    wrong = folder / "leu.bz2"
    wrong_body = b"x" * len(payloads["leukemia"]["leu.bz2"])
    wrong.write_bytes(wrong_body)  # Same size: rejection must check content hash.
    monkeypatch.setattr(module, "_download_to_staging", lambda *args: pytest.fail("Preflight must fail first"))
    with pytest.raises(ValueError, match="frozen hash"):
        module.prepare_data(root)
    assert wrong.read_bytes() == wrong_body
    assert not (root / "data/raw").exists()


def test_staged_payload_must_match_every_manifest_file_before_copy(frozen_repo, monkeypatch):
    root, payloads, _ = frozen_repo
    monkeypatch.setattr(module, "_download_to_staging", fake_downloader(payloads, [], bad_dataset="leukemia"))
    with pytest.raises(ValueError, match="frozen hash"):
        module.prepare_data(root)
    assert not (root / "data").exists()


@pytest.mark.parametrize("name", ["../outside.csv", "/outside.csv", "nested/file.csv", "C:\\outside.csv"])
def test_manifest_paths_must_be_basenames(frozen_repo, monkeypatch, name):
    root, _, _ = frozen_repo
    path = root / module.DATASETS["uji"][0]
    record = json.loads(path.read_bytes())
    record["files"][name] = record["files"].pop("trainingData.csv")
    path.write_text(json.dumps(record))
    monkeypatch.setattr(module, "_download_to_staging", lambda *args: pytest.fail("Unsafe manifest must fail first"))
    with pytest.raises(ValueError, match="basenames"):
        module.prepare_data(root, ("uji",))


@pytest.mark.parametrize("relative", ["data/raw", "data/raw/OnlineNewsPopularity.csv",
                                     "data/raw/source.json", "data/acquisition_log.json"])
def test_destination_symlinks_cannot_escape_repository(frozen_repo, monkeypatch, tmp_path, relative):
    root, _, _ = frozen_repo
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"untouched")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(outside if relative == "data/raw" else sentinel,
                    target_is_directory=relative == "data/raw")
    monkeypatch.setattr(module, "_download_to_staging", lambda *args: pytest.fail("Symlink must fail first"))
    with pytest.raises(ValueError, match="Symlink"):
        module.prepare_data(root, ("news",))
    assert sentinel.read_bytes() == b"untouched"


def test_existing_noncanonical_source_is_not_silently_overwritten(frozen_repo, monkeypatch):
    root, _, _ = frozen_repo
    folder = root / module.DATASETS["news"][1]
    folder.mkdir(parents=True)
    path = folder / "source.json"
    path.write_bytes(b'{"retrieved_at_utc":"different"}')
    monkeypatch.setattr(module, "_download_to_staging", lambda *args: pytest.fail("Metadata must fail first"))
    with pytest.raises(ValueError, match="historical metadata"):
        module.prepare_data(root, ("news",))
    assert path.read_bytes() == b'{"retrieved_at_utc":"different"}'


def test_interrupted_copy_leaves_no_partial_destination_and_retry_recovers(frozen_repo, monkeypatch):
    root, payloads, historical = frozen_repo
    calls = []
    monkeypatch.setattr(module, "_download_to_staging", fake_downloader(payloads, calls))
    copy = module.shutil.copyfileobj
    def interrupted(source, output, length):
        output.write(source.read(4))
        raise OSError("simulated full disk")
    monkeypatch.setattr(module.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="full disk"):
        module.prepare_data(root, ("news",))
    folder = root / module.DATASETS["news"][1]
    assert not list(folder.glob("*"))  # Neither a partial raw file nor a temp remains.
    log = json.loads((root / "data/acquisition_log.json").read_text())
    assert log["events"][0]["publication_state"] == "pending"
    monkeypatch.setattr(module.shutil, "copyfileobj", copy)
    assert module.prepare_data(root, ("news",))["pending_acquisitions"] == 0
    assert (folder / "source.json").read_bytes() == historical["news"]


def test_failed_completion_log_write_recovers_offline_without_losing_acquisition_time(frozen_repo, monkeypatch):
    root, payloads, _ = frozen_repo
    monkeypatch.setattr(module, "_download_to_staging", fake_downloader(payloads, []))
    writer = module._write_log
    calls = []
    def fail_completion(root, log):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("simulated completion log failure")
        writer(root, log)
    monkeypatch.setattr(module, "_write_log", fail_completion)
    with pytest.raises(OSError, match="completion log failure"):
        module.prepare_data(root, ("news",))
    log_path = root / "data/acquisition_log.json"
    pending = json.loads(log_path.read_text())
    timestamp = pending["events"][0]["network_acquisition_completed_at_utc"]
    assert pending["events"][0]["publication_state"] == "pending"
    monkeypatch.setattr(module, "_write_log", writer)
    monkeypatch.setattr(module, "_download_to_staging", lambda *args: pytest.fail("Retry must stay offline"))
    report = module.prepare_data(root, ("news",))
    recovered = json.loads(log_path.read_text())
    assert report["network_used"] is False and report["pending_acquisitions"] == 0
    assert len(recovered["events"]) == 1
    assert recovered["events"][0]["network_acquisition_completed_at_utc"] == timestamp
    assert recovered["events"][0]["publication_state"] == "completed"


def test_pending_log_must_be_durable_before_publishing_raw_files(frozen_repo, monkeypatch):
    root, payloads, _ = frozen_repo
    monkeypatch.setattr(module, "_download_to_staging", fake_downloader(payloads, []))
    def fail(*args):
        raise OSError("simulated pending log failure")
    monkeypatch.setattr(module, "_write_log", fail)
    with pytest.raises(OSError, match="pending log failure"):
        module.prepare_data(root, ("news",))
    assert not (root / module.DATASETS["news"][1]).exists()


@pytest.mark.parametrize("dataset,filename,helper_name", [
    ("news", "download_data.py", "download"),
    ("uji", "download_benchmarks.py", "download_uji"),
    ("leukemia", "download_benchmarks.py", "download_leukemia"),
])
def test_original_helper_adapter_calls_only_staging(frozen_repo, dataset, filename, helper_name):
    root, _, _ = frozen_repo
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / filename).write_text(
        f"def {helper_name}(folder, force=False):\n"
        "    assert force is False\n"
        "    return {'staging': str(folder)}\n"
    )
    staging = root / "disposable_staging"
    assert module._download_to_staging(root, dataset, staging) == {"staging": str(staging)}

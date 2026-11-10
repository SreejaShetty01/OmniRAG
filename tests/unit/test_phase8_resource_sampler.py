import json

import pytest

from scripts.phase8.resource_sampler import ResourceSampler


def test_sampler_records_container_process_pagefile_and_ollama(monkeypatch):
    monkeypatch.setattr("scripts.phase8.resource_sampler._psutil_snapshot", lambda pids: {
        "available": True, "processes": [{"pid": 10, "cpu_percent": 11.0, "rss_bytes": 2048}],
        "pagefile": {"used_bytes": 4096, "total_bytes": 8192, "source": "test"}})
    monkeypatch.setattr("scripts.phase8.resource_sampler._docker_stats", lambda names: [
        {"name": name, "cpu_percent": "20.0%", "memory_usage": "1MiB / 2MiB", "memory_percent": "50%"}
        for name in names])
    monkeypatch.setattr("scripts.phase8.resource_sampler._ollama_loaded_models", lambda url: {
        "available": True, "models": [{"name": "local-model"}]})
    record = ResourceSampler(containers=["api-eval"], pids=[10]).sample()
    assert record["host_processes"][0]["rss_bytes"] == 2048
    assert record["pagefile"]["used_bytes"] == 4096
    assert record["docker_containers"][0]["name"] == "api-eval"
    assert record["ollama_loaded_models"]["models"][0]["name"] == "local-model"


def test_sampler_writes_ndjson_and_links_samples_to_new_run_manifest(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"run_id": "new-run", "status": "running"}), encoding="utf-8")
    samples_path = tmp_path / "resources.ndjson"
    monkeypatch.setattr(ResourceSampler, "sample", lambda self: {"timestamp_utc": "sample", "cpu": 1})
    sampler = ResourceSampler(containers=["api-eval"], ollama_url="http://localhost:11434")
    count = sampler.run(duration_seconds=0.02, output_path=samples_path, manifest_path=manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    assert count == len(samples) == 1
    assert manifest["resource_sampling"]["artifact"] == "resources.ndjson"
    assert manifest["resource_sampling"]["interval_seconds"] == 1.0
    assert manifest["resource_sampling"]["sample_count"] == 1
    assert manifest["resource_sampling"]["status"] == "completed"


def test_sampler_refuses_historical_or_unlinked_outputs(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"run_id": "historical", "resource_sampling": {}}),
                             encoding="utf-8")
    sampler = ResourceSampler()
    with pytest.raises(ValueError, match="unmodified run manifest"):
        sampler.run(duration_seconds=0.01, output_path=tmp_path / "resources.ndjson",
                    manifest_path=manifest_path)

    manifest_path.write_text(json.dumps({"run_id": "new-run"}), encoding="utf-8")
    with pytest.raises(ValueError, match="stored beside or below"):
        sampler.run(duration_seconds=0.01, output_path=tmp_path.parent / "outside.ndjson",
                    manifest_path=manifest_path)


def test_sampler_requires_exactly_one_second_interval():
    with pytest.raises(ValueError, match="fixed at one second"):
        ResourceSampler(interval_seconds=2.0)

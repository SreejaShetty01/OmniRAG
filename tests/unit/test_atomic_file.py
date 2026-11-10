from pathlib import Path

import pytest

import utils.atomic_file as atomic_file


def test_atomic_replace_retries_transient_windows_permission_lock(tmp_path, monkeypatch):
    source = tmp_path / "temp.json"
    destination = tmp_path / "result.json"
    source.write_text("new", encoding="utf-8")
    actual_replace = atomic_file.os.replace
    attempts = 0

    def transient_replace(src, dst):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporary Windows file lock")
        actual_replace(src, dst)

    monkeypatch.setattr(atomic_file, "IS_WINDOWS", True)
    monkeypatch.setattr(atomic_file.os, "replace", transient_replace)
    monkeypatch.setattr(atomic_file.time, "sleep", lambda _: None)

    atomic_file.atomic_replace(source, destination)

    assert attempts == 3
    assert destination.read_text(encoding="utf-8") == "new"
    assert not source.exists()


def test_atomic_replace_remains_bounded_for_persistent_permission_error(monkeypatch, tmp_path):
    source = tmp_path / "temp.json"
    destination = tmp_path / "result.json"
    source.write_text("new", encoding="utf-8")
    attempts = 0

    def always_locked(src: Path, dst: Path):
        nonlocal attempts
        attempts += 1
        raise PermissionError("persistent lock")

    monkeypatch.setattr(atomic_file, "IS_WINDOWS", True)
    monkeypatch.setattr(atomic_file.os, "replace", always_locked)
    monkeypatch.setattr(atomic_file.time, "sleep", lambda _: None)

    with pytest.raises(PermissionError, match="persistent lock"):
        atomic_file.atomic_replace(source, destination)

    assert attempts == atomic_file.MAX_REPLACE_ATTEMPTS

"""Read-only one-second resource sampler for an already-running evaluation run.

Usage example (no service is started by this script):
python -m scripts.phase8.resource_sampler --manifest data/phase8/benchmarks/<run>/manifest.json \
  --output data/phase8/benchmarks/<run>/resources.ndjson --duration-seconds 3600 \
  --container <verified-evaluation-api-name> --container <verified-evaluation-qdrant-name>

Resolve container names and the Ollama endpoint from the isolated deployment before use;
the example intentionally does not assume names or that Ollama is evaluation-exclusive.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.request import Request, urlopen


def _psutil_snapshot(pids: list[int]) -> dict[str, Any]:
    try:
        import psutil  # Optional: sampler does not add it as an application dependency.
    except ImportError:
        return {"available": False, "processes": [], "pagefile": None,
                "reason": "psutil_unavailable"}
    processes = []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            processes.append({"pid": pid, "name": proc.name(), "cpu_percent": proc.cpu_percent(None),
                              "rss_bytes": proc.memory_info().rss})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            processes.append({"pid": pid, "unavailable": True})
    swap = psutil.swap_memory()
    return {"available": True, "processes": processes,
            "pagefile": {"used_bytes": swap.used, "total_bytes": swap.total,
                         "percent": swap.percent, "source": "psutil.swap_memory"}}


def _windows_pagefile_snapshot() -> dict[str, Any] | None:
    if sys.platform != "win32":
        return None
    command = ("$x=Get-CimInstance Win32_PageFileUsage; "
               "[pscustomobject]@{current_usage_mb=(($x | Measure-Object -Property CurrentUsage -Sum).Sum); "
               "allocated_mb=(($x | Measure-Object -Property AllocatedBaseSize -Sum).Sum)} | "
               "ConvertTo-Json -Compress")
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                capture_output=True, text=True, timeout=2, check=True)
        values = json.loads(result.stdout)
        if not isinstance(values, dict):
            return None
        return {"used_bytes": int(values.get("current_usage_mb") or 0) * 1024 * 1024,
                "total_bytes": int(values.get("allocated_mb") or 0) * 1024 * 1024,
                "source": "Win32_PageFileUsage"}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return None


def _docker_stats(names: list[str]) -> list[dict[str, Any]]:
    if not names:
        return []
    command = ["docker", "stats", "--no-stream", "--format",
               "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}", *names]
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) == 4:
            rows.append({"name": fields[0], "cpu_percent": fields[1],
                         "memory_usage": fields[2], "memory_percent": fields[3]})
    return rows


def _ollama_loaded_models(url: str, timeout_seconds: float = 2.0) -> dict[str, Any]:
    endpoint = url.rstrip("/") + "/api/ps"
    try:
        with urlopen(Request(endpoint, method="GET"), timeout=timeout_seconds) as response:
            payload = json.loads(response.read())
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return {"available": False, "error_type": "InvalidResponse"}
        return {"available": True, "models": [
            {key: row.get(key) for key in ("name", "model", "size", "size_vram", "expires_at")}
            for row in models if isinstance(row, dict)]}
    except Exception as exc:
        return {"available": False, "error_type": type(exc).__name__}


class ResourceSampler:
    def __init__(self, *, containers: list[str] | None = None, pids: list[int] | None = None,
                 ollama_url: str = "http://localhost:11434", interval_seconds: float = 1.0):
        if interval_seconds != 1.0:
            raise ValueError("Phase 8 resource sampling interval is fixed at one second")
        self.containers, self.pids = containers or [], pids or []
        self.ollama_url = ollama_url

    def sample(self) -> dict[str, Any]:
        stamp = datetime.now(timezone.utc).isoformat()
        process = _psutil_snapshot(self.pids)
        pagefile = process.get("pagefile") or _windows_pagefile_snapshot()
        docker = []
        docker_error = None
        try:
            docker = _docker_stats(self.containers)
        except Exception as exc:
            docker_error = type(exc).__name__
        return {"timestamp_utc": stamp, "host_processes": process.get("processes", []),
                "pagefile": pagefile, "docker_containers": docker,
                "docker_error_type": docker_error, "ollama_loaded_models": _ollama_loaded_models(self.ollama_url)}

    def run(self, *, duration_seconds: float, output_path: Path,
            manifest_path: Path) -> int:
        if duration_seconds <= 0:
            raise ValueError("Sampling duration must be positive")
        manifest_path = manifest_path.resolve()
        output_path = output_path.resolve()
        try:
            relative_output = output_path.relative_to(manifest_path.parent)
        except ValueError:
            raise ValueError("Resource samples must be stored beside or below the run manifest") from None
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite resource file: {output_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("run_id") or "resource_sampling" in manifest:
            raise ValueError("Expected an unmodified run manifest with a run_id and no resource sampling")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)
        count = 0
        interrupted = False
        deadline = time.monotonic() + duration_seconds
        next_tick = time.monotonic()
        try:
            with output_path.open("x", encoding="utf-8", newline="\n") as stream:
                while next_tick < deadline:
                    wait = next_tick - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    tick = time.monotonic()
                    if tick >= deadline:
                        break
                    record = self.sample()
                    stream.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
                    stream.flush()
                    count += 1
                    next_tick = tick + 1.0
        except KeyboardInterrupt:
            interrupted = True
        manifest["resource_sampling"] = {
            "artifact": relative_output.as_posix(), "interval_seconds": 1.0,
            "sample_count": count, "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "interrupted" if interrupted else "completed",
            "containers": list(self.containers), "pids": list(self.pids),
            "ollama_endpoint": self.ollama_url,
        }
        temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, manifest_path)
        if interrupted:
            raise KeyboardInterrupt
        return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-seconds", required=True, type=float)
    parser.add_argument("--container", action="append", default=[])
    parser.add_argument("--pid", action="append", type=int, default=[])
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()
    count = ResourceSampler(containers=args.container, pids=args.pid,
                            ollama_url=args.ollama_url).run(
                                duration_seconds=args.duration_seconds,
                                output_path=args.output, manifest_path=args.manifest)
    print(f"resource_samples={count} interval_seconds=1.0 manifest={args.manifest}")


if __name__ == "__main__":
    main()

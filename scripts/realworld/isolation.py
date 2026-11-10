"""Read-only Docker evidence for the dedicated M1 real-world Qdrant path."""
from __future__ import annotations

import hashlib
import json
import subprocess
from urllib.parse import urlsplit


EXPECTED_PROJECT = "omnirag"
EXPECTED_API_SERVICE = "api-eval-rw"
EXPECTED_QDRANT_SERVICE = "qdrant-rw"
EXPECTED_API_CONTAINER = "omnirag-api-eval-rw"
EXPECTED_QDRANT_CONTAINER = "omnirag-qdrant-rw"
EXPECTED_QDRANT_VOLUME = "omnirag_omnirag_eval_rw_qdrant_data"
EXPECTED_SQLITE_VOLUME = "omnirag_omnirag_eval_rw_data"
EXPECTED_INTERNAL_QDRANT_URL = "http://qdrant-rw:6333"
PROTECTED_COLLECTIONS = {"omnirag", "omnirag_eval_phase8_20261104"}
PROTECTED_CONTAINER_NAMES = ("omnirag-api-1", "omnirag-api-eval", "omnirag-qdrant-1")


def _name(container: dict) -> str:
    return str(container.get("Name", "")).lstrip("/")


def _labels(container: dict) -> dict:
    return (container.get("Config") or {}).get("Labels") or {}


def _environment(container: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in (container.get("Config") or {}).get("Env") or []:
        key, sep, value = str(entry).partition("=")
        if sep:
            if key in result:
                raise RuntimeError(f"Container has duplicate effective environment key: {key}")
            result[key] = value
    return result


def _volume_mounts(container: dict) -> list[dict]:
    return [mount for mount in container.get("Mounts", []) if mount.get("Type") == "volume"]


def _identity(container: dict) -> dict:
    state = container.get("State") or {}
    return {
        "name": _name(container), "id": container.get("Id"),
        "running": bool(state.get("Running")),
        "health": ((state.get("Health") or {}).get("Status")),
        "project": _labels(container).get("com.docker.compose.project"),
        "service": _labels(container).get("com.docker.compose.service"),
        "volume_mounts": sorted((mount.get("Name"), mount.get("Destination"))
                                 for mount in _volume_mounts(container)),
    }


def validate_dedicated_qdrant(*, api_container: dict, qdrant_container: dict,
                              other_qdrant_containers: list[dict],
                              protected_containers: list[dict],
                              existing_qdrant_volume_names: list[str],
                              qdrant_volume: dict, api_settings: dict,
                              expected_collection: str, qdrant_host_url: str,
                              user_id: str, tenant_id: str) -> dict:
    """Validate that API and target collection are backed only by the new Qdrant volume."""
    errors: list[str] = []
    if not expected_collection.startswith("omnirag_eval_") or expected_collection in PROTECTED_COLLECTIONS:
        errors.append("target collection is not an allowed evaluation collection")

    api_name, qdrant_name = _name(api_container), _name(qdrant_container)
    api_labels, qdrant_labels = _labels(api_container), _labels(qdrant_container)
    if (api_name != EXPECTED_API_CONTAINER or api_labels.get("com.docker.compose.project") != EXPECTED_PROJECT or
            api_labels.get("com.docker.compose.service") != EXPECTED_API_SERVICE):
        errors.append("API container identity is not the prepared isolated Compose service")
    if (qdrant_name != EXPECTED_QDRANT_CONTAINER or qdrant_labels.get("com.docker.compose.project") != EXPECTED_PROJECT or
            qdrant_labels.get("com.docker.compose.service") != EXPECTED_QDRANT_SERVICE):
        errors.append("Qdrant container identity is not the dedicated Compose service")
    qdrant_image = str((qdrant_container.get("Config") or {}).get("Image", ""))
    if not qdrant_image.startswith("qdrant/qdrant:"):
        errors.append("dedicated service is not running the expected Qdrant image")

    env = _environment(api_container)
    if env.get("OMNIRAG_QDRANT_URL") != EXPECTED_INTERNAL_QDRANT_URL:
        errors.append("API effective Qdrant URL is not http://qdrant-rw:6333")
    if env.get("OMNIRAG_QDRANT_COLLECTION") != expected_collection:
        errors.append("API effective collection does not match the corpus target")
    if "QDRANT_URL" in env or "OMNIRAG_QDRANT_URL" not in env:
        errors.append("API has an ambiguous or missing Qdrant URL environment setting")
    if api_settings.get("physical_collection") != expected_collection:
        errors.append("signed API settings do not identify the target collection")
    if (api_settings.get("user_id"), api_settings.get("tenant_id")) != (user_id, tenant_id):
        errors.append("signed runtime identity was not verified")
    allowed = api_settings.get("accepted_identities") or []
    if not any(item.get("user_id") == user_id and item.get("tenant_id") == tenant_id
               for item in allowed if isinstance(item, dict)):
        errors.append("principal/tenant allowlist was not verified")
    limits = api_settings.get("limits") or {}
    if (limits.get("max_cost_usd") != 0 or limits.get("max_total_tokens") != 4_200_000 or
            limits.get("max_calls") != 3_000):
        errors.append("approved budget caps were not verified")

    for container, expected_name in ((api_container, EXPECTED_API_CONTAINER),
                                     (qdrant_container, EXPECTED_QDRANT_CONTAINER)):
        state = container.get("State") or {}
        if not state.get("Running") or (state.get("Health") or {}).get("Status") != "healthy":
            errors.append(f"{expected_name} is not running and healthy")
        networks = (container.get("NetworkSettings") or {}).get("Networks") or {}
        if "omnirag_default" not in networks:
            errors.append(f"{expected_name} is not attached to omnirag_default")
    qdrant_network = ((qdrant_container.get("NetworkSettings") or {}).get("Networks") or {}).get("omnirag_default") or {}
    if "qdrant-rw" not in (qdrant_network.get("Aliases") or []):
        errors.append("dedicated Qdrant has no qdrant-rw network alias")

    api_mounts = _volume_mounts(api_container)
    if len(api_mounts) != 1 or api_mounts[0].get("Name") != EXPECTED_SQLITE_VOLUME or api_mounts[0].get("Destination") != "/app/data":
        errors.append("API SQLite/state mount is not the dedicated M1 volume")
    qdrant_mounts = _volume_mounts(qdrant_container)
    if len(qdrant_mounts) != 1 or qdrant_mounts[0].get("Name") != EXPECTED_QDRANT_VOLUME or qdrant_mounts[0].get("Destination") != "/qdrant/storage":
        errors.append("Qdrant storage mount is not the dedicated M1 volume")
    if any(mount.get("Type") != "volume" for mount in qdrant_container.get("Mounts", [])):
        errors.append("dedicated Qdrant has a non-volume storage mount")
    volume_labels = qdrant_volume.get("Labels") or {}
    if (qdrant_volume.get("Name") != EXPECTED_QDRANT_VOLUME or
            volume_labels.get("com.docker.compose.project") != EXPECTED_PROJECT or
            volume_labels.get("com.docker.compose.volume") != "omnirag_eval_rw_qdrant_data"):
        errors.append("Qdrant volume is not the dedicated Compose volume")
    if EXPECTED_QDRANT_VOLUME in existing_qdrant_volume_names:
        errors.append("dedicated Qdrant storage volume is reused by another Qdrant service")
    if EXPECTED_QDRANT_VOLUME in {m.get("Name") for c in other_qdrant_containers for m in _volume_mounts(c)}:
        errors.append("dedicated Qdrant storage volume is mounted by another Qdrant container")
    if {m.get("Name") for m in api_mounts} & {m.get("Name") for m in qdrant_mounts}:
        errors.append("API state and Qdrant storage volumes are not separate")
    if {m.get("Name") for m in api_mounts} & set(existing_qdrant_volume_names):
        errors.append("API state volume reuses an existing Qdrant storage volume")

    protected_by_name = {_name(item): item for item in protected_containers}
    if set(PROTECTED_CONTAINER_NAMES) - set(protected_by_name):
        errors.append("one or more protected legacy containers are missing from the Docker inventory")
    protected_identities = sorted((_identity(item) for item in protected_containers),
                                  key=lambda row: row["name"])

    url = urlsplit(qdrant_host_url)
    if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"} or not url.port:
        errors.append("host Qdrant URL must use a loopback HTTP endpoint")
    else:
        published = ((qdrant_container.get("NetworkSettings") or {}).get("Ports") or {}).get("6333/tcp") or []
        if not any(binding.get("HostIp") in {"127.0.0.1", "::1"} and
                   str(binding.get("HostPort")) == str(url.port) for binding in published):
            errors.append("host Qdrant URL does not map to the dedicated container")

    existing_ids = sorted((_identity(c) for c in other_qdrant_containers), key=lambda row: row["name"])
    if errors:
        raise RuntimeError("Dedicated Qdrant isolation proof failed: " + "; ".join(errors))
    proof = {
        "mode": "dedicated_qdrant_storage_isolation", "verified": True,
        "collection": expected_collection, "api_qdrant_url": EXPECTED_INTERNAL_QDRANT_URL,
        "host_qdrant_url": qdrant_host_url,
        "api": _identity(api_container), "qdrant": _identity(qdrant_container),
        "qdrant_volume": qdrant_volume.get("Name"),
        "other_qdrant_containers": existing_ids,
        "protected_legacy_containers": protected_identities,
        "existing_qdrant_volume_names": sorted(existing_qdrant_volume_names),
        "live_protected_collection_fingerprints": "not_measured_dedicated_storage_boundary",
    }
    canonical = json.dumps(proof, sort_keys=True, separators=(",", ":")).encode()
    proof["proof_sha256"] = hashlib.sha256(canonical).hexdigest()
    return proof


def inspect_dedicated_runtime(api_container_name: str = EXPECTED_API_CONTAINER,
                              qdrant_container_name: str = EXPECTED_QDRANT_CONTAINER) -> dict:
    """Read Docker metadata without emitting container environment or secrets."""
    def run(args: list[str]) -> str:
        result = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError("Docker isolation inspection failed")
        return result.stdout

    ids = [line.strip() for line in run(["ps", "-aq"]).splitlines() if line.strip()]
    inspected = json.loads(run(["inspect", api_container_name, qdrant_container_name, *ids])) if ids else []
    by_name = {_name(item): item for item in inspected}
    try:
        api = by_name[api_container_name]
        qdrant = by_name[qdrant_container_name]
    except KeyError as exc:
        raise RuntimeError("Expected isolated API or Qdrant container is missing") from exc
    others = []
    for item in inspected:
        if _name(item) in {api_container_name, qdrant_container_name}:
            continue
        labels, image = _labels(item), str((item.get("Config") or {}).get("Image", ""))
        if labels.get("com.docker.compose.service") == "qdrant" or image.startswith("qdrant/qdrant"):
            others.append(item)
    qdrant_volumes = sorted({str(mount.get("Name")) for item in others
                             for mount in _volume_mounts(item) if mount.get("Name")})
    listed = [name for name in run(["volume", "ls", "-q", "--filter", "name=qdrant"]).splitlines() if name.strip()]
    qdrant_volumes = sorted((set(qdrant_volumes) | set(listed)) - {EXPECTED_QDRANT_VOLUME})
    protected = []
    for name in PROTECTED_CONTAINER_NAMES:
        if name not in by_name:
            raise RuntimeError("A protected legacy container is missing from the Docker inventory")
        protected.append(by_name[name])
    if len(_volume_mounts(qdrant)) != 1:
        raise RuntimeError("Dedicated Qdrant must have exactly one volume mount")
    volume_name = _volume_mounts(qdrant)[0].get("Name")
    volume_rows = json.loads(run(["volume", "inspect", str(volume_name)]))
    if len(volume_rows) != 1:
        raise RuntimeError("Dedicated Qdrant volume inspection was ambiguous")
    return {"api_container": api, "qdrant_container": qdrant,
            "other_qdrant_containers": others, "protected_containers": protected,
            "existing_qdrant_volume_names": qdrant_volumes,
            "qdrant_volume": volume_rows[0]}

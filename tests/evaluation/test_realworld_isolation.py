import json

import pytest

from scripts.realworld.isolation import validate_dedicated_qdrant
from scripts.realworld.ingest_corpus import verify_api


def _docker_fixture():
    labels_api = {"com.docker.compose.project": "omnirag",
                  "com.docker.compose.service": "api-eval-rw"}
    labels_qdrant = {"com.docker.compose.project": "omnirag",
                     "com.docker.compose.service": "qdrant-rw"}
    api = {
        "Id": "api-id", "Name": "/omnirag-api-eval-rw",
        "Config": {"Image": "omnirag-api-eval-rw", "Labels": labels_api,
                   "Env": ["OMNIRAG_QDRANT_URL=http://qdrant-rw:6333",
                           "OMNIRAG_QDRANT_COLLECTION=omnirag_eval_rw_test_corpus_nomic768"]},
        "State": {"Running": True, "Health": {"Status": "healthy"}},
        "NetworkSettings": {"Networks": {"omnirag_default": {"Aliases": ["api-eval-rw"]}}},
        "Mounts": [{"Type": "volume", "Name": "omnirag_omnirag_eval_rw_data", "Destination": "/app/data"}],
    }
    qdrant = {
        "Id": "qdrant-id", "Name": "/omnirag-qdrant-rw",
        "Config": {"Image": "qdrant/qdrant:latest", "Labels": labels_qdrant},
        "State": {"Running": True, "Health": {"Status": "healthy"}},
        "NetworkSettings": {
            "Networks": {"omnirag_default": {"Aliases": ["qdrant-rw"]}},
            "Ports": {"6333/tcp": [{"HostIp": "127.0.0.1", "HostPort": "6335"}]},
        },
        "Mounts": [{"Type": "volume", "Name": "omnirag_omnirag_eval_rw_qdrant_data",
                    "Destination": "/qdrant/storage"}],
    }
    old_qdrant = {
        "Id": "old-qdrant-id", "Name": "/omnirag-qdrant-1",
        "Config": {"Image": "qdrant/qdrant:latest",
                   "Labels": {"com.docker.compose.project": "omnirag",
                              "com.docker.compose.service": "qdrant"}},
        "State": {"Running": False}, "Mounts": [
            {"Type": "volume", "Name": "omnirag_qdrant_storage", "Destination": "/qdrant/storage"}],
    }
    protected_api = [
        {"Id": "dev-api-id", "Name": "/omnirag-api-1",
         "Config": {"Labels": {"com.docker.compose.project": "omnirag",
                                "com.docker.compose.service": "api"}},
         "State": {"Running": False}, "Mounts": [
             {"Type": "volume", "Name": "omnirag_omnirag_data", "Destination": "/app/data"}]},
        {"Id": "eval-api-id", "Name": "/omnirag-api-eval",
         "Config": {"Labels": {"com.docker.compose.project": "omnirag",
                                "com.docker.compose.service": "api-eval"}},
         "State": {"Running": False}, "Mounts": [
             {"Type": "volume", "Name": "omnirag_omnirag_eval_data", "Destination": "/app/data"}]},
    ]
    settings = {
        "user_id": "m1-ingest", "tenant_id": "realworld",
        "physical_collection": "omnirag_eval_rw_test_corpus_nomic768",
        "accepted_identities": [{"user_id": "m1-ingest", "tenant_id": "realworld"}],
        "limits": {"max_cost_usd": 0.0, "max_total_tokens": 4_200_000, "max_calls": 3_000},
    }
    volume = {"Name": "omnirag_omnirag_eval_rw_qdrant_data",
              "Labels": {"com.docker.compose.project": "omnirag",
                         "com.docker.compose.volume": "omnirag_eval_rw_qdrant_data"}}
    return {"api_container": api, "qdrant_container": qdrant,
            "other_qdrant_containers": [old_qdrant],
            "protected_containers": [*protected_api, old_qdrant],
            "existing_qdrant_volume_names": ["omnirag_qdrant_storage"],
            "qdrant_volume": volume, "api_settings": settings,
            "expected_collection": settings["physical_collection"],
            "qdrant_host_url": "http://localhost:6335", "user_id": "m1-ingest",
            "tenant_id": "realworld"}


def test_dedicated_qdrant_passes_with_verified_storage_and_signed_settings():
    proof = validate_dedicated_qdrant(**_docker_fixture())
    assert proof["verified"] is True
    assert proof["qdrant_volume"] == "omnirag_omnirag_eval_rw_qdrant_data"
    assert proof["live_protected_collection_fingerprints"] == "not_measured_dedicated_storage_boundary"
    assert proof["other_qdrant_containers"][0]["id"] == "old-qdrant-id"


def test_dedicated_qdrant_rejects_wrong_internal_url():
    fixture = _docker_fixture()
    fixture["api_container"]["Config"]["Env"][0] = "OMNIRAG_QDRANT_URL=http://qdrant:6333"
    with pytest.raises(RuntimeError, match="Qdrant URL"):
        validate_dedicated_qdrant(**fixture)


def test_dedicated_qdrant_rejects_unexpected_image():
    fixture = _docker_fixture()
    fixture["qdrant_container"]["Config"]["Image"] = "example/unrelated:latest"
    with pytest.raises(RuntimeError, match="expected Qdrant image"):
        validate_dedicated_qdrant(**fixture)


@pytest.mark.parametrize("reuse", ["listed", "mounted"])
def test_dedicated_qdrant_rejects_shared_or_reused_storage(reuse):
    fixture = _docker_fixture()
    if reuse == "listed":
        fixture["existing_qdrant_volume_names"].append("omnirag_omnirag_eval_rw_qdrant_data")
    else:
        fixture["other_qdrant_containers"][0]["Mounts"][0]["Name"] = "omnirag_omnirag_eval_rw_qdrant_data"
    with pytest.raises(RuntimeError, match="reused|mounted"):
        validate_dedicated_qdrant(**fixture)


def test_dedicated_qdrant_rejects_wrong_physical_collection():
    fixture = _docker_fixture()
    fixture["api_settings"]["physical_collection"] = "omnirag"
    with pytest.raises(RuntimeError, match="signed API settings"):
        validate_dedicated_qdrant(**fixture)


def test_signed_settings_identity_mismatch_fails_before_isolation_proof():
    budget = {"run_id": "iso-test", "max_cost_usd": 0.0, "max_total_tokens": 4_200_000,
              "max_calls": 3_000, "calls": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}

    def request(url, method, headers, body, timeout):
        if url.endswith("/health/ready"):
            return 200, {}, {"status": "ready"}
        return 200, {"X-OmniRAG-Budget-Usage": json.dumps(budget)}, {
            "user_id": "somebody-else", "tenant_id": "realworld",
        }

    with pytest.raises(RuntimeError, match="Signed API identity"):
        verify_api("http://localhost:8002", secret="test-secret", user_id="m1-ingest",
                   tenant_id="realworld", run_id="iso-test",
                   expected_collection="omnirag_eval_rw_test_corpus_nomic768",
                   embedding_alias="nomic-embed-text-768", timeout=5, request_fn=request)

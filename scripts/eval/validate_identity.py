"""Non-inference signed identity check against the isolated evaluation API."""

import hashlib
import hmac
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omnirag.config.settings import get_settings


def main():
    settings = get_settings()
    if settings.auth_shared_secret is None:
        raise RuntimeError("Trusted proxy shared secret is not configured")
    secret = settings.auth_shared_secret.get_secret_value()
    contexts = [
        ("phase7a-user-a", "phase7a-tenant-a"),
        ("phase7a-user-b", "phase7a-tenant-a"),
        ("phase7a-user-a", "phase7a-tenant-b"),
        ("phase7a-user-b", "phase7a-tenant-b"),
    ]
    results = []
    for user_id, tenant_id in contexts:
        stamp = str(int(time.time()))
        signed = f"{user_id}\n{tenant_id}\n{stamp}".encode()
        signature = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        headers = {"X-OmniRAG-User-ID": user_id, "X-OmniRAG-Tenant-ID": tenant_id,
                   "X-OmniRAG-Auth-Timestamp": stamp, "X-OmniRAG-Auth-Signature": signature}
        request = Request("http://localhost:8000/api/v1/settings", headers=headers)
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
            if response.status != 200 or payload.get("user_id") != user_id:
                raise RuntimeError("Signed identity did not propagate through the API")
            results.append({"http_status": response.status, "user_id": payload["user_id"],
                            "tenant_signature_context": tenant_id})
    print(json.dumps({"status": "passed", "inference_calls": 0, "contexts": results}, sort_keys=True))


if __name__ == "__main__":
    main()

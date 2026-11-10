"""Verification for identities asserted by a trusted, header-scrubbing reverse proxy."""

import hashlib
import hmac
import time


def identity_is_allowed(identity: dict[str, str], accepted_identities: tuple[tuple[str, str], ...]) -> bool:
    """Apply an optional instance-specific principal/tenant allowlist."""
    return not accepted_identities or (identity.get("user_id"), identity.get("tenant_id")) in accepted_identities


def verify_proxy_identity(user_id: str | None, tenant_id: str | None, timestamp: str | None,
                          signature: str | None, secret: str | None, *, now: float | None = None
                          ) -> dict[str, str] | None:
    if not all((user_id, tenant_id, timestamp, signature, secret)):
        return None
    if len(user_id) > 255 or len(tenant_id) > 255 or "\n" in user_id or "\n" in tenant_id:
        return None
    try:
        timestamp_value = int(timestamp)
    except ValueError:
        return None
    if abs((time.time() if now is None else now) - timestamp_value) > 60:
        return None
    signed = f"{user_id}\n{tenant_id}\n{timestamp}".encode()
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return None
    return {"user_id": user_id, "tenant_id": tenant_id}

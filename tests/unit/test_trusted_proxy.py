import hashlib
import hmac

from api.trusted_proxy import verify_proxy_identity


def test_signed_fresh_proxy_identity_is_accepted():
    timestamp = "1700000000"
    message = f"alice\nteam-a\n{timestamp}".encode()
    signature = hmac.new(b"proxy-secret", message, hashlib.sha256).hexdigest()
    assert verify_proxy_identity("alice", "team-a", timestamp, signature, "proxy-secret",
                                 now=1700000000) == {"user_id": "alice", "tenant_id": "team-a"}


def test_unsigned_tampered_and_stale_proxy_identity_is_rejected():
    timestamp = "1700000000"
    message = f"alice\nteam-a\n{timestamp}".encode()
    signature = hmac.new(b"proxy-secret", message, hashlib.sha256).hexdigest()
    assert verify_proxy_identity("alice", "team-b", timestamp, signature, "proxy-secret",
                                 now=1700000000) is None
    assert verify_proxy_identity("alice", "team-a", timestamp, signature, "proxy-secret",
                                 now=1700000100) is None
    assert verify_proxy_identity("alice", "team-a", timestamp, None, "proxy-secret",
                                 now=1700000000) is None

"""Small stdlib HTTP wrapper around a locally installed MiniCheck model.

This module is run in a separate process/container. It is intentionally not imported
by the application except in tests; PyTorch and MiniCheck never enter the API process.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
from typing import Protocol


MAX_BODY_BYTES = 1_000_000


class ClaimScorer(Protocol):
    def score(self, evidence: str, claim: str) -> float: ...


def cache_digest(cache_dir: str | Path) -> str:
    """Fingerprint all cached model files in sorted relative-path order."""
    root = Path(cache_dir)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("MiniCheck model cache is empty")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class MiniCheckScorer:
    def __init__(self, cache_dir: str | Path, expected_digest: str):
        root = Path(cache_dir)
        if not root.is_dir():
            raise RuntimeError("Configured local MiniCheck cache directory is missing")
        actual = cache_digest(root)
        if not hmac.compare_digest(actual, expected_digest):
            raise RuntimeError("Local MiniCheck cache digest does not match configured digest")
        # Prevent first-run model downloads. The cache and digest must be provisioned
        # separately after explicit model-download approval.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from minicheck.minicheck import MiniCheck
        self.model = MiniCheck(model_name="flan-t5-large", cache_dir=str(root))

    def score(self, evidence: str, claim: str) -> float:
        _, scores, _, _ = self.model.score(docs=[evidence], claims=[claim])
        score = float(scores[0])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("MiniCheck returned an invalid score")
        return score


def verify_request(request: object, scorer: ClaimScorer, *, model_id: str,
                   model_digest: str, threshold: float) -> dict[str, object]:
    """Validate the wire contract and return a provenance-pinned score result."""
    if not isinstance(request, dict):
        raise ValueError("Request body must be an object")
    if (request.get("model_id") != model_id or request.get("model_digest") != model_digest or
            request.get("threshold") != threshold):
        raise ValueError("Pinned model or threshold mismatch")
    claim, evidence = request.get("claim"), request.get("evidence")
    if not isinstance(claim, str) or not claim.strip() or len(claim) > 20_000:
        raise ValueError("Invalid claim")
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 500_000:
        raise ValueError("Invalid evidence")
    score = scorer.score(evidence, claim)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid model score")
    return {"label": "supported" if score >= threshold else "unsupported", "score": score,
            "threshold": threshold, "model_id": model_id, "model_digest": model_digest}


def make_handler(scorer: ClaimScorer, *, model_id: str, model_digest: str,
                 threshold: float, api_key: str | None = None):
    if not model_id or not model_digest or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Model identity and calibrated threshold are required")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health/ready":
                self.send_error(404)
                return
            body = json.dumps({"status": "ready", "model_id": model_id,
                               "model_digest": model_digest, "threshold": threshold},
                              separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/verify":
                self.send_error(404)
                return
            if self.headers.get_content_type() != "application/json":
                self.send_error(415)
                return
            if api_key and not hmac.compare_digest(
                    self.headers.get("Authorization", ""), f"Bearer {api_key}"):
                self.send_error(401)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > MAX_BODY_BYTES:
                    raise ValueError("Invalid request size")
                body_value = json.loads(self.rfile.read(size))
                result = verify_request(body_value, scorer, model_id=model_id,
                                        model_digest=model_digest, threshold=threshold)
                body = json.dumps(result, separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, TypeError, json.JSONDecodeError):
                self.send_error(400)
            except Exception:
                # Avoid exposing evidence, local paths, or model exceptions to clients.
                self.send_error(503)

        def log_message(self, _format, *_args):
            return

    return Handler


def main() -> None:
    cache = os.environ["MINICHECK_CACHE_DIR"]
    digest = os.environ["MINICHECK_MODEL_DIGEST"]
    model_id = os.environ.get("MINICHECK_MODEL_ID", "lytang/MiniCheck-Flan-T5-Large")
    threshold = float(os.environ["MINICHECK_THRESHOLD"])
    api_key = os.environ.get("MINICHECK_API_KEY")
    scorer = MiniCheckScorer(cache, digest)
    server = ThreadingHTTPServer((os.environ.get("MINICHECK_HOST", "127.0.0.1"),
                                  int(os.environ.get("MINICHECK_PORT", "8799"))),
                                 make_handler(scorer, model_id=model_id, model_digest=digest,
                                              threshold=threshold, api_key=api_key))
    server.serve_forever()


if __name__ == "__main__":
    main()

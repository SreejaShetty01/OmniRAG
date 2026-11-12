import csv
import json
from io import BytesIO

import pytest

from omnirag.config.settings import Settings
from eval.p3_grounding_scorer import score_grounding, write_audit_csv
from omnirag.generation.answer_pipeline import AnswerPipeline
from omnirag.models.response import ClaimVerificationStatus
from omnirag.generation.answer_generator import AnswerGenerator
from omnirag.models.chunk import AccessControlMetadata, Chunk
from omnirag.models.document import DocumentType
from omnirag.retrieval.contracts import (EvidenceAssessment, EvidenceAssessmentState, QueryAnalysis,
                                 QueryCharacteristics, RetrievalEvidence, RetrievalFilters, RetrievalResult)
from omnirag.services.claim_entailment.server import verify_request
from omnirag.verification.claim_extractor import extract_claims
from omnirag.verification.claim_verifier import verify_claims
from omnirag.verification.nli_verifier import (EntailmentLabel, EntailmentResult,
                                        HttpMiniCheckVerifier, UnconfiguredEntailmentVerifier)


MODEL_ID = "lytang/MiniCheck-Flan-T5-Large"
MODEL_DIGEST = "a" * 64
THRESHOLD = 0.7


class FakeScorer:
    def __init__(self):
        self.inputs = []

    def score(self, evidence, claim):
        self.inputs.append((evidence, claim))
        return 0.91 if "retention" in evidence.lower() else 0.2


class FakeProvider:
    def __init__(self, payload):
        self.payload, self.requests = payload, []

    def complete(self, request):
        from omnirag.llm.llm_client import ChatResponse
        self.requests.append(request)
        value = self.payload.pop(0) if isinstance(self.payload, list) else self.payload
        return ChatResponse(json.dumps(value), "fake", "test-model", 10, 5, 0.0)


def retrieval():
    chunk = Chunk(chunk_id="chunk-a", document_id="doc-a", parent_id="doc-a",
                  content="The retention period is 30 days.", document_type=DocumentType.TEXT,
                  content_hash="hash", source="policy.md", version="v1", section_path=["Retention"],
                  access_control=AccessControlMetadata())
    item = RetrievalEvidence(chunk=chunk, dense_rank=1, sparse_rank=1, fusion_rank=1)
    return RetrievalResult(original_query="How long?", normalized_query="How long?",
                           query_analysis=QueryAnalysis(original_query="How long?", normalized_query="How long?",
                                                       characteristics=QueryCharacteristics(),
                                                       extracted_filters=RetrievalFilters()),
                           applied_filters=RetrievalFilters(), authorization_scope_hash="scope",
                           evidence=[item], assessment=EvidenceAssessment(state=EvidenceAssessmentState.PARTIAL),
                           selected_token_estimate=10)


@pytest.fixture
def mocked_service(monkeypatch):
    scorer = FakeScorer()
    calls = []

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        payload = json.loads(request.data)
        result = verify_request(payload, scorer, model_id=MODEL_ID,
                                model_digest=MODEL_DIGEST, threshold=THRESHOLD)
        return Response(json.dumps(result).encode())

    monkeypatch.setattr("verification.nli_verifier.urlopen", fake_urlopen)
    return scorer, calls


def test_http_verifier_service_contract_multi_evidence_and_provenance(mocked_service):
    scorer, calls = mocked_service
    verifier = HttpMiniCheckVerifier("http://example.invalid/verify", MODEL_ID,
                                     MODEL_DIGEST, THRESHOLD, api_key="test-key")
    result = verifier.verify("The retention is 30 days.", ["First source.", "Retention policy."])
    assert result.label == EntailmentLabel.SUPPORTED
    assert result.raw_score == 0.91
    assert result.model_id == MODEL_ID and result.model_digest == MODEL_DIGEST
    assert scorer.inputs == [("First source.\n\nRetention policy.", "The retention is 30 days.")]
    assert calls[0][0].get_header("Authorization") == "Bearer test-key"


def test_service_rejects_unpinned_model_or_threshold():
    with pytest.raises(ValueError, match="Pinned model"):
        verify_request({"model_id": MODEL_ID, "model_digest": "b" * 64, "threshold": THRESHOLD,
                        "claim": "claim", "evidence": "evidence"}, FakeScorer(),
                       model_id=MODEL_ID, model_digest=MODEL_DIGEST, threshold=THRESHOLD)


def test_verifier_off_preserves_existing_uncertain_behavior():
    provider = FakeProvider({"answer": "Retention is 30 days [E1].", "claims": [
        {"claim": "Retention is 30 days.", "support_evidence_ids": ["E1"]}]})
    result = AnswerPipeline(AnswerGenerator(provider, "test-model"), UnconfiguredEntailmentVerifier()).answer(
        "How long?", retrieval())
    assert result.claims[0].status == ClaimVerificationStatus.UNCERTAIN
    assert "semantic_verification_unconfigured" in result.warnings
    assert len(provider.requests) == 1


def test_switch_on_uses_http_verifier_and_failure_is_uncertain(mocked_service, monkeypatch):
    verifier = HttpMiniCheckVerifier("http://example.invalid/verify", MODEL_ID,
                                     MODEL_DIGEST, THRESHOLD, api_key="test-key")
    claims = extract_claims({"claims": [{"claim": "Retention is 30 days.",
                                         "support_evidence_ids": ["E1"]}]})
    verified, warnings = verify_claims(claims, retrieval().evidence, verifier)
    assert verified[0].status == ClaimVerificationStatus.SUPPORTED
    assert verified[0].verification_score == 0.91
    assert verified[0].verifier_model_digest == MODEL_DIGEST
    assert not warnings

    def fail_urlopen(*args, **kwargs):
        from urllib.error import URLError
        raise URLError("offline")

    monkeypatch.setattr("verification.nli_verifier.urlopen", fail_urlopen)
    unavailable = HttpMiniCheckVerifier("http://example.invalid/verify", MODEL_ID,
                                        MODEL_DIGEST, THRESHOLD, timeout_seconds=0.2)
    failed, warnings = verify_claims(extract_claims({"claims": [{"claim": "Retention is 30 days.",
                                                                  "support_evidence_ids": ["E1"]}]}),
                                     retrieval().evidence, unavailable)
    assert failed[0].status == ClaimVerificationStatus.UNCERTAIN
    assert any("semantic_verification_failed" in warning for warning in warnings)


def test_pipeline_switch_on_calls_minicheck_and_records_provenance(mocked_service):
    scorer, _ = mocked_service
    verifier = HttpMiniCheckVerifier("http://example.invalid/verify", MODEL_ID,
                                     MODEL_DIGEST, THRESHOLD, api_key="test-key")
    generator = FakeProvider({"answer": "The retention period is 30 days [E1].", "claims": [
        {"claim": "The retention period is 30 days.", "support_evidence_ids": ["E1"]}]})
    result = AnswerPipeline(AnswerGenerator(generator, "test-model"), verifier).answer(
        "How long?", retrieval(), semantic_verification=True)
    assert result.claims[0].status == ClaimVerificationStatus.SUPPORTED
    assert result.claims[0].verifier_model_digest == MODEL_DIGEST
    assert len(scorer.inputs) == 1


def test_semantic_switch_off_keeps_deterministic_checks_and_bounded_correction():
    generator = FakeProvider({"answer": "It lasts 90 days [E1].", "claims": [
        {"claim": "It lasts 90 days.", "support_evidence_ids": ["E1"]}]})
    generator.payload = [
        {"answer": "It lasts 90 days [E1].", "claims": [
            {"claim": "It lasts 90 days.", "support_evidence_ids": ["E1"]}]},
        {"answer": "The retention period is 30 days [E1].", "claims": [
            {"claim": "The retention period is 30 days.", "support_evidence_ids": ["E1"]}]},
    ]
    result = AnswerPipeline(AnswerGenerator(generator, "test-model"), UnconfiguredEntailmentVerifier()).answer(
        "How long?", retrieval(), semantic_verification=False, allow_correction=True)
    assert len(generator.requests) == 2
    assert result.execution_metadata["correction_attempts"] == 1
    assert result.claims[0].status == ClaimVerificationStatus.UNCERTAIN
    assert "numerical_check" in result.claims[0].methods


def test_semantic_unsupported_uses_existing_one_correction_then_abstains(monkeypatch):
    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def unsupported_response(request, timeout):
        payload = json.loads(request.data)
        value = {"label": "unsupported", "score": 0.1, "threshold": THRESHOLD,
                 "model_id": MODEL_ID, "model_digest": MODEL_DIGEST}
        assert payload["model_id"] == MODEL_ID
        return Response(json.dumps(value).encode())

    monkeypatch.setattr("verification.nli_verifier.urlopen", unsupported_response)
    verifier = HttpMiniCheckVerifier("http://example.invalid/verify", MODEL_ID,
                                     MODEL_DIGEST, THRESHOLD)
    generator = FakeProvider([
        {"answer": "It lasts 30 days [E1].", "claims": [
            {"claim": "It lasts 30 days.", "support_evidence_ids": ["E1"]}]},
        {"answer": "It lasts 30 days [E1].", "claims": [
            {"claim": "It lasts 30 days.", "support_evidence_ids": ["E1"]}]},
    ])
    result = AnswerPipeline(AnswerGenerator(generator, "test-model"), verifier).answer("How long?", retrieval())
    assert result.execution_metadata["correction_attempts"] == 1
    assert len(generator.requests) == 2
    assert result.abstained
    assert result.abstention_reason == "unsupported_claims_remain_after_bounded_correction"


def test_existing_numeric_identifier_checks_precede_remote_verifier():
    class MustNotCall:
        def verify(self, claim, evidence):
            raise AssertionError("numeric check should run first")

    claims = extract_claims({"claims": [{"claim": "Retention is 90 days.",
                                         "support_evidence_ids": ["E1"]}]})
    verified, _ = verify_claims(claims, retrieval().evidence, MustNotCall())
    assert verified[0].status == ClaimVerificationStatus.UNSUPPORTED
    assert "numerical_check" in verified[0].methods


def test_existing_claim_extractor_splits_answer_sentences():
    claims = extract_claims({"answer": "The policy lasts thirty days. It covers active accounts."})
    assert [claim.claim_id for claim in claims] == ["CL1", "CL2"]
    assert all(claim.claim_text.endswith(".") for claim in claims)


def test_minicheck_requires_digest_and_dev_calibrated_threshold():
    with pytest.raises(ValueError, match="MiniCheck requires"):
        Settings(_env_file=None, verification_provider="minicheck_http",
                 verification_endpoint="http://localhost:8799/verify",
                 verification_model_id=MODEL_ID)


def test_offline_scorer_denominators_and_csv_labels_are_blank(tmp_path):
    artifacts = [{"run_id": "r1", "case_id": "c1", "status": "completed", "abstained": True,
                  "claims": [{"claim_id": "CL1", "claim_text": "x", "status": "unsupported"}],
                  "citations": [{"citation_id": "C1", "evidence_id": "E1", "chunk_id": "ch1"}]}]
    cases = {"c1": {"expected_abstention": False}}
    csv_path = tmp_path / "audit.csv"
    assert write_audit_csv(csv_path, artifacts, fill_empty_slots=True) == 80
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert sum(row["record_type"] == "claim" for row in rows) == 50
    assert sum(row["record_type"] == "citation" for row in rows) == 30
    assert all(not row["human_label"] for row in rows)
    result = score_grounding(artifacts, cases, [{"record_type": "citation", "human_label": "supported",
                                                "prediction": "supported", "evidence_references": "E1"}])
    assert result["unsupported_claim_rate"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert result["false_abstention_rate"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert result["citation_entailment_precision"]["rate"] == 1.0
    assert result["citation_entailment_coverage"]["rate"] == 1.0

"""Traceable records for each bounded Research retrieval step."""

from pydantic import BaseModel, ConfigDict, Field


class LedgerEvidenceReference(BaseModel):
    document_id: str
    chunk_id: str
    source: str
    version: str | None = None
    section_path: list[str] = Field(default_factory=list)
    excerpt: str


class ResearchLedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    round_number: int
    sub_question: str
    evidence_references: list[LedgerEvidenceReference] = Field(default_factory=list)
    supporting_findings: list[str] = Field(default_factory=list)
    contradictory_findings: list[dict[str, object]] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)
    evidence_sufficiency_state: str
    verification_status: str = "not_run"


class EvidenceLedger:
    def __init__(self):
        self.entries: list[ResearchLedgerEntry] = []

    def record(self, entry: ResearchLedgerEntry) -> None:
        self.entries.append(entry)

    def as_models(self) -> list[ResearchLedgerEntry]:
        return list(self.entries)

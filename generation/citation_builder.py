"""Build deterministic citation references from actual Phase 3 evidence."""

from models.response import Citation, EvidenceReference
from retrieval.contracts import RetrievalEvidence


def evidence_id_map(evidence: list[RetrievalEvidence]) -> dict[str, RetrievalEvidence]:
    return {f"E{index}": item for index, item in enumerate(evidence, start=1)}


def build_references(evidence: list[RetrievalEvidence]
                     ) -> tuple[list[EvidenceReference], list[Citation]]:
    references, citations = [], []
    for evidence_id, item in evidence_id_map(evidence).items():
        chunk = item.chunk
        reference = EvidenceReference(
            evidence_id=evidence_id, document_id=chunk.document_id, chunk_id=chunk.chunk_id,
            source=chunk.source, title=chunk.metadata.get("title"),
            version=chunk.version,
            page_number=chunk.page_number, section_path=chunk.section_path,
            excerpt=chunk.content[:500],
        )
        references.append(reference)
        citations.append(Citation(
            citation_id=f"C{len(citations) + 1}", source=chunk.source,
            document_id=chunk.document_id, chunk_id=chunk.chunk_id,
            version=chunk.version,
            title=reference.title, page_number=chunk.page_number,
            section_path=chunk.section_path, evidence_id=evidence_id,
            source_location=(f"line {chunk.metadata.get('start_line')}" if chunk.metadata.get("start_line") else None),
        ))
    return references, citations

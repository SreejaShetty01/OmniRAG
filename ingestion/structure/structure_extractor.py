"""Normalization passes over parser-independent structural blocks."""

from ingestion.structure.cross_reference import extract_cross_references
from ingestion.structure.schema import NormalizedDocument


class StructureExtractor:
    def extract(self, document: NormalizedDocument) -> NormalizedDocument:
        extract_cross_references(document.blocks)
        return document

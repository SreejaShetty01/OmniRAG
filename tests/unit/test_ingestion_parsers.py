from pathlib import Path

import pytest

from omnirag.ingestion.loaders.document_loader import DocumentLoader
from omnirag.ingestion.loaders.code_loader import CodeLoader
from omnirag.ingestion.loaders.base import SourceLoadError
from omnirag.ingestion.parsers.code_parser import CodeParser
from omnirag.ingestion.parsers.document_parser import DocumentParser
from omnirag.ingestion.structure.schema import ContentKind
from omnirag.ingestion.loaders.web_loader import WebLoader


def test_markdown_structure_and_references_survive_parse(tmp_path: Path):
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\n## Install\n\nSee [setup](setup.md).\n\n| A | B |\n| - | - |\n| 1 | 2 |", encoding="utf-8")
    normalized = DocumentParser().parse(DocumentLoader().load(path))
    assert normalized.document.title == "Guide"
    assert [b.content for b in normalized.blocks if b.content_type is ContentKind.HEADING] == ["Guide", "Install"]
    assert any(b.content_type is ContentKind.TABLE for b in normalized.blocks)
    assert any("setup.md" in b.links for b in normalized.blocks)


def test_html_preserves_headings_links_and_table(tmp_path: Path):
    path = tmp_path / "page.html"
    path.write_text("<html><head><title>Page</title></head><body><h1>Intro</h1><p><a href='/more'>More</a></p><table><tr><th>A</th></tr><tr><td>B</td></tr></table></body></html>", encoding="utf-8")
    normalized = DocumentParser().parse(DocumentLoader().load(path))
    assert normalized.document.title == "Page"
    assert any("/more" in b.links for b in normalized.blocks)
    assert any(b.content_type is ContentKind.TABLE for b in normalized.blocks)


def test_plain_text_preserves_paragraphs(tmp_path: Path):
    path = tmp_path / "plain.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    normalized = DocumentParser().parse(DocumentLoader().load(path))
    assert [block.content for block in normalized.blocks] == ["First paragraph.", "Second paragraph."]


def test_docx_preserves_heading_and_table(tmp_path: Path):
    from docx import Document as DocxDocument
    path = tmp_path / "report.docx"
    docx = DocxDocument()
    docx.core_properties.title = "Quarterly Report"
    docx.add_heading("Summary", level=1)
    docx.add_paragraph("Results are listed below.")
    table = docx.add_table(rows=2, cols=1)
    table.cell(0, 0).text = "Metric"
    table.cell(1, 0).text = "Value"
    docx.save(path)
    normalized = DocumentParser().parse(DocumentLoader().load(path))
    assert normalized.document.title == "Quarterly Report"
    assert any(block.content_type is ContentKind.HEADING for block in normalized.blocks)
    assert any(block.content_type is ContentKind.TABLE for block in normalized.blocks)


def test_pdf_preserves_page_numbers(tmp_path: Path):
    import fitz
    path = tmp_path / "brief.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Page one content")
    pdf.save(path)
    pdf.close()
    normalized = DocumentParser().parse(DocumentLoader().load(path))
    assert normalized.blocks[0].page_number == 1
    assert "Page one content" in normalized.blocks[0].content


def test_text_loading_rejects_oversized_and_malformed_file(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(SourceLoadError):
        DocumentParser().parse(DocumentLoader().load(path))
    path.write_text("0123456789", encoding="utf-8")
    with pytest.raises(SourceLoadError):
        DocumentLoader(max_bytes=4).load(path)


def test_python_parser_extracts_symbols_imports_docstrings_and_locations(tmp_path: Path):
    path = tmp_path / "sample.py"
    path.write_text('"""Module info"""\nimport os\n\nclass Store:\n    """Store docs"""\n    def get(self):\n        return os.getcwd()\n', encoding="utf-8")
    blocks = CodeParser().parse(CodeLoader().load(path)).blocks
    symbols = {b.symbol_name: b for b in blocks if b.symbol_name}
    assert {"Store", "get", "os"}.issubset(symbols)
    assert symbols["get"].parent_id == symbols["Store"].block_id
    assert symbols["get"].start_line == 6
    assert symbols["Store"].metadata["docstring"] == "Store docs"


def test_invalid_python_returns_meaningful_error(tmp_path: Path):
    path = tmp_path / "broken.py"
    path.write_text("def nope(:\n", encoding="utf-8")
    with pytest.raises(SourceLoadError, match="Invalid Python syntax"):
        CodeParser().parse(CodeLoader().load(path))


def test_web_loader_rejects_private_network_targets():
    with pytest.raises(SourceLoadError, match="private or non-public"):
        WebLoader._validate_public_url("http://127.0.0.1:8080/internal")


def test_web_loader_reads_html_with_bounded_source_metadata(monkeypatch):
    import email.message
    import socket
    import omnirag.ingestion.loaders.web_loader as web_module

    monkeypatch.setattr(socket, "getaddrinfo", lambda *args: [(None, None, None, None, ("93.184.216.34", 80))])

    class FakeResponse:
        headers = email.message.Message()

        def __init__(self):
            self.headers["Content-Type"] = "text/html; charset=utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return b"<h1>Remote</h1>"

    class FakeOpener:
        def open(self, request, timeout):
            assert timeout > 0
            return FakeResponse()

    monkeypatch.setattr(web_module, "build_opener", lambda *args: FakeOpener())
    loaded = WebLoader(max_bytes=100).load("https://example.org/page")
    assert loaded.source_type == "web"
    assert loaded.media_type == "text/html"
    assert b"Remote" in loaded.data


def test_typescript_parser_extracts_functions_and_imports(tmp_path: Path):
    path = tmp_path / "sample.ts"
    path.write_text('import { helper } from "./helper";\nexport function run() { return helper(); }\n', encoding="utf-8")
    blocks = CodeParser().parse(CodeLoader().load(path)).blocks
    assert any(block.metadata.get("symbol_kind") == "function" and block.symbol_name == "run" for block in blocks)
    assert any(block.content_type is ContentKind.IMPORT for block in blocks)


def test_unsupported_code_language_uses_safe_generic_fallback(tmp_path: Path):
    path = tmp_path / "Example.java"
    path.write_text("class Example {\n  void run() {}\n}\n", encoding="utf-8")
    normalized = CodeParser().parse(CodeLoader().load(path))
    assert normalized.metadata["parser"] == "generic-line-boundary-fallback"
    assert normalized.blocks[0].metadata["generic_fallback"] is True


def test_proto_schema_uses_safe_generic_code_fallback(tmp_path: Path):
    path = tmp_path / "points.proto"
    path.write_text('message Point {\n  string id = 1;\n}\n', encoding="utf-8")
    normalized = CodeParser().parse(CodeLoader().load(path))
    assert normalized.document.document_type == "code"
    assert normalized.metadata["parser"] == "generic-line-boundary-fallback"
    assert normalized.blocks[0].metadata["file_extension"] == ".proto"


def test_restructured_text_is_loaded_as_plain_text(tmp_path: Path):
    path = tmp_path / "guide.rst"
    path.write_text("Guide\n=====\n\nA supported RST source.", encoding="utf-8")
    normalized = DocumentParser().parse(DocumentLoader().load(path))
    assert normalized.document.document_type == "text"
    assert "Guide" in " ".join(block.content for block in normalized.blocks)

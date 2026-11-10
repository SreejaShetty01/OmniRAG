"""Structure-preserving parser for PDF, DOCX, TXT, Markdown, and HTML."""

import html
import re
from html.parser import HTMLParser
from pathlib import PurePath

from ingestion.loaders.base import SourceLoadError
from ingestion.structure.normalize import make_normalized_document, stable_id
from ingestion.structure.schema import ContentBlock, ContentKind, LoadedSource, NormalizedDocument
from models.document import DocumentType


class _HtmlBlocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[ContentBlock] = []
        self.headings: list[str] = []
        self.title_parts: list[str] = []
        self._tag: str | None = None
        self._parts: list[str] = []
        self._href: str | None = None
        self._table_rows: list[list[str]] = []
        self._table_cell: list[str] | None = None
        self._in_title = False
        self._skip = 0
        self._links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = True
        if tag == "a":
            self._href = attrs_map.get("href")
        if tag == "tr":
            self._table_rows.append([])
        if tag in {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            self._links = []
            if tag == "li" and self._tag == "li":
                self._flush()
            self._tag, self._parts = tag, []
        if tag in {"td", "th"}:
            self._table_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = False
        if tag in {"td", "th"} and self._table_cell is not None:
            if self._table_rows:
                self._table_rows[-1].append("".join(self._table_cell).strip())
            self._table_cell = None
        if self._tag == tag:
            value = re.sub(r"\s+", " ", "".join(self._parts)).strip()
            if value:
                if tag.startswith("h") and tag[1:].isdigit():
                    level = int(tag[1:])
                    self.headings = self.headings[: level - 1]
                    self.headings.append(value)
                    kind = ContentKind.HEADING
                elif tag == "li":
                    kind = ContentKind.LIST
                else:
                    kind = ContentKind.PARAGRAPH
                self.blocks.append(ContentBlock(
                    stable_id("blk", f"{self.headings}:{value}"), value, kind,
                    section_path=list(self.headings), heading=self.headings[-1] if self.headings else None,
                    links=self._links.copy(),
                ))
            self._links = []
            self._tag = None
        if tag == "table" and self._table_rows:
            rows = "\n".join(" | ".join(row) for row in self._table_rows)
            if rows.strip():
                self.blocks.append(ContentBlock(
                    stable_id("blk", f"table:{rows}"), rows, ContentKind.TABLE,
                    section_path=list(self.headings), heading=self.headings[-1] if self.headings else None,
                ))
            self._table_rows = []
        if tag == "a":
            self._href = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title_parts.append(data)
        if self._table_cell is not None:
            self._table_cell.append(data)
        if self._tag:
            self._parts.append(data)
        if data.strip() and self._href and self._tag:
            self._links.append(self._href)

    def _flush(self) -> None:
        value = " ".join(self._parts).strip()
        if value:
            self.blocks.append(ContentBlock(stable_id("blk", value), value, ContentKind.LIST))


class DocumentParser:
    def parse(self, source: LoadedSource) -> NormalizedDocument:
        suffix = PurePath(source.filename).suffix.lower()
        try:
            if suffix == ".pdf":
                structured = self._docling(source, DocumentType.PDF)
                if structured is not None:
                    return structured
                return self._pdf(source)
            if suffix == ".docx":
                structured = self._docling(source, DocumentType.DOCX)
                if structured is not None:
                    return structured
                return self._docx(source)
            text = source.data.decode("utf-8-sig")
            if source.media_type in {"text/html", "application/xhtml+xml"}:
                return self._html(source, text)
            if suffix in {".md", ".markdown"}:
                return self._markdown(source, text)
            if suffix in {".html", ".htm"} or source.media_type in {"text/html", "application/xhtml+xml"}:
                return self._html(source, text)
            return self._text(source, text)
        except SourceLoadError:
            raise
        except Exception as exc:
            raise SourceLoadError(f"Unable to parse {source.filename}: {exc}") from exc

    def _docling(self, source: LoadedSource, document_type: DocumentType) -> NormalizedDocument | None:
        """Prefer Docling's layout-aware conversion when the optional extra is installed."""
        try:
            from io import BytesIO
            from docling.datamodel.base_models import DocumentStream
            from docling.document_converter import DocumentConverter
        except ImportError:
            return None
        try:
            stream = DocumentStream(name=source.filename, stream=BytesIO(source.data))
            converted = DocumentConverter().convert(stream).document
            markdown = converted.export_to_markdown()
            markdown_source = LoadedSource(source.source, source.filename + ".md", "text/markdown",
                                           markdown.encode("utf-8"), source.source_type)
            normalized = self._markdown(markdown_source, markdown)
            normalized.document.document_type = document_type
            normalized.document.content_hash = hashlib.sha256(source.data).hexdigest()
            normalized.document.source = source.source
            normalized.document.source_type = source.source_type
            normalized.document.metadata["media_type"] = source.media_type
            normalized.metadata["parser"] = "docling"
            return normalized
        except Exception:
            # Fall back to the format-native parser, which reports malformed input explicitly.
            return None

    def _text(self, source: LoadedSource, text: str) -> NormalizedDocument:
        blocks = [ContentBlock(stable_id("blk", p.strip()), p.strip())
                  for i, p in enumerate(re.split(r"\n\s*\n", text)) if p.strip()]
        if not blocks:
            raise SourceLoadError("Document contains no extractable text")
        return make_normalized_document(source, PurePath(source.filename).stem, DocumentType.TEXT, blocks)

    def _markdown(self, source: LoadedSource, text: str) -> NormalizedDocument:
        blocks: list[ContentBlock] = []
        headings: list[str] = []
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading_match:
                level, title = len(heading_match.group(1)), heading_match.group(2)
                headings = headings[:level - 1] + [title]
                blocks.append(ContentBlock(stable_id("blk", f"heading:{'/'.join(headings)}"), title, ContentKind.HEADING,
                                           section_path=headings.copy(), heading=title))
                i += 1
                continue
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                marker = line.strip()[:3]
                lang = line.strip()[3:].strip() or None
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(marker):
                    code_lines.append(lines[i]); i += 1
                if i >= len(lines):
                    raise SourceLoadError("Markdown contains an unterminated fenced code block")
                content = "\n".join(code_lines)
                blocks.append(ContentBlock(stable_id("blk", f"{headings}:{content}"), content, ContentKind.CODE,
                                           section_path=headings.copy(), heading=headings[-1] if headings else None,
                                           language=lang))
                i += 1
                continue
            if not line.strip():
                i += 1
                continue
            kind = ContentKind.TABLE if "|" in line and i + 1 < len(lines) and re.match(r"\s*\|?\s*:?-+", lines[i + 1]) else None
            if kind:
                group = [line]
                i += 1
                while i < len(lines) and "|" in lines[i] and lines[i].strip():
                    group.append(lines[i]); i += 1
                content = "\n".join(group)
                blocks.append(ContentBlock(stable_id("blk", f"{headings}:{content}"), content, kind,
                                           section_path=headings.copy(), heading=headings[-1] if headings else None))
                continue
            is_list = bool(re.match(r"\s*(?:[-*+] |\d+[.)] )", line))
            group = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^#{1,6}\s", lines[i]) and bool(re.match(r"\s*(?:[-*+] |\d+[.)] )", lines[i])) == is_list:
                group.append(lines[i].strip()); i += 1
            content = "\n".join(group)
            links = re.findall(r"\[[^\]]+\]\(([^)]+)\)|\[[^\]]+\]\[([^]]+)\]", content)
            flattened_links = [a or b for a, b in links]
            blocks.append(ContentBlock(stable_id("blk", f"{headings}:{content}"), content,
                                       ContentKind.LIST if is_list else ContentKind.PARAGRAPH,
                                       section_path=headings.copy(), heading=headings[-1] if headings else None,
                                       links=flattened_links))
        if not blocks:
            raise SourceLoadError("Document contains no extractable content")
        title = next((b.content for b in blocks if b.content_type is ContentKind.HEADING), PurePath(source.filename).stem)
        return make_normalized_document(source, title, DocumentType.MARKDOWN, blocks)

    def _html(self, source: LoadedSource, text: str) -> NormalizedDocument:
        parser = _HtmlBlocks()
        parser.feed(text)
        parser.close()
        if not parser.blocks:
            raise SourceLoadError("HTML contains no extractable content")
        title = " ".join(parser.title_parts).strip() or PurePath(source.filename).stem
        for block in parser.blocks:
            block.links = list(dict.fromkeys(block.links))
        return make_normalized_document(source, html.unescape(title), DocumentType.HTML, parser.blocks)

    def _pdf(self, source: LoadedSource) -> NormalizedDocument:
        try:
            import fitz
        except ImportError as exc:
            raise SourceLoadError("PDF support requires PyMuPDF") from exc
        blocks = []
        try:
            with fitz.open(stream=source.data, filetype="pdf") as pdf:
                title = pdf.metadata.get("title") or PurePath(source.filename).stem
                for page_no, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        blocks.append(ContentBlock(stable_id("blk", f"{page_no}:{text}"), text,
                                                   ContentKind.PARAGRAPH, page_number=page_no))
        except Exception as exc:
            raise SourceLoadError(f"Malformed or unreadable PDF: {exc}") from exc
        if not blocks:
            raise SourceLoadError("PDF contains no extractable text; OCR is not enabled")
        return make_normalized_document(source, title, DocumentType.PDF, blocks)

    def _docx(self, source: LoadedSource) -> NormalizedDocument:
        try:
            from docx import Document as DocxDocument
            from docx.table import Table
            from docx.text.paragraph import Paragraph
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from io import BytesIO
            document = DocxDocument(BytesIO(source.data))
            blocks: list[ContentBlock] = []
            headings: list[str] = []
            for child in document.element.body.iterchildren():
                if isinstance(child, CT_P):
                    paragraph = Paragraph(child, document)
                    content = paragraph.text.strip()
                    if not content:
                        continue
                    style = (paragraph.style.name or "") if paragraph.style else ""
                    if style.lower().startswith("heading"):
                        try:
                            level = int(style.split()[-1])
                        except ValueError:
                            level = 1
                        headings = headings[:max(level - 1, 0)] + [content]
                        kind = ContentKind.HEADING
                    else:
                        kind = ContentKind.LIST if "list" in style.lower() else ContentKind.PARAGRAPH
                    blocks.append(ContentBlock(stable_id("blk", f"{headings}:{content}"), content,
                                               kind, section_path=headings.copy(),
                                               heading=headings[-1] if headings else None))
                elif isinstance(child, CT_Tbl):
                    table = Table(child, document)
                    content = "\n".join(" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows)
                    if content.strip():
                        blocks.append(ContentBlock(stable_id("blk", content), content, ContentKind.TABLE,
                                                   section_path=headings.copy(), heading=headings[-1] if headings else None))
            title = document.core_properties.title or PurePath(source.filename).stem
        except ImportError as exc:
            raise SourceLoadError("DOCX support requires python-docx") from exc
        except Exception as exc:
            raise SourceLoadError(f"Malformed or unreadable DOCX: {exc}") from exc
        if not blocks:
            raise SourceLoadError("DOCX contains no extractable content")
        return make_normalized_document(source, title, DocumentType.DOCX, blocks)

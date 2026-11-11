"""Syntax-aware code parsing for Python, JavaScript, and TypeScript."""

import ast
import hashlib
import io
import re
import tokenize
from pathlib import PurePath

from omnirag.ingestion.loaders.code_loader import CODE_LANGUAGES, GENERIC_CODE_EXTENSIONS
from omnirag.ingestion.loaders.base import SourceLoadError
from omnirag.ingestion.structure.normalize import make_normalized_document
from omnirag.ingestion.structure.schema import ContentBlock, ContentKind, LoadedSource, NormalizedDocument
from omnirag.models.document import DocumentType


class CodeParser:
    def parse(self, source: LoadedSource) -> NormalizedDocument:
        suffix = PurePath(source.filename).suffix.lower()
        language = CODE_LANGUAGES.get(suffix)
        if not language:
            if suffix not in GENERIC_CODE_EXTENSIONS:
                raise SourceLoadError(f"Unsupported source language: {suffix or '(no extension)'}")
            return self._generic(source, suffix)
        text = source.data.decode("utf-8-sig")
        if language == "python":
            blocks = self._python(text, source.filename)
        else:
            blocks = self._tree_sitter(text, language, source.filename)
        if not blocks:
            raise SourceLoadError("Source file contains no recognized code structures")
        doc = make_normalized_document(source, PurePath(source.filename).name, DocumentType.CODE, blocks, language)
        doc.metadata["language"] = language
        return doc

    def _generic(self, source: LoadedSource, extension: str) -> NormalizedDocument:
        text = source.data.decode("utf-8-sig")
        blocks = []
        start_line = 1
        for group in re.split(r"\n\s*\n", text):
            content = group.strip("\n")
            if content.strip():
                lines = content.splitlines()
                digest = hashlib.sha256(f"{source.filename}:{start_line}:{content}".encode()).hexdigest()[:20]
                blocks.append(ContentBlock(f"code_{digest}", content, ContentKind.CODE,
                                           language=None, start_line=start_line,
                                           end_line=start_line + len(lines) - 1,
                                           metadata={"generic_fallback": True,
                                                     "file_extension": extension}))
            start_line += len(group.splitlines()) + 1
        if not blocks:
            raise SourceLoadError("Source file contains no extractable code")
        normalized = make_normalized_document(source, PurePath(source.filename).name, DocumentType.CODE, blocks)
        normalized.metadata["parser"] = "generic-line-boundary-fallback"
        normalized.metadata["file_extension"] = extension
        return normalized

    def _python(self, text: str, filename: str) -> list[ContentBlock]:
        try:
            tree = ast.parse(text, filename=filename)
        except SyntaxError as exc:
            raise SourceLoadError(f"Invalid Python syntax at line {exc.lineno}: {exc.msg}") from exc
        lines = text.splitlines()
        blocks: list[ContentBlock] = []
        def walk(node: ast.AST, parent_symbol: str | None = None) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = child.name
                    kind = ContentKind.CODE
                    start, end = child.lineno, getattr(child, "end_lineno", child.lineno)
                    content = "\n".join(lines[start - 1:end])
                    block_id = "sym_" + hashlib.sha256(f"{filename}:{name}:{start}:{content}".encode()).hexdigest()[:20]
                    docstring = ast.get_docstring(child, clean=False)
                    blocks.append(ContentBlock(block_id, content, kind, parent_id=parent_symbol,
                                               language="python", symbol_name=name, start_line=start,
                                               end_line=end, metadata={"symbol_kind": type(child).__name__,
                                                                       "docstring": docstring}))
                    walk(child, block_id)
                elif isinstance(child, (ast.Import, ast.ImportFrom)):
                    start = child.lineno
                    content = lines[start - 1]
                    module = child.module if isinstance(child, ast.ImportFrom) else ",".join(a.name for a in child.names)
                    blocks.append(ContentBlock("imp_" + hashlib.sha256(f"{filename}:{start}:{content}".encode()).hexdigest()[:20],
                                               content, ContentKind.IMPORT, parent_id=parent_symbol,
                                               language="python", symbol_name=module, start_line=start,
                                               end_line=start, links=[module] if module else []))
                elif isinstance(child, (ast.Expr,)) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str):
                    if child.lineno == 1 or (child.lineno == 2 and lines[0].startswith("#!")):
                        content = child.value.value
                        blocks.append(ContentBlock("doc_" + hashlib.sha256(f"{filename}:{content}".encode()).hexdigest()[:20],
                                                   content, ContentKind.COMMENT, language="python",
                                                   start_line=child.lineno, end_line=getattr(child, "end_lineno", child.lineno),
                                                   metadata={"docstring": True}))
                else:
                    walk(child, parent_symbol)
        walk(tree)
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT:
                    blocks.append(ContentBlock(
                        "com_" + hashlib.sha256(f"{filename}:{token.start}:{token.string}".encode()).hexdigest()[:20],
                        token.string, ContentKind.COMMENT, language="python",
                        start_line=token.start[0], end_line=token.end[0],
                    ))
        except (tokenize.TokenError, IndentationError) as exc:
            raise SourceLoadError(f"Unable to tokenize Python comments: {exc}") from exc
        blocks.sort(key=lambda b: (b.start_line or 0, b.parent_id or ""))
        return blocks

    def _tree_sitter(self, text: str, language: str, filename: str) -> list[ContentBlock]:
        try:
            from tree_sitter_language_pack import get_parser
            parser = get_parser(language)
        except ImportError as exc:
            raise SourceLoadError("JavaScript/TypeScript parsing requires tree-sitter-language-pack") from exc
        except Exception as exc:
            raise SourceLoadError(f"Tree-sitter grammar unavailable for {language}: {exc}") from exc
        raw = text.encode("utf-8")
        tree = parser.parse(raw)
        if tree.root_node.has_error:
            # Error-tolerant trees still contain usable declarations; the parse status is preserved.
            parse_error = True
        else:
            parse_error = False
        targets = {
            "function_declaration": "function", "function_expression": "function",
            "generator_function_declaration": "function", "class_declaration": "class",
            "method_definition": "method", "import_statement": "import",
            "import_declaration": "import", "interface_declaration": "interface",
            "type_alias_declaration": "type", "comment": "comment",
            "lexical_declaration": "declaration", "variable_declaration": "declaration",
        }
        blocks: list[ContentBlock] = []

        def visit(node, parent_symbol: str | None = None) -> None:
            kind_name = targets.get(node.type)
            current_parent = parent_symbol
            if kind_name:
                start, end = node.start_point.row + 1, node.end_point.row + 1
                snippet = raw[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                name_node = node.child_by_field_name("name")
                symbol_name = raw[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace") if name_node else None
                source_node = node.child_by_field_name("source")
                import_target = (raw[source_node.start_byte:source_node.end_byte].decode("utf-8", errors="replace").strip("\"'` ")
                                 if source_node else symbol_name)
                content_kind = ContentKind.IMPORT if kind_name == "import" else ContentKind.COMMENT if kind_name == "comment" else ContentKind.CODE
                block_id = "sym_" + hashlib.sha256(f"{filename}:{start}:{kind_name}:{snippet}".encode()).hexdigest()[:20]
                blocks.append(ContentBlock(block_id, snippet, content_kind, parent_id=parent_symbol,
                                           language=language, symbol_name=symbol_name,
                                           start_line=start, end_line=end,
                                           links=[import_target] if kind_name == "import" and import_target else [],
                                           metadata={"symbol_kind": kind_name, "parse_error_recovered": parse_error}))
                if kind_name in {"class", "function", "method"}:
                    current_parent = block_id
            for child in node.children:
                visit(child, current_parent)

        visit(tree.root_node)
        blocks.sort(key=lambda b: (b.start_line or 0, -(b.end_line or 0)))
        return blocks

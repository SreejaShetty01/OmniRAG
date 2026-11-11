from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
SOURCE = ROOT / "backend" / "src" / "omnirag"
PACKAGE_ROOTS = (
    "agent",
    "api",
    "config",
    "generation",
    "indexing",
    "infrastructure",
    "ingestion",
    "llm",
    "models",
    "profiles",
    "retrieval",
    "services",
    "utils",
    "verification",
)
MODULES = set(PACKAGE_ROOTS)
TEXT_SUFFIXES = {
    ".md", ".rst", ".txt", ".toml", ".yml", ".yaml", ".ini", ".cfg",
    ".ps1", ".sh", ".env", ".example", ".json",
}
FROM_IMPORT = re.compile(
    r"(?m)^([ \t]*from[ \t]+)([A-Za-z_][A-Za-z0-9_.]*)([ \t]+import\b)"
)
IMPORT_LINE = re.compile(r"(?m)^([ \t]*import[ \t]+)([^\n#]+)")
DYNAMIC_IMPORT = re.compile(
    r"([\"'])(agent|api|config|generation|indexing|infrastructure|ingestion|llm|models|profiles|retrieval|services|utils|verification)(?=\.|\1)"
)


def rewrite_python_imports(text: str) -> str:
    def replace_from(match: re.Match[str]) -> str:
        module = match.group(2)
        first = module.split(".", 1)[0]
        if first in MODULES:
            module = "omnirag." + module
        return match.group(1) + module + match.group(3)

    text = FROM_IMPORT.sub(replace_from, text)

    def replace_import_line(match: re.Match[str]) -> str:
        prefix, body = match.group(1), match.group(2)
        parts = []
        for item in body.split(","):
            found = re.match(
                r"^(\s*)([A-Za-z_][A-Za-z0-9_.]*)(.*)$", item
            )
            if found and found.group(2).split(".", 1)[0] in MODULES:
                module = "omnirag." + found.group(2)
                item = found.group(1) + module + found.group(3)
            parts.append(item)
        return prefix + ",".join(parts)

    text = IMPORT_LINE.sub(replace_import_line, text)
    text = DYNAMIC_IMPORT.sub(lambda m: m.group(1) + "omnirag." + m.group(2), text)
    return text


def update_text_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False

    updated = original
    if path.suffix == ".py" and "tests/fixtures/" not in path.as_posix():
        updated = rewrite_python_imports(updated)

    # Update documented commands and container entry points.
    updated = updated.replace("uvicorn api.main:app", "uvicorn omnirag.api.main:app")
    updated = updated.replace('"api.main:app"', '"omnirag.api.main:app"')
    updated = updated.replace("'api.main:app'", "'omnirag.api.main:app'")

    if updated != original:
        path.write_text(updated, encoding="utf-8", newline="")
        return True
    return False


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    moved = []
    for name in PACKAGE_ROOTS:
        source = ROOT / name
        target = SOURCE / name
        if source.exists():
            if target.exists():
                raise RuntimeError(f"Both source and target exist: {source} / {target}")
            shutil.move(str(source), str(target))
            moved.append(name)
        elif not target.exists():
            raise RuntimeError(f"Expected package missing: {source}")

    legacy_config = ROOT / "config.py"
    if legacy_config.exists():
        if legacy_config.stat().st_size:
            raise RuntimeError("Refusing to delete non-empty legacy config.py")
        legacy_config.unlink()

    init_file = SOURCE / "__init__.py"
    init_file.write_text(
        '"""OmniRAG application package."""\n\n__version__ = "0.6.0"\n',
        encoding="utf-8",
    )

    pyproject = ROOT / "pyproject.toml"
    config = pyproject.read_text(encoding="utf-8")
    old_block = re.compile(
        r"(?ms)^\[tool\.setuptools\.packages\.find\]\n.*?(?=^\[|\Z)"
    )
    new_block = (
        '[tool.setuptools.packages.find]\n'
        'where = ["backend/src"]\n'
        'include = ["omnirag*"]\n\n'
    )
    config, count = old_block.subn(new_block, config, count=1)
    if count != 1:
        raise RuntimeError("Could not safely update setuptools package discovery")
    if 'pythonpath = ["backend/src", "."]' not in config:
        config = config.replace(
            'addopts = "-ra"\n',
            'addopts = "-ra"\npythonpath = ["backend/src", "."]\n',
            1,
        )
    pyproject.write_text(config, encoding="utf-8", newline="")

    api_main = SOURCE / "api" / "main.py"
    text = api_main.read_text(encoding="utf-8")
    if "import os\n" not in text:
        text = text.replace("import logging\n", "import logging\nimport os\n", 1)
    old_frontend = (
        'frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"'
    )
    new_frontend = (
        'frontend_dist = Path(os.environ.get(\n'
        '    "OMNIRAG_FRONTEND_DIST",\n'
        '    str(Path(__file__).resolve().parents[4] / "frontend" / "dist"),\n'
        '))'
    )
    if old_frontend not in text:
        raise RuntimeError("Could not locate frontend asset path in api/main.py")
    text = text.replace(old_frontend, new_frontend, 1)
    api_main.write_text(text, encoding="utf-8", newline="")

    dockerfile = ROOT / "Dockerfile"
    docker = dockerfile.read_text(encoding="utf-8")
    docker = docker.replace(
        "ENV OMNIRAG_ENVIRONMENT=production",
        "ENV OMNIRAG_ENVIRONMENT=production OMNIRAG_FRONTEND_DIST=/app/frontend/dist",
    )
    docker = docker.replace("api.main:app", "omnirag.api.main:app")
    dockerfile.write_text(docker, encoding="utf-8", newline="")

    changed = 0
    skipped = {".git", ".venv", "node_modules", "__pycache__"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in skipped for part in path.parts):
            continue
        if path == pyproject or path == api_main or path == dockerfile:
            continue
        if path.suffix in TEXT_SUFFIXES or path.suffix == ".py":
            changed += int(update_text_file(path))

    # Verify every moved package is now inside the namespace and all command references changed.
    for name in PACKAGE_ROOTS:
        if not (SOURCE / name / "__init__.py").exists():
            raise RuntimeError(f"Package did not land correctly: omnirag.{name}")

    remaining = []
    for path in ROOT.rglob("*.py"):
        if any(part in skipped for part in path.parts) or "tests/fixtures/" in path.as_posix():
            continue
        content = path.read_text(encoding="utf-8")
        if re.search(
            r"(?m)^\s*(?:from|import)\s+(?:agent|api|config|generation|indexing|infrastructure|ingestion|llm|models|profiles|retrieval|services|utils|verification)(?:\.|\s)",
            content,
        ):
            remaining.append(path.relative_to(ROOT).as_posix())
    if remaining:
        raise RuntimeError("Unmigrated absolute imports remain: " + ", ".join(remaining[:20]))

    print("Moved packages:", ", ".join(moved))
    print("Updated additional text files:", changed)
    print("Removed empty legacy config.py")
    print("Repository layout migration applied.")


if __name__ == "__main__":
    main()

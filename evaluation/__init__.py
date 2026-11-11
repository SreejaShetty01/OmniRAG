"""Backward-compatible namespace for the relocated eval package."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "eval")]

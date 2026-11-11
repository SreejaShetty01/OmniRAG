"""Compatibility alias for the relocated :mod:`eval` package."""
import sys
from importlib import import_module

_eval_package = import_module("eval")
sys.modules[__name__] = _eval_package

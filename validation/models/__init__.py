"""The campaign's models: every module in this package that defines NAME, runs(scale) and
analyse(scale, data, figures) (see validation/core.py), keyed by NAME."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def _discover() -> dict[str, ModuleType]:
    found: dict[str, ModuleType] = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        name = getattr(mod, "NAME", None)
        if name is None or not callable(getattr(mod, "runs", None)) or not callable(getattr(mod, "analyse", None)):
            raise ImportError(f"{mod.__name__}: a model module defines NAME, runs(scale) and "
                              "analyse(scale, data, figures)")
        if name in found:
            raise ImportError(f"{mod.__name__}: model name '{name}' is already used by "
                              f"{found[name].__name__}")
        found[name] = mod
    return found


MODELS: dict[str, ModuleType] = _discover()

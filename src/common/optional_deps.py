"""Utilities for optional third-party dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any


class MissingDependencyError(RuntimeError):
    """Raised when a required optional dependency is unavailable."""


def optional_import(module_path: str, package_name: str | None = None) -> Any:
    """Import a module lazily and raise a clear error on failure."""
    try:
        return import_module(module_path)
    except Exception as exc:  # pragma: no cover
        pkg = package_name or module_path
        raise MissingDependencyError(
            f"Optional dependency '{pkg}' is required for this operation."
        ) from exc

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# -------------------------
# Project root resolution
# -------------------------


@lru_cache(maxsize=1)
def project_root() -> Path:
    """
    Resolve project root directory.

    Resolution order:
    1. Environment variable `LAB_PRJ_ROOT`
    2. Walk upward from CWD looking for markers
    3. Fallback to CWD
    """
    env_root = os.getenv("LAB_PRJ_ROOT")
    if env_root:
        return Path(env_root).resolve()

    markers: Iterable[str] = ("pyproject.toml", ".git")
    cur = Path.cwd().resolve()

    for _ in range(12):
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    return Path.cwd().resolve()


# -------------------------
# Internal helper
# -------------------------


def _resolve_path(*parts: str, path: Path | str | None = None) -> Path:
    base = project_root().joinpath(*parts)
    return base if path is None else base / Path(path)


# -------------------------
# Data directories
# -------------------------


def raw_data_dir(path: Path | str | None = None) -> Path:
    return _resolve_path("data", "raw", path=path)


def ext_data_dir(path: Path | str | None = None) -> Path:
    return _resolve_path("data", "ext", path=path)


def cache_data_dir(path: Path | str | None = None) -> Path:
    return _resolve_path("data", "cache", path=path)


# -------------------------
# Output directories
# -------------------------


def out_dir(path: Path | str | None = None) -> Path:
    return _resolve_path("outputs", path=path)


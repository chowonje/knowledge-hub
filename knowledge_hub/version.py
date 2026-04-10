from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as metadata_version
from pathlib import Path
import re

DISTRIBUTION_NAME = "knowledge-hub-cli"

_PROJECT_SECTION_RE = re.compile(
    r"^\[project\]\n(?P<body>.*?)(?=^\[|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
_VERSION_RE = re.compile(r'^version\s*=\s*["\'](?P<version>[^"\']+)["\']$', flags=re.MULTILINE)


def _read_pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    section_match = _PROJECT_SECTION_RE.search(text)
    if section_match is None:
        raise RuntimeError(f"Missing [project] section in {pyproject}")

    version_match = _VERSION_RE.search(section_match.group("body"))
    if version_match is None:
        raise RuntimeError(f"Missing project.version in {pyproject}")
    return version_match.group("version")


@lru_cache(maxsize=1)
def get_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject.exists():
        return _read_pyproject_version()
    try:
        return str(metadata_version(DISTRIBUTION_NAME))
    except PackageNotFoundError:
        raise RuntimeError(f"Unable to resolve version for {DISTRIBUTION_NAME}") from None

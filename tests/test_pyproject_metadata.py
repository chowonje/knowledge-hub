from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from knowledge_hub import __version__
from knowledge_hub.version import get_version


def _parse_toml_literal_section(text: str, section_name: str) -> dict[str, object]:
    section_match = re.search(
        rf"^\[{re.escape(section_name)}\]\n(?P<body>.*?)(?=^\[|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert section_match is not None
    body = section_match.group("body")

    result: dict[str, object] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    def _maybe_flush() -> bool:
        nonlocal current_name, current_lines
        if current_name is None:
            return False
        payload = "\n".join(current_lines).strip()
        try:
            result[current_name] = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            return False
        current_name = None
        current_lines = []
        return True

    for raw_line in body.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if current_name is None:
            match = re.match(r"^([A-Za-z0-9_.-]+)\s*=\s*(.+)$", raw_line)
            assert match is not None
            current_name = match.group(1)
            current_lines = [match.group(2)]
            _maybe_flush()
            continue
        current_lines.append(raw_line)
        _maybe_flush()

    assert current_name is None
    return result


def _parse_optional_dependency_lists(text: str) -> dict[str, list[str]]:
    parsed = _parse_toml_literal_section(text, "project.optional-dependencies")
    return {name: value for name, value in parsed.items() if isinstance(value, list)}


def _parse_console_scripts(text: str) -> dict[str, str]:
    parsed = _parse_toml_literal_section(text, "project.scripts")
    return {name: value for name, value in parsed.items() if isinstance(value, str)}


def _module_from_entrypoint(entrypoint: str) -> str:
    return entrypoint.split(":", 1)[0]


def _project_version(text: str) -> str:
    section_match = re.search(
        r"^\[project\]\n(?P<body>.*?)(?=^\[|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert section_match is not None
    version_match = re.search(
        r'^version\s*=\s*["\'](?P<version>[^"\']+)["\']$',
        section_match.group("body"),
        flags=re.MULTILINE,
    )
    assert version_match is not None
    return version_match.group("version")


def test_all_extra_contains_all_non_meta_optional_dependencies():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    extras = _parse_optional_dependency_lists(text)

    assert "all" in extras
    expected = set()
    for name, deps in extras.items():
        if name in {"all", "dev"}:
            continue
        expected.update(deps)

    assert expected
    assert set(extras["all"]) == expected


def test_console_scripts_use_canonical_entrypoints():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    scripts = _parse_console_scripts(text)

    assert scripts["khub"] == "knowledge_hub.interfaces.cli.main:cli"
    assert scripts["khub-mcp"] == "knowledge_hub.interfaces.mcp.server:main"


def test_root_package_helpers_use_canonical_module_entrypoints():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    package_json = Path(__file__).resolve().parents[1] / "package.json"
    pyproject_text = pyproject.read_text(encoding="utf-8")
    package = json.loads(package_json.read_text(encoding="utf-8"))
    scripts = _parse_console_scripts(pyproject_text)

    assert package["scripts"]["cli"] == f'python -m {_module_from_entrypoint(scripts["khub"])}'
    assert package["scripts"]["mcp"] == f'python -m {_module_from_entrypoint(scripts["khub-mcp"])}'


def test_runtime_version_matches_project_metadata():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject_text = pyproject.read_text(encoding="utf-8")
    project_version = _project_version(pyproject_text)

    assert get_version() == project_version
    assert __version__ == project_version

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.interfaces.cli.commands import mcp_cmd, status_cmd
from knowledge_hub.interfaces.cli.main import cli


def _project_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise AssertionError("project.version not found")


def test_cli_version_surfaces_match_project_metadata():
    version = _project_version()
    runner = CliRunner()

    cli_result = runner.invoke(cli, ["--version"])
    mcp_result = runner.invoke(mcp_cmd.mcp_cmd, ["--version"])

    assert cli_result.exit_code == 0
    assert version in cli_result.output
    assert mcp_result.exit_code == 0
    assert version in mcp_result.output
    assert status_cmd._get_version() == version

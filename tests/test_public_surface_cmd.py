from __future__ import annotations

from click.testing import CliRunner

from knowledge_hub.interfaces.cli.main import cli


def _command_lines(output: str) -> set[str]:
    commands: set[str] = set()
    in_commands = False
    for line in output.splitlines():
        if line.strip() == "Commands:":
            in_commands = True
            continue
        if not in_commands:
            continue
        if not line.startswith("  "):
            if line.strip():
                break
            continue
        stripped = line.strip()
        if stripped and not stripped.startswith("-"):
            commands.add(stripped.split()[0])
    return commands


def test_default_help_is_public_surface_contract():
    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    commands = _command_lines(result.output)
    assert {
        "discover",
        "index",
        "search",
        "ask",
        "inspect",
        "compare",
        "trace",
        "papers",
        "doctor",
        "setup",
        "status",
        "labs",
        "help",
    }.issubset(commands)
    assert "paper" not in commands
    assert "crawl" not in commands
    assert "agent" not in commands
    assert "mcp" not in commands
    assert "provider" not in commands


def test_paper_compat_alias_is_hidden_but_callable():
    runner = CliRunner()

    root = runner.invoke(cli, ["--help"])
    alias_help = runner.invoke(cli, ["paper", "--help"])
    hidden_subcommand = runner.invoke(cli, ["paper", "review-card-plan", "--help"])

    assert root.exit_code == 0
    assert "paper" not in _command_lines(root.output)
    assert alias_help.exit_code == 0
    assert "Usage:" in alias_help.output
    assert hidden_subcommand.exit_code == 0
    assert "review-card-plan" in hidden_subcommand.output


def test_papers_help_keeps_operator_remediation_hidden():
    runner = CliRunner()

    result = runner.invoke(cli, ["papers", "--help"])

    assert result.exit_code == 0
    commands = _command_lines(result.output)
    assert {"add", "import-csv", "list", "summary", "evidence", "memory", "related"}.issubset(commands)
    assert "repair-source" not in commands
    assert "repair-source-queue" not in commands
    assert "review-card" not in commands
    assert "review-card-apply" not in commands
    assert "review-card-apply-batch" not in commands
    assert "canon-quality-audit" not in commands


def test_help_advanced_records_public_discover_policy_and_compat_alias():
    runner = CliRunner()

    result = runner.invoke(cli, ["help", "advanced"])

    assert result.exit_code == 0
    assert "discover is part of the public default source lifecycle" in result.output
    assert "paper -> papers" in result.output
    assert "hidden/operator commands remain directly invokable" in result.output

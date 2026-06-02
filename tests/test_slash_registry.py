from __future__ import annotations

from knowledge_hub.interfaces.cli.commands.slash_registry import (
    help_rows,
    parse_slash_input,
    slash_command_for,
    startup_rows,
)


def test_slash_registry_resolves_aliases():
    assert slash_command_for("model").name == "models"
    assert slash_command_for("/commands").name == "help"
    assert slash_command_for("reset").name == "clear"
    assert slash_command_for("quit").name == "exit"


def test_slash_registry_parses_command_args():
    parsed = parse_slash_input("/models use openai/gpt-5.4")

    assert parsed is not None
    command, args = parsed
    assert command.name == "models"
    assert args == "use openai/gpt-5.4"


def test_slash_registry_feeds_help_and_startup_rows():
    help_labels = [label for label, _summary in help_rows()]
    startup_labels = [label for label, _summary in startup_rows()]

    assert "/auth [login PROVIDER ENV_VAR]" in help_labels
    assert "/models [select|use PROVIDER/MODEL]" in help_labels
    assert "/clear" in help_labels
    assert help_labels == startup_labels

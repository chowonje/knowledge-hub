"""Central slash-command registry for the interactive assistant shell."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommandDef:
    name: str
    usage: str
    summary: str
    category: str
    aliases: tuple[str, ...] = ()
    visible: bool = True

    @property
    def label(self) -> str:
        return f"/{self.usage}"


SLASH_COMMANDS: tuple[SlashCommandDef, ...] = (
    SlashCommandDef(
        name="help",
        usage="help",
        summary="show this command registry",
        category="session",
        aliases=("commands",),
    ),
    SlashCommandDef(
        name="auth",
        usage="auth [login PROVIDER ENV_VAR]",
        summary="show auth status or store an API-key env reference",
        category="config",
    ),
    SlashCommandDef(
        name="models",
        usage="models [select|use PROVIDER/MODEL]",
        summary="show, choose, or change the active chat model",
        category="config",
        aliases=("model",),
    ),
    SlashCommandDef(
        name="chat",
        usage="chat [text]",
        summary="plain LLM chat; bare text also chats",
        category="chat",
    ),
    SlashCommandDef(
        name="paper",
        usage="paper QUESTION",
        summary="answer from saved-paper evidence",
        category="evidence",
    ),
    SlashCommandDef(
        name="ask",
        usage="ask QUESTION",
        summary="compat alias for /paper",
        category="evidence",
    ),
    SlashCommandDef(
        name="clear",
        usage="clear",
        summary="clear in-memory chat history",
        category="session",
        aliases=("reset",),
    ),
    SlashCommandDef(
        name="exit",
        usage="exit",
        summary="quit the shell",
        category="session",
        aliases=("quit",),
    ),
)

_REGISTRY = {command.name: command for command in SLASH_COMMANDS}
_ALIASES: dict[str, str] = {}
for _command in SLASH_COMMANDS:
    _ALIASES[_command.name] = _command.name
    for _alias in _command.aliases:
        _ALIASES[_alias] = _command.name


def visible_slash_commands() -> tuple[SlashCommandDef, ...]:
    return tuple(command for command in SLASH_COMMANDS if command.visible)


def slash_command_for(name: str) -> SlashCommandDef | None:
    key = str(name or "").strip().lower().removeprefix("/")
    canonical = _ALIASES.get(key)
    if canonical is None:
        return None
    return _REGISTRY.get(canonical)


def parse_slash_input(text: str) -> tuple[SlashCommandDef, str] | None:
    raw = str(text or "").strip()
    if not raw.startswith("/"):
        return None
    name, _, rest = raw[1:].partition(" ")
    command = slash_command_for(name)
    if command is None:
        return None
    return command, rest.strip()


def help_rows() -> list[tuple[str, str]]:
    return [(command.label, command.summary) for command in visible_slash_commands()]


def startup_rows() -> list[tuple[str, str]]:
    return [(command.label, command.summary) for command in visible_slash_commands()]

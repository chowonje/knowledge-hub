from __future__ import annotations

import importlib
import logging
import click
import types

from click.testing import CliRunner


def test_interfaces_cli_main_exports_canonical_entrypoint():
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")
    assert module.cli is not None
    assert callable(module.main)
    assert module.KhubContext is not None


def test_legacy_cli_main_is_a_shim_to_interfaces_module():
    legacy = importlib.import_module("knowledge_hub.cli.main")
    canonical = importlib.import_module("knowledge_hub.interfaces.cli.main")
    assert legacy.cli is canonical.cli
    assert legacy.main is canonical.main
    assert legacy.KhubContext is canonical.KhubContext


def test_cli_help_hides_labs_commands_from_top_level():
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")
    runner = CliRunner()

    result = runner.invoke(module.cli, ["--help"])

    assert result.exit_code == 0
    assert "add" in result.output
    assert "search" in result.output
    assert "ask" in result.output
    assert "agent" in result.output
    assert "provider" in result.output
    assert "doctor" in result.output
    assert "labs" in result.output
    assert "crawl" not in result.output
    assert "discover" not in result.output
    assert "dinger" not in result.output
    assert "eval" not in result.output
    assert "learn" not in result.output
    assert "belief" not in result.output
    assert "decision" not in result.output
    assert "outcome" not in result.output
    assert "ontology" not in result.output
    assert "graph" not in result.output
    assert "claims" not in result.output
    assert "feature" not in result.output
    assert "rag-report" not in result.output
    assert "ops-report-run" not in result.output


def test_cli_labs_help_exposes_demoted_groups():
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")
    runner = CliRunner()

    result = runner.invoke(module.cli, ["labs", "--help"])

    assert result.exit_code == 0
    assert "learn" in result.output
    assert "belief" in result.output
    assert "decision" in result.output
    assert "outcome" in result.output
    assert "ontology" in result.output
    assert "graph" in result.output
    assert "claims" in result.output
    assert "section-cards" in result.output
    assert "feature" in result.output
    assert "crawl" in result.output
    assert "ops" in result.output
    assert "transform" in result.output
    assert "ask-graph" in result.output
    assert "memory" in result.output
    assert "paper" in result.output
    assert "eval" in result.output
    assert "foundry" in result.output


def test_cli_labs_ops_help_exposes_operator_commands():
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")
    runner = CliRunner()

    result = runner.invoke(module.cli, ["labs", "ops", "--help"])

    assert result.exit_code == 0
    assert "rag-report" in result.output
    assert "report-run" in result.output
    assert "action-list" in result.output
    assert "action-ack" in result.output
    assert "action-resolve" in result.output
    assert "action-execute" in result.output
    assert "action-receipts" in result.output


def test_cli_labs_section_cards_help_exposes_build_show_preview():
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")
    runner = CliRunner()

    result = runner.invoke(module.cli, ["labs", "section-cards", "--help"])

    assert result.exit_code == 0
    assert "build" in result.output
    assert "show" in result.output
    assert "preview" in result.output


def test_cli_top_level_uses_lazy_command_loading(monkeypatch):
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")
    fake_module = types.SimpleNamespace()

    @click.command("dummy")
    def _dummy():
        return None

    fake_module.dummy_cmd = _dummy
    imported: list[str] = []
    real_import_module = module.import_module

    def _tracking_import(name: str):
        imported.append(name)
        if name == "tests.fake_lazy_module":
            return fake_module
        return real_import_module(name)

    monkeypatch.setattr(module, "import_module", _tracking_import)
    group = module._LazyCommandGroup(name="root")
    group.add_lazy_command("tests.fake_lazy_module", "dummy_cmd", "dummy")

    names = group.list_commands(click.Context(group))
    assert "dummy" in names
    assert imported == []

    command = group.get_command(click.Context(group), "dummy")
    assert command is not None
    assert "tests.fake_lazy_module" in imported


def test_cli_help_does_not_import_unloaded_lazy_commands(monkeypatch):
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")
    imported: list[str] = []

    def _tracking_import(name: str):
        imported.append(name)
        raise AssertionError(f"help should not import lazy module {name}")

    monkeypatch.setattr(module, "import_module", _tracking_import)
    group = module._LazyCommandGroup(name="root")
    group.add_lazy_command("tests.fake_lazy_module", "dummy_cmd", "dummy", short_help="fake command")

    ctx = click.Context(group)
    formatter = ctx.make_formatter()
    group.format_commands(ctx, formatter)

    assert imported == []
    assert "dummy" in formatter.getvalue()


def test_cli_formatter_summarizes_provider_outbound_warning_by_default():
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")
    logger = logging.getLogger("khub.providers.openai")
    record = logger.makeRecord(
        logger.name,
        logging.WARNING,
        __file__,
        1,
        "Provider outbound warning trace_id=%s warnings=%s",
        ("policy_123", ["P1 structured facts detected", "provider=openai", "model=gpt-5.4"]),
        None,
    )

    formatter = module._KhubCliFormatter("%(message)s", summarize_provider_warnings=True)

    rendered = formatter.format(record)

    assert rendered == (
        "outbound policy warning: classification=P1 provider=openai model=gpt-5.4 trace_id=policy_123"
    )


def test_cli_formatter_keeps_raw_provider_outbound_warning_in_verbose_mode():
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")
    logger = logging.getLogger("khub.providers.openai")
    record = logger.makeRecord(
        logger.name,
        logging.WARNING,
        __file__,
        1,
        "Provider outbound warning trace_id=%s warnings=%s",
        ("policy_123", ["P1 structured facts detected", "provider=openai", "model=gpt-5.4"]),
        None,
    )

    formatter = module._KhubCliFormatter("%(message)s", summarize_provider_warnings=False)

    rendered = formatter.format(record)

    assert "Provider outbound warning trace_id=policy_123" in rendered
    assert "P1 structured facts detected" in rendered


def test_setup_logging_uses_summary_formatter_when_not_verbose():
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")

    module._setup_logging(verbose=False)

    handler = logging.getLogger().handlers[-1]
    assert isinstance(handler.formatter, module._KhubCliFormatter)
    assert handler.formatter._summarize_provider_warnings is True


def test_setup_logging_keeps_raw_formatter_when_verbose():
    module = importlib.import_module("knowledge_hub.interfaces.cli.main")

    module._setup_logging(verbose=True)

    handler = logging.getLogger().handlers[-1]
    assert isinstance(handler.formatter, module._KhubCliFormatter)
    assert handler.formatter._summarize_provider_warnings is False

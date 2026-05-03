"""
khub CLI - Knowledge Hub canonical entrypoint.

Usage:
    khub init                          # 초기 설정
    khub discover "topic" [OPTIONS]    # 논문 검색 → 다운로드 → 요약 → 연결
    khub paper summary|evidence|memory|related
    khub config get|set|list
    khub provider recommend|setup|add|use
    khub search "query"
    khub ask "question"
    khub health                         # 설정/의존성 진단
    khub doctor                         # 일반 사용자용 설치/실패 경로 요약
    khub setup [--quick]               # 초보자용 빠른 시작
"""

from __future__ import annotations

import ast
from copy import copy
from importlib import import_module
import logging
import os
from pathlib import Path

import click
from rich.console import Console

from knowledge_hub.version import get_version

console = Console()
log = logging.getLogger("khub")


def _parse_provider_warning_record(record: logging.LogRecord) -> tuple[str, list[str]] | None:
    msg = str(getattr(record, "msg", "") or "")
    rendered = record.getMessage()
    if "Provider outbound warning trace_id=" not in msg and "Provider outbound warning trace_id=" not in rendered:
        return None

    trace_id = ""
    warnings: list[str] = []
    args = getattr(record, "args", ())
    if isinstance(args, tuple) and len(args) >= 2:
        trace_id = str(args[0] or "")
        raw_warnings = args[1]
        if isinstance(raw_warnings, (list, tuple)):
            warnings = [str(item).strip() for item in raw_warnings if str(item).strip()]
    if warnings:
        return trace_id, warnings

    marker = " warnings="
    if marker not in rendered or "trace_id=" not in rendered:
        return None
    before, _, raw_warnings = rendered.partition(marker)
    trace_id = before.rsplit("trace_id=", 1)[-1].strip()
    try:
        parsed = ast.literal_eval(raw_warnings)
    except (ValueError, SyntaxError):
        parsed = raw_warnings
    if isinstance(parsed, (list, tuple)):
        warnings = [str(item).strip() for item in parsed if str(item).strip()]
    elif str(parsed).strip():
        warnings = [str(parsed).strip()]
    return trace_id, warnings


def _summarize_provider_warning_record(record: logging.LogRecord) -> str | None:
    parsed = _parse_provider_warning_record(record)
    if parsed is None:
        return None
    trace_id, warnings = parsed
    provider = next((item.split("=", 1)[1] for item in warnings if item.startswith("provider=")), "-")
    model = next((item.split("=", 1)[1] for item in warnings if item.startswith("model=")), "-")
    classification = next(
        (
            item.split()[0]
            for item in warnings
            if item and item[0] == "P" and any(char.isdigit() for char in item.split()[0])
        ),
        "warning",
    )
    return (
        "outbound policy warning: "
        f"classification={classification} provider={provider} model={model} trace_id={trace_id or '-'}"
    )


class _KhubCliFormatter(logging.Formatter):
    def __init__(self, *args, summarize_provider_warnings: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._summarize_provider_warnings = summarize_provider_warnings

    def format(self, record: logging.LogRecord) -> str:
        if self._summarize_provider_warnings:
            summary = _summarize_provider_warning_record(record)
            if summary:
                replacement = logging.makeLogRecord(dict(record.__dict__))
                replacement.msg = summary
                replacement.args = ()
                return super().format(replacement)
        return super().format(record)


def _auto_load_dotenv():
    """프로젝트 또는 cwd의 .env 파일에서 환경 변수 로드"""
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
            break


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler()
    handler.setFormatter(
        _KhubCliFormatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            summarize_provider_warnings=not verbose,
        )
    )
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,
    )


_auto_load_dotenv()


class KhubContext:
    """CLI 컨텍스트 - Config와 핵심 객체를 보관"""

    def __init__(self, config_path: str | None = None):
        from knowledge_hub.application.context import AppContextFactory

        self._factory = AppContextFactory(config_path)
        self._config_path = self._factory.config_path
        self._app_context = None

    @property
    def config(self):
        return self.factory.config

    @property
    def factory(self):
        return self._factory

    @property
    def app_context(self):
        if self._app_context is None:
            self._app_context = self._factory.app_context
        return self._app_context

    def app(self, *, require_search: bool = False, core_only: bool = False):
        if require_search or core_only:
            return self._factory.build(require_search=require_search, core_only=core_only)
        return self.app_context

    def sqlite_db(self, **kwargs):
        return self._factory.create_sqlite_db(**kwargs)

    def vector_db(self, *, repair_on_init: bool = True):
        return self._factory.create_vector_db(repair_on_init=repair_on_init)

    def searcher(self):
        return self._factory.get_searcher()

    def build_llm(self, provider: str, model: str | None = None):
        return self._factory.build_llm(provider, model)

    def build_embedder(self, provider: str, model: str | None = None):
        return self._factory.build_embedder(provider, model)

    def learning_service(self):
        return self._factory.create_learning_service()

    def web_ingest_service(self):
        return self._factory.create_web_ingest_service()


class _ErrorHandlingGroup(click.Group):
    """CLI 최상위 그룹에 공통 에러 핸들링 적용"""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except click.exceptions.Exit:
            raise
        except click.exceptions.Abort:
            raise
        except click.ClickException:
            raise
        except SystemExit:
            raise
        except Exception as e:
            from knowledge_hub.infrastructure.config import ConfigError

            if isinstance(e, ConfigError):
                console.print(f"[bold red]설정 오류:[/bold red] {e}")
                console.print("[dim]khub init 또는 환경변수를 확인하세요.[/dim]")
                ctx.exit(1)
            else:
                log.exception("예상치 못한 오류")
                console.print(f"[bold red]오류:[/bold red] {e}")
                console.print("[dim]--verbose 플래그로 디버그 정보를 확인할 수 있습니다.[/dim]")
                ctx.exit(1)


class _LazyCommandGroup(_ErrorHandlingGroup):
    """Click group that loads subcommands only when they are accessed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lazy_commands: dict[str, tuple[str, str, str, bool]] = {}

    def add_lazy_command(
        self,
        module_path: str,
        attr_name: str,
        command_name: str,
        short_help: str = "",
        *,
        hidden: bool = False,
    ) -> None:
        self._lazy_commands[str(command_name)] = (str(module_path), str(attr_name), str(short_help or ""), bool(hidden))

    def get_command(self, ctx, cmd_name):
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        spec = self._lazy_commands.get(str(cmd_name))
        if spec is None:
            return None
        module = import_module(spec[0])
        command = getattr(module, spec[1])
        if spec[3]:
            command = copy(command)
            command.hidden = True
        self.add_command(command, cmd_name)
        return command

    def list_commands(self, ctx):
        names = {
            name
            for name in super().list_commands(ctx)
            if not getattr(self.commands.get(name), "hidden", False)
        }
        names.update(
            name
            for name, spec in self._lazy_commands.items()
            if not spec[3]
        )
        return sorted(names)

    def format_commands(self, ctx, formatter):
        rows = []
        for subcommand in self.list_commands(ctx):
            command = self.commands.get(subcommand)
            if command is not None:
                if command.hidden:
                    continue
                rows.append((subcommand, command.get_short_help_str(formatter.width)))
                continue
            spec = self._lazy_commands.get(str(subcommand))
            if spec is None:
                continue
            rows.append((subcommand, spec[2]))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(cls=_LazyCommandGroup)
@click.option("--config", "-c", "config_path", default=None, help="설정 파일 경로 (기본: ~/.khub/config.yaml)")
@click.option("--verbose", "-v", is_flag=True, help="디버그 로그 출력")
@click.version_option(version=get_version(), prog_name="knowledge-hub")
@click.pass_context
def cli(ctx, config_path, verbose):
    """Knowledge Hub - AI 논문 검색, 번역, 요약, 지식 연결 파이프라인"""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["khub"] = KhubContext(config_path)


@cli.command()
@click.pass_context
def status(ctx):
    """시스템 상태 및 통계"""
    from knowledge_hub.interfaces.cli.commands.status_cmd import run_status

    run_status(ctx.obj["khub"])


@click.group("labs", cls=_LazyCommandGroup)
def labs_group():
    """실험적이거나 비핵심 subsystems"""


@click.group("ops", cls=_LazyCommandGroup)
def labs_ops_group():
    """고급 운영/리포트 commands"""
cli.add_command(labs_group, "labs")

cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.init_cmd", "init_cmd", "init")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.config_cmd", "config_group", "config")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.provider_cmd", "provider_group", "provider")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.discover_cmd", "discover", "discover")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.paper_cmd", "paper_group", "paper")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.paper_memory_cmd", "paper_memory_group", "paper-memory", hidden=True)
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.math_memory_cmd", "math_memory_group", "math-memory", hidden=True)
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.explore_cmd", "explore_group", "explore")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.search_cmd", "search", "search")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.search_cmd", "ask", "ask")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.index_cmd", "index_cmd", "index")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.health_cmd", "health_cmd", "health")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.doctor_cmd", "doctor_cmd", "doctor")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.vector_compare_cmd", "vector_compare_cmd", "vector-compare", hidden=True)
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.vector_cmd", "vector_restore_cmd", "vector-restore", hidden=True)
cli.add_lazy_command(
    "knowledge_hub.interfaces.cli.commands.vector_source_metadata_cmd",
    "vector_source_metadata_cmd",
    "vector-source-metadata",
    hidden=True,
)
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.setup_cmd", "setup_cmd", "setup")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.crawl_cmd", "crawl_group", "crawl")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.dinger_cmd", "dinger_group", "dinger", hidden=True)
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.mcp_cmd", "mcp_cmd", "mcp")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.vault_cmd", "vault_group", "vault")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.agent_cmd", "agent_group", "agent")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.os_cmd", "os_group", "os", hidden=True)
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.eval_cmd", "eval_compat_group", "eval", hidden=True)

labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.learn_cmd", "learn_group", "learn")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.foundry_cmd", "foundry_group", "foundry")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.ontology_cmd", "ontology_group", "ontology")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.claims_cmd", "claims_group", "claims")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.claim_cards_cmd", "claim_cards_group", "claim-cards")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.section_cards_cmd", "section_cards_group", "section-cards")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.feature_cmd", "feature_group", "feature")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.graph_cmd", "graph_group", "graph")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.belief_cmd", "belief_group", "belief")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.decision_cmd", "decision_group", "decision")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.outcome_cmd", "outcome_group", "outcome")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.crawl_cmd", "labs_crawl_group", "crawl")
labs_group.add_command(labs_ops_group, "ops")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.transform_cmd", "transform_group", "transform")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.ask_graph_cmd", "ask_graph_cmd", "ask-graph")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.memory_cmd", "memory_group", "memory")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.rag_labs_cmd", "rag_labs_group", "rag")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.paper_labs_cmd", "paper_labs_group", "paper")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.paper_summary_cmd", "paper_summary_group", "paper-summary")
labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.eval_cmd", "eval_group", "eval")

labs_ops_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.search_cmd", "rag_report", "rag-report")
labs_ops_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.ops_cmd", "ops_report_run", "report-run")
labs_ops_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.ops_cmd", "ops_action_list", "action-list")
labs_ops_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.ops_cmd", "ops_action_ack", "action-ack")
labs_ops_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.ops_cmd", "ops_action_resolve", "action-resolve")
labs_ops_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.ops_cmd", "ops_action_execute", "action-execute")
labs_ops_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.ops_cmd", "ops_action_receipts", "action-receipts")


def main():
    cli()


if __name__ == "__main__":
    main()

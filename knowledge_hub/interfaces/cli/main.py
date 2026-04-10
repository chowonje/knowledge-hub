"""
khub CLI - Knowledge Hub canonical entrypoint.

Usage:
    khub init                          # 초기 설정
    khub discover "topic" [OPTIONS]    # 논문 검색 → 다운로드 → 요약 → 연결
    khub dinger ingest --paper "topic" # 단순한 personal knowledge facade
    khub paper list|translate|summarize
    khub config get|set|list
    khub search "query"
    khub ask "question"
    khub health                         # 설정/의존성 진단
    khub doctor                         # 일반 사용자용 설치/실패 경로 요약
    khub setup [--quick]               # 초보자용 빠른 시작
"""

from __future__ import annotations

from importlib import import_module
import logging
import os
from pathlib import Path

import click
from rich.console import Console

from knowledge_hub.version import get_version

console = Console()
log = logging.getLogger("khub")


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
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
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

    def vector_db(self):
        return self._factory.create_vector_db()

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
        except SystemExit:
            raise
        except Exception as e:
            from knowledge_hub.infrastructure.config import ConfigError

            if isinstance(e, ConfigError):
                console.print(f"[bold red]설정 오류:[/bold red] {e}")
                console.print("[dim]khub init 또는 환경변수를 확인하세요.[/dim]")
                ctx.exit(1)
            elif isinstance(e, click.BadParameter):
                raise
            else:
                log.exception("예상치 못한 오류")
                console.print(f"[bold red]오류:[/bold red] {e}")
                console.print("[dim]--verbose 플래그로 디버그 정보를 확인할 수 있습니다.[/dim]")
                ctx.exit(1)


class _LazyCommandGroup(_ErrorHandlingGroup):
    """Click group that loads subcommands only when they are accessed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lazy_commands: dict[str, tuple[str, str]] = {}

    def add_lazy_command(self, module_path: str, attr_name: str, command_name: str) -> None:
        self._lazy_commands[str(command_name)] = (str(module_path), str(attr_name))

    def get_command(self, ctx, cmd_name):
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        spec = self._lazy_commands.get(str(cmd_name))
        if spec is None:
            return None
        module = import_module(spec[0])
        command = getattr(module, spec[1])
        self.add_command(command, cmd_name)
        return command

    def list_commands(self, ctx):
        names = set(super().list_commands(ctx))
        names.update(self._lazy_commands.keys())
        return sorted(names)


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
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.discover_cmd", "discover", "discover")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.paper_cmd", "paper_group", "paper")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.paper_memory_cmd", "paper_memory_group", "paper-memory")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.explore_cmd", "explore_group", "explore")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.notebook_cmd", "notebook_group", "notebook")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.search_cmd", "search", "search")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.search_cmd", "ask", "ask")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.index_cmd", "index_cmd", "index")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.health_cmd", "health_cmd", "health")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.doctor_cmd", "doctor_cmd", "doctor")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.setup_cmd", "setup_cmd", "setup")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.crawl_cmd", "crawl_group", "crawl")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.dinger_cmd", "dinger_group", "dinger")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.mcp_cmd", "mcp_cmd", "mcp")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.vault_cmd", "vault_group", "vault")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.agent_cmd", "agent_group", "agent")
cli.add_lazy_command("knowledge_hub.interfaces.cli.commands.os_cmd", "os_group", "os")

labs_group.add_lazy_command("knowledge_hub.interfaces.cli.commands.learn_cmd", "learn_group", "learn")
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

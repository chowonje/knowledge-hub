"""
khub CLI - Knowledge Hub 메인 진입점

Usage:
    khub init                          # 초기 설정
    khub discover "topic" [OPTIONS]    # 논문 검색 → 다운로드 → 요약 → 연결
    khub paper list|translate|summarize
    khub config get|set|list
    khub search "query"
    khub ask "question"
    khub status
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console

from knowledge_hub import __version__

console = Console()
log = logging.getLogger("khub")


def _auto_load_dotenv():
    """프로젝트 또는 cwd의 .env 파일에서 환경 변수 로드"""
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]:
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
        self._config = None
        self._config_path = config_path

    @property
    def config(self):
        if self._config is None:
            from knowledge_hub.core.config import Config
            self._config = Config(self._config_path)
        return self._config


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
            from knowledge_hub.core.config import ConfigError
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


@click.group(cls=_ErrorHandlingGroup)
@click.option("--config", "-c", "config_path", default=None, help="설정 파일 경로 (기본: ~/.khub/config.yaml)")
@click.option("--verbose", "-v", is_flag=True, help="디버그 로그 출력")
@click.version_option(version=__version__, prog_name="knowledge-hub")
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
    from knowledge_hub.cli.status_cmd import run_status
    run_status(ctx.obj["khub"])


# --- Sub-commands are registered via imports ---
from knowledge_hub.cli.init_cmd import init_cmd
from knowledge_hub.cli.config_cmd import config_group
from knowledge_hub.cli.discover_cmd import discover
from knowledge_hub.cli.paper_cmd import paper_group
from knowledge_hub.cli.search_cmd import search, ask
from knowledge_hub.cli.index_cmd import index_cmd
from knowledge_hub.cli.notebook_cmd import notebook_group
from knowledge_hub.cli.graph_cmd import graph_group as kg_graph_group

cli.add_command(init_cmd, "init")
cli.add_command(config_group, "config")
cli.add_command(discover, "discover")
cli.add_command(paper_group, "paper")
cli.add_command(notebook_group, "notebook")
cli.add_command(kg_graph_group, "graph")
cli.add_command(search, "search")
cli.add_command(ask, "ask")
cli.add_command(index_cmd, "index")


def main():
    cli()


if __name__ == "__main__":
    main()

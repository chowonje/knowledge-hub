"""Labs-only structured paper summary CLI surface."""

from __future__ import annotations

import click
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown

from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import _resolve_summary_build_options
from knowledge_hub.papers.opendataloader_adapter import ODL_READING_ORDER_CHOICES, ODL_TABLE_METHOD_CHOICES
from knowledge_hub.papers.structured_summary import StructuredPaperSummaryService

console = Console()


def _validate_cli_payload(config, payload: dict, schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


def _service(khub) -> StructuredPaperSummaryService:
    return StructuredPaperSummaryService(khub.sqlite_db(), khub.config)


def _odl_cli_overrides(*, reading_order: str | None, use_struct_tree: bool | None, table_method: str | None) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if reading_order is not None:
        overrides["reading_order"] = str(reading_order).strip().lower()
    if use_struct_tree is not None:
        overrides["use_struct_tree"] = bool(use_struct_tree)
    if table_method is not None:
        overrides["table_method"] = str(table_method).strip().lower()
    return overrides


@click.group("paper-summary")
def paper_summary_group():
    """structured paper-summary build/show"""


@paper_summary_group.command("build")
@click.option("--paper-id", required=True, help="paper id / arXiv id")
@click.option(
    "--paper-parser",
    default="auto",
    type=click.Choice(["auto", "raw", "pymupdf", "mineru", "opendataloader"]),
    show_default=True,
    help="paper source parsing mode for structured summary",
)
@click.option("--refresh-parse", is_flag=True, default=False, help="force re-parse parser artifacts for paper builds")
@click.option("--quick", is_flag=True, default=False, help="use a smaller context bundle")
@click.option("--provider", default=None, help="요약 프로바이더 override (기본: config)")
@click.option("--model", default=None, help="요약 모델 override (기본: config)")
@click.option(
    "--allow-external/--no-allow-external",
    default=None,
    help="외부 API 사용 허용 여부. 기본값은 설정된 요약 provider를 따릅니다.",
)
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="LLM 라우팅 모드 (provider override가 없을 때만 사용)",
)
@click.option(
    "--odl-reading-order",
    type=click.Choice(list(ODL_READING_ORDER_CHOICES)),
    default=None,
    help="OpenDataLoader reading-order override (opendataloader only)",
)
@click.option(
    "--odl-use-struct-tree/--no-odl-use-struct-tree",
    default=None,
    help="OpenDataLoader struct-tree override (opendataloader only)",
)
@click.option(
    "--odl-table-method",
    type=click.Choice(list(ODL_TABLE_METHOD_CHOICES)),
    default=None,
    help="OpenDataLoader table-method override (opendataloader only)",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def build_paper_summary(
    ctx,
    paper_id,
    paper_parser,
    refresh_parse,
    quick,
    provider,
    model,
    allow_external,
    llm_mode,
    odl_reading_order,
    odl_use_struct_tree,
    odl_table_method,
    as_json,
):
    """논문 구조화 요약 빌드"""
    khub = ctx.obj["khub"]
    summary_options = _resolve_summary_build_options(
        khub.config,
        provider=provider,
        model=model,
        allow_external=allow_external,
        llm_mode=llm_mode,
    )
    payload = _service(khub).build(
        paper_id=str(paper_id).strip(),
        paper_parser=str(paper_parser or "raw").strip().lower(),
        refresh_parse=bool(refresh_parse),
        quick=bool(quick),
        allow_external=bool(summary_options["allow_external"]),
        llm_mode=str(llm_mode or "auto"),
        provider_override=summary_options["provider_override"],
        model_override=summary_options["model_override"],
        opendataloader_options=_odl_cli_overrides(
            reading_order=odl_reading_order,
            use_struct_tree=odl_use_struct_tree,
            table_method=odl_table_method,
        ),
    )
    schema_id = str(payload.get("schema") or "knowledge-hub.paper-summary.build.result.v1")
    _validate_cli_payload(khub.config, payload, schema_id)
    if as_json:
        console.print_json(data=payload)
        return
    if payload.get("status") == "blocked":
        console.print(f"[yellow]paper-summary blocked[/yellow] {paper_id}")
        for warning in list(payload.get("warnings") or [])[:10]:
            console.print(f"[yellow]- {warning}[/yellow]")
        return
    console.print(f"[bold]paper-summary build[/bold] paper={payload.get('paperId')} status={payload.get('status')}")
    summary_md_path = Path(str((payload.get("artifactPaths") or {}).get("summaryMdPath") or "")).expanduser()
    if summary_md_path.exists():
        console.print(Markdown(summary_md_path.read_text(encoding="utf-8")))


@paper_summary_group.command("show")
@click.option("--paper-id", required=True, help="paper id / arXiv id")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def show_paper_summary(ctx, paper_id, as_json):
    """논문 구조화 요약 조회"""
    khub = ctx.obj["khub"]
    payload = _service(khub).show(paper_id=str(paper_id).strip())
    schema_id = str(payload.get("schema") or "knowledge-hub.paper-summary.card.result.v1")
    _validate_cli_payload(khub.config, payload, schema_id)
    if as_json:
        console.print_json(data=payload)
        return
    if payload.get("status") == "failed":
        raise click.ClickException("; ".join(list(payload.get("warnings") or [])) or f"paper summary not found: {paper_id}")
    summary = dict(payload.get("summary") or {})
    lines = [
        f"# {payload.get('paperTitle') or payload.get('paperId')}",
        "",
        "## 한줄 요약",
        "",
        str(summary.get("oneLine") or ""),
        "",
        "## 문제",
        "",
        str(summary.get("problem") or ""),
        "",
        "## 핵심 아이디어",
        "",
        str(summary.get("coreIdea") or ""),
        "",
        "## 방법",
        "",
    ]
    for item in list(summary.get("methodSteps") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## 주요 결과", ""])
    for item in list(summary.get("keyResults") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## 한계", ""])
    for item in list(summary.get("limitations") or []):
        lines.append(f"- {item}")
    console.print(Markdown("\n".join(lines)))


__all__ = ["paper_summary_group"]

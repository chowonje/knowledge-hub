from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from knowledge_hub.vault.math_memory import MathConceptMemoryService

console = Console()


def _service(khub) -> MathConceptMemoryService:
    return MathConceptMemoryService(khub.sqlite_db(), config=khub.config)


def _load_concepts(concepts: tuple[str, ...], concept_file: Path | None) -> list[str]:
    rows = [str(item).strip() for item in concepts if str(item).strip()]
    if concept_file is None:
        return rows
    lines = [line.strip() for line in concept_file.read_text(encoding="utf-8").splitlines()]
    rows.extend(line for line in lines if line and not line.startswith("#"))
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in rows:
        lowered = raw.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(raw)
    return deduped


@click.group("math-memory")
def math_memory_group():
    """math concept memory card build/show/search"""


@math_memory_group.command("build")
@click.option("--concept", "concepts", multiple=True, help="대상 math concept 이름 (여러 번 사용 가능)")
@click.option(
    "--concept-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="newline-delimited concept 목록 파일",
)
@click.option("--all-math", is_flag=True, default=False, help="Math 하위 개념 노트 전체를 대상으로 빌드")
@click.option("--allow-external/--no-allow-external", default=False, show_default=True, help="document-memory 생성 시 외부 API 사용 허용")
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="document-memory 추출용 LLM 라우팅 모드",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def build_math_memory(ctx, concepts, concept_file, all_math, allow_external, llm_mode, as_json):
    """수학 개념 노트를 document-memory + vault-card-v2로 빌드"""
    khub = ctx.obj["khub"]
    requested = _load_concepts(concepts, concept_file)
    payload = _service(khub).build(
        concepts=requested,
        all_math=bool(all_math),
        allow_external=bool(allow_external),
        llm_mode=str(llm_mode or "auto"),
    )
    if as_json:
        console.print_json(data=payload)
        return
    route = dict(payload.get("route") or {})
    console.print(
        f"[bold]math-memory build[/bold] count={payload.get('count')} "
        f"status={payload.get('status')} route={route.get('route') or 'fallback-only'} "
        f"provider={route.get('provider') or '-'} model={route.get('model') or '-'}"
    )
    for item in list(payload.get("items") or [])[:12]:
        dm = dict(item.get("documentMemory") or {})
        console.print(
            f"- {item.get('concept')} card={item.get('cardId')} "
            f"quality={item.get('qualityFlag')} mode={dm.get('mode')} "
            f"applied={dm.get('applied')} fallback={dm.get('fallbackUsed')}"
        )
    missing = list(payload.get("missingConcepts") or [])
    if missing:
        console.print(f"[yellow]missing:[/yellow] {', '.join(missing)}")


@math_memory_group.command("show")
@click.option("--concept", required=True, help="조회할 math concept 이름")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def show_math_memory(ctx, concept, as_json):
    """저장된 math memory card를 조회"""
    khub = ctx.obj["khub"]
    item = _service(khub).show(concept=str(concept).strip())
    if as_json:
        console.print_json(data={"status": "ok" if item else "failed", "item": item or {}})
        return
    if not item:
        raise click.ClickException(f"math memory card not found: {concept}")
    console.print(f"[bold]{item.get('title')}[/bold]")
    console.print(item.get("concept_core") or item.get("note_core") or "")
    claim_refs = list(item.get("claim_refs") or [])
    if claim_refs:
        console.print(f"[dim]claim refs:[/dim] {len(claim_refs)}")


@math_memory_group.command("search")
@click.option("--query", required=True)
@click.option("--limit", type=int, default=10, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def search_math_memory(ctx, query, limit, as_json):
    """저장된 math memory card를 검색"""
    khub = ctx.obj["khub"]
    items = _service(khub).search(query=str(query).strip(), limit=max(1, int(limit)))
    payload = {
        "status": "ok",
        "query": str(query).strip(),
        "count": len(items),
        "items": items,
    }
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]math-memory search[/bold] query={query} count={len(items)}")
    for item in items[:10]:
        console.print(f"- {item.get('title')} :: {item.get('concept_core') or item.get('note_core') or ''}")


@math_memory_group.command("bridge-papers")
@click.option("--apply/--no-apply", default=False, show_default=True, help="paper->math concept relation 실제 반영")
@click.option(
    "--rebuild-paper-memory/--no-rebuild-paper-memory",
    default=False,
    show_default=True,
    help="관계 반영 후 paper-memory card 재생성",
)
@click.option(
    "--ensure-math-cards/--no-ensure-math-cards",
    default=True,
    show_default=True,
    help="브리지에서 참조한 수학 개념 노트의 math-memory card 보장",
)
@click.option("--allow-external/--no-allow-external", default=False, show_default=True, help="card rebuild 시 외부 API 사용 허용")
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="math card / paper-memory rebuild 시 LLM 라우팅 모드",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def bridge_papers_math_memory(ctx, apply, rebuild_paper_memory, ensure_math_cards, allow_external, llm_mode, as_json):
    """AI paper math bridge note를 paper-memory concept relation으로 동기화"""
    khub = ctx.obj["khub"]
    payload = _service(khub).sync_bridge_papers(
        apply=bool(apply),
        rebuild_paper_memory=bool(rebuild_paper_memory),
        ensure_math_cards=bool(ensure_math_cards),
        allow_external=bool(allow_external),
        llm_mode=str(llm_mode or "auto"),
    )
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]math-memory bridge-papers[/bold] apply={payload.get('apply')} "
        f"bridge={payload.get('bridgeCount')} matched={payload.get('matchedPaperCount')} "
        f"relations={payload.get('relationCount')} rebuilt={payload.get('rebuiltPaperMemoryCount')}"
    )
    for item in list(payload.get("items") or [])[:12]:
        console.print(
            f"- {item.get('bridgeNote')} :: status={item.get('status')} "
            f"paper={item.get('resolvedPaperId') or '-'} concepts={len(list(item.get('concepts') or []))}"
        )

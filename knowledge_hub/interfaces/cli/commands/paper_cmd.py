"""
khub paper - 논문 개별 관리 명령어

개별 작업:
  khub paper add <URL>           URL로 논문 추가 (arXiv, OpenReview, HuggingFace 등)
  khub paper import-csv <CSV>    CSV로 큐레이션 논문 배치 추가
  khub paper download <ID>       단일 다운로드
  khub paper translate <ID>      단일 번역
  khub paper summarize <ID>      단일 요약
  khub paper embed <ID>          단일 임베딩
  khub paper info <ID>           상세 정보

배치 작업:
  khub paper translate-all       미번역 전체 번역
  khub paper summarize-all       미요약 전체 요약
  khub paper embed-all           미인덱싱 전체 임베딩
  khub paper list                목록 조회
  khub paper board-export        Obsidian board용 JSON export
"""

from __future__ import annotations

import json
import logging
import re

import click
import requests
from pathlib import Path
from rich.console import Console

from knowledge_hub.interfaces.cli.commands.paper_admin_runtime import (
    run_paper_info,
    run_paper_list,
    run_paper_review,
)
from knowledge_hub.interfaces.cli.commands.paper_materialization_runtime import (
    run_paper_embed,
    run_paper_embed_all,
    run_paper_download,
    run_paper_summarize_all,
    run_paper_summarize,
    run_paper_translate_all,
    run_paper_translate,
)
from knowledge_hub.interfaces.cli.commands.paper_maintenance_runtime import (
    run_paper_build_concepts,
    run_paper_normalize_concepts,
    run_paper_resummary_vault,
    run_paper_sync_keywords,
)
from knowledge_hub.interfaces.cli.commands.paper_board_cmd import paper_board_export
from knowledge_hub.interfaces.cli.commands.paper_support import (
    assess_summary_quality as _assess_summary_quality,
    assess_vault_note_quality as _assess_vault_note_quality,
    batch_describe_concepts as _batch_describe_concepts,
    build_concept_note as _build_concept_note,
    collect_vault_note_text as _collect_vault_note_text,
    concept_id as _concept_id,
    detect_synonym_groups as _detect_synonym_groups,
    extract_keywords_with_evidence as _extract_keywords_with_evidence,
    extract_summary_text as _extract_summary_text,
    fallback_to_mini_llm as _fallback_to_mini_llm,
    merge_obsidian_concept as _merge_obsidian_concept,
    rebuild_concept_index_with_relations as _rebuild_concept_index_with_relations,
    regenerate_concept_index as _regenerate_concept_index,
    replace_in_paper_notes as _replace_in_paper_notes,
    resolve_routed_llm as _resolve_routed_llm,
    update_note_concepts as _update_note_concepts,
    update_vault_note_summary as _update_vault_note_summary,
    upsert_ai_concept as _upsert_ai_concept,
)
from knowledge_hub.papers.judge_feedback import PaperJudgeFeedbackLogger
from knowledge_hub.interfaces.cli.commands.paper_public_reading_cmd import (
    paper_public_evidence,
    paper_public_memory,
    paper_public_related,
    paper_public_summary,
)
from knowledge_hub.interfaces.cli.commands.paper_import_support import (
    default_manifest_path,
    parse_steps,
    render_import_summary,
    run_import_csv,
)
from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import (
    _build_embedder,
    _build_llm,
    _collect_paper_text,
    _extract_note_concepts,
    _resolve_vault_concepts_dir,
    _resolve_vault_papers_dir,
    _sqlite_db,
    _update_obsidian_summary,
    _validate_arxiv_id,
    _vector_db,
)

console = Console()
log = logging.getLogger("khub.paper")


@click.group("paper")
def paper_group():
    """논문 관리 (add/import-csv/download/translate/summarize/summary/evidence/memory/related/embed/list/info)"""
    pass


paper_group.add_command(paper_public_summary)
paper_group.add_command(paper_public_evidence)
paper_group.add_command(paper_public_memory)
paper_group.add_command(paper_public_related)
paper_group.add_command(paper_board_export)


@paper_group.command("feedback")
@click.argument("paper_id")
@click.option("--label", type=click.Choice(["keep", "skip"]), required=True, help="수동 판단 라벨")
@click.option("--reason", default="", help="판단 이유")
@click.option("--topic", default="", help="관련 주제")
@click.option("--title", default="", help="논문 제목 (DB/로그에 없을 때만 필요)")
@click.option("--source", default="manual", show_default=True, help="피드백 출처 라벨")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_feedback(ctx, paper_id, label, reason, topic, title, source, as_json):
    """Judge keep/skip 보정용 수동 피드백 기록"""
    khub = ctx.obj["khub"]
    config = khub.config
    sqlite_db = _sqlite_db(config, khub=khub)
    logger = PaperJudgeFeedbackLogger(config)

    paper = sqlite_db.get_paper(paper_id) or {}
    metadata = {
        "year": paper.get("year"),
        "field": paper.get("field"),
        "authors": paper.get("authors"),
    }
    payload = logger.log_feedback(
        paper_id=paper_id,
        label=label,
        source=source,
        reason=reason,
        title=title or str(paper.get("title") or ""),
        topic=topic,
        extra={key: value for key, value in metadata.items() if value not in {None, ""}},
    )

    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    console.print(
        f"[green]judge feedback recorded[/green] "
        f"{paper_id} → [bold]{label}[/bold]"
    )
    if payload.get("is_override"):
        console.print(
            f"[yellow]latest judge decision was {payload.get('judge_decision')} "
            f"(score={float(payload.get('judge_score', 0.0) or 0.0):.3f})[/yellow]"
        )
    if payload.get("reason"):
        console.print(f"[dim]{payload['reason']}[/dim]")


# ─────────────────────────────────────────────
# paper list
# ─────────────────────────────────────────────
@paper_group.command("list")
@click.option("--field", "-f", default=None, help="분야 필터")
@click.option("--limit", "-n", default=50, help="표시할 최대 수")
@click.pass_context
def paper_list(ctx, field, limit):
    """수집된 논문 목록"""
    run_paper_list(
        khub=ctx.obj["khub"],
        field=field,
        limit=limit,
        console=console,
        sqlite_db_fn=_sqlite_db,
    )


# ─────────────────────────────────────────────
# paper add <URL>
# ─────────────────────────────────────────────
@paper_group.command("add")
@click.argument("url")
@click.option("--download/--no-download", default=True, help="PDF 다운로드 여부")
@click.pass_context
def paper_add(ctx, url, download):
    """URL로 논문 추가 (arXiv, OpenReview, PapersWithCode, HuggingFace, S2, PDF URL)"""
    config = ctx.obj["khub"].config
    from knowledge_hub.papers.url_resolver import resolve_url
    from knowledge_hub.papers.downloader import PaperDownloader

    with console.status(f"[cyan]URL 분석 중: {url[:60]}...[/cyan]"):
        paper = resolve_url(url)

    if not paper:
        console.print("[red]논문을 찾을 수 없습니다.[/red]")
        return

    console.print(f"[bold]{paper.title}[/bold]")
    console.print(f"  저자: {paper.authors}")
    console.print(f"  연도: {paper.year} | 인용: {paper.citation_count} | 소스: {paper.source}")
    if paper.abstract:
        console.print(f"  초록: {paper.abstract[:120]}...")

    sqlite_db = _sqlite_db(config, khub=ctx.obj["khub"])
    existing = sqlite_db.get_paper(paper.arxiv_id) if paper.arxiv_id else None
    if existing:
        console.print(f"[yellow]이미 등록된 논문입니다: {paper.arxiv_id}[/yellow]")
        return

    paper_data = {
        "arxiv_id": paper.arxiv_id or re.sub(r'[^\w]', '_', paper.title)[:30],
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "field": ", ".join(paper.fields_of_study[:3]),
        "importance": 3,
        "notes": f"citations: {paper.citation_count}",
        "pdf_path": None,
        "text_path": None,
        "translated_path": None,
    }

    if download:
        downloader = PaperDownloader(config.papers_dir)
        with console.status("다운로드 중..."):
            result = downloader.download_single(paper.arxiv_id, paper.title)
        paper_data["pdf_path"] = result.get("pdf")
        paper_data["text_path"] = result.get("text")
        if result["success"]:
            console.print("  [green]PDF 다운로드 완료[/green]")
        else:
            console.print("  [yellow]PDF 다운로드 실패[/yellow]")

    sqlite_db.upsert_paper(paper_data)
    console.print(f"[green]논문 등록 완료: {paper_data['arxiv_id']}[/green]")
    console.print("[dim]khub paper summarize / translate / embed 로 후속 작업 가능[/dim]")


# ─────────────────────────────────────────────
# paper import-csv <path>
# ─────────────────────────────────────────────
@paper_group.command("import-csv")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--min-priority", type=int, default=5, show_default=True, help="처리할 최소 priority")
@click.option("--limit", type=int, default=0, show_default=True, help="최대 처리 수 (0=전체)")
@click.option(
    "--steps",
    default="register,download,embed,paper-memory,document-memory",
    show_default=True,
    help="실행할 단계 목록 (쉼표 구분)",
)
@click.option(
    "--manifest",
    "manifest_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="실행 상태 매니페스트 경로",
)
@click.option("--fail-fast/--continue-on-error", default=False, show_default=True, help="첫 실패에서 중단")
@click.option(
    "--document-memory-parser",
    type=click.Choice(["raw", "mineru", "opendataloader"]),
    default="raw",
    show_default=True,
    help="document-memory paper parser",
)
@click.option("--rebuild-memory/--no-rebuild-memory", default=False, show_default=True, help="기존 메모리도 다시 빌드")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_import_csv(
    ctx,
    csv_path,
    min_priority,
    limit,
    steps,
    manifest_path,
    fail_fast,
    document_memory_parser,
    rebuild_memory,
    as_json,
):
    """CSV로 큐레이션 논문 배치 추가"""
    try:
        selected_steps = parse_steps(steps)
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint="--steps") from error

    payload = run_import_csv(
        khub=ctx.obj["khub"],
        csv_path=str(csv_path),
        manifest_path=str(manifest_path or default_manifest_path(csv_path)),
        min_priority=max(0, int(min_priority)),
        limit=max(0, int(limit)),
        steps=selected_steps,
        fail_fast=bool(fail_fast),
        document_memory_parser=str(document_memory_parser or "raw"),
        rebuild_memory=bool(rebuild_memory),
    )
    if as_json:
        console.print_json(data=payload)
        return
    render_import_summary(payload)


# ─────────────────────────────────────────────
# paper download <arxiv_id>
# ─────────────────────────────────────────────
@paper_group.command("download")
@click.argument("arxiv_id")
@click.pass_context
def paper_download(ctx, arxiv_id):
    """단일 논문 PDF/텍스트 다운로드"""
    arxiv_id = _validate_arxiv_id(arxiv_id)
    from knowledge_hub.papers.downloader import PaperDownloader

    run_paper_download(
        khub=ctx.obj["khub"],
        arxiv_id=arxiv_id,
        console=console,
        sqlite_db_fn=_sqlite_db,
        downloader_factory=PaperDownloader,
    )


# ─────────────────────────────────────────────
# paper translate <arxiv_id>
# ─────────────────────────────────────────────
@paper_group.command("translate")
@click.argument("arxiv_id")
@click.option("--provider", "-p", default=None, help="번역 프로바이더 (기본: config)")
@click.option("--model", "-m", default=None, help="번역 모델 (기본: config)")
@click.option("--allow-external/--no-allow-external", default=False, show_default=True, help="외부 API 사용 허용")
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="LLM 라우팅 모드",
)
@click.pass_context
def paper_translate(ctx, arxiv_id, provider, model, allow_external, llm_mode):
    """논문 전체 번역 (arXiv ID 지정)"""
    arxiv_id = _validate_arxiv_id(arxiv_id)
    run_paper_translate(
        khub=ctx.obj["khub"],
        arxiv_id=arxiv_id,
        provider=provider,
        model=model,
        allow_external=allow_external,
        llm_mode=llm_mode,
        console=console,
        sqlite_db_fn=_sqlite_db,
        resolve_routed_llm_fn=_resolve_routed_llm,
        fallback_to_mini_llm_fn=_fallback_to_mini_llm,
    )


# ─────────────────────────────────────────────
# paper summarize <arxiv_id>
# ─────────────────────────────────────────────
@paper_group.command("summarize")
@click.argument("arxiv_id")
@click.option("--provider", "-p", default=None, help="요약 프로바이더 (기본: config)")
@click.option("--model", "-m", default=None, help="요약 모델 (기본: config)")
@click.option("--quick", is_flag=True, help="간단 요약 (5문장, abstract만 사용)")
@click.option("--allow-external/--no-allow-external", default=False, show_default=True, help="외부 API 사용 허용")
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="LLM 라우팅 모드",
)
@click.pass_context
def paper_summarize(ctx, arxiv_id, provider, model, quick, allow_external, llm_mode):
    """논문 심층 요약 생성 (구조화된 분석)"""
    arxiv_id = _validate_arxiv_id(arxiv_id)
    run_paper_summarize(
        khub=ctx.obj["khub"],
        arxiv_id=arxiv_id,
        provider=provider,
        model=model,
        quick=quick,
        allow_external=allow_external,
        llm_mode=llm_mode,
        console=console,
        sqlite_db_fn=_sqlite_db,
        collect_paper_text_fn=_collect_paper_text,
        resolve_routed_llm_fn=_resolve_routed_llm,
        fallback_to_mini_llm_fn=_fallback_to_mini_llm,
        update_obsidian_summary_fn=_update_obsidian_summary,
    )


@paper_group.command("review")
@click.option("--bad-only", is_flag=True, help="품질이 나쁜 요약만 표시 (점수 50 미만)")
@click.option("--threshold", "-t", default=50, help="나쁜 요약 기준 점수 (기본: 50)")
@click.option("--field", "-f", default=None, help="분야 필터")
@click.option("--show-summary", "-s", is_flag=True, help="요약 내용 미리보기 포함")
@click.option("--limit", "-n", default=100, help="최대 표시 수")
@click.pass_context
def paper_review(ctx, bad_only, threshold, field, show_summary, limit):
    """논문 요약 품질 리뷰 — 나쁜 요약을 식별하고 재요약 대상 파악

    \b
    사용 예시:
      khub paper review                     # 전체 품질 리뷰
      khub paper review --bad-only          # 나쁜 요약만 표시
      khub paper review --bad-only -s       # 나쁜 요약 + 내용 미리보기
      khub paper review -t 70              # 70점 미만만 표시

    \b
    품질 리뷰 후 재요약:
      khub paper summarize <arxiv_id>                    # 개별 재요약
      khub paper summarize-all --bad-only                # 나쁜 요약 일괄 재요약
      khub paper summarize-all --bad-only -p openai -m gpt-4o  # 더 좋은 모델로
    """
    run_paper_review(
        khub=ctx.obj["khub"],
        bad_only=bad_only,
        threshold=threshold,
        field=field,
        show_summary=show_summary,
        limit=limit,
        console=console,
        sqlite_db_fn=_sqlite_db,
        assess_summary_quality_fn=_assess_summary_quality,
    )


# ─────────────────────────────────────────────
# paper embed <arxiv_id>
# ─────────────────────────────────────────────
@paper_group.command("embed")
@click.argument("arxiv_id")
@click.pass_context
def paper_embed(ctx, arxiv_id):
    """단일 논문 벡터 임베딩"""
    arxiv_id = _validate_arxiv_id(arxiv_id)
    run_paper_embed(
        khub=ctx.obj["khub"],
        arxiv_id=arxiv_id,
        console=console,
        sqlite_db_fn=_sqlite_db,
        build_embedder_fn=_build_embedder,
        vector_db_fn=_vector_db,
    )


# ─────────────────────────────────────────────
# paper translate-all
# ─────────────────────────────────────────────
@paper_group.command("translate-all")
@click.option("--limit", "-n", default=0, help="최대 번역 수 (0=전체)")
@click.option("--field", "-f", default=None, help="분야 필터")
@click.option("--provider", "-p", default=None, help="번역 프로바이더")
@click.option("--model", "-m", default=None, help="번역 모델")
@click.pass_context
def paper_translate_all(ctx, limit, field, provider, model):
    """미번역 논문 전체 번역"""
    run_paper_translate_all(
        khub=ctx.obj["khub"],
        limit=limit,
        field=field,
        provider=provider,
        model=model,
        console=console,
        sqlite_db_fn=_sqlite_db,
        build_llm_fn=_build_llm,
    )


# ─────────────────────────────────────────────
# paper summarize-all
# ─────────────────────────────────────────────
@paper_group.command("summarize-all")
@click.option("--limit", "-n", default=0, help="최대 요약 수 (0=전체)")
@click.option("--field", "-f", default=None, help="분야 필터")
@click.option("--quick", is_flag=True, help="간단 요약 (구조화 분석 대신 3-5문장)")
@click.option("--resummary", is_flag=True, help="이미 요약된 논문도 재요약")
@click.option("--bad-only", is_flag=True, help="품질이 나쁜 요약만 재요약 (khub paper review로 확인)")
@click.option("--threshold", "-t", default=50, help="나쁜 요약 기준 점수 (--bad-only와 함께 사용, 기본: 50)")
@click.option("--provider", "-p", default=None, help="요약 프로바이더 (기본: config)")
@click.option("--model", "-m", default=None, help="요약 모델 (기본: config)")
@click.pass_context
def paper_summarize_all(ctx, limit, field, quick, resummary, bad_only, threshold, provider, model):
    """전체 논문 심층 요약 (구조화된 분석)

    \b
    사용 예시:
      khub paper summarize-all                              # 미요약 논문만
      khub paper summarize-all --resummary                  # 전체 재요약
      khub paper summarize-all --bad-only                   # 나쁜 요약만 재요약
      khub paper summarize-all --bad-only -p openai -m gpt-4o  # 좋은 모델로 재요약
      khub paper summarize-all --bad-only -t 70             # 70점 미만 재요약
    """
    run_paper_summarize_all(
        khub=ctx.obj["khub"],
        limit=limit,
        field=field,
        quick=quick,
        resummary=resummary,
        bad_only=bad_only,
        threshold=threshold,
        provider=provider,
        model=model,
        console=console,
        sqlite_db_fn=_sqlite_db,
        build_llm_fn=_build_llm,
        collect_paper_text_fn=_collect_paper_text,
        assess_summary_quality_fn=_assess_summary_quality,
        update_obsidian_summary_fn=_update_obsidian_summary,
        requests_post_fn=requests.post,
        log=log,
    )


# ─────────────────────────────────────────────
# paper embed-all
# ─────────────────────────────────────────────
@paper_group.command("embed-all")
@click.option("--all", "index_all", is_flag=True, help="이미 인덱싱된 논문도 재인덱싱")
@click.pass_context
def paper_embed_all(ctx, index_all):
    """미인덱싱 논문 전체 벡터 임베딩"""
    run_paper_embed_all(
        khub=ctx.obj["khub"],
        index_all=index_all,
        console=console,
        sqlite_db_fn=_sqlite_db,
        build_embedder_fn=_build_embedder,
        vector_db_fn=_vector_db,
    )


# ─────────────────────────────────────────────
# paper info <arxiv_id>
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# paper sync-keywords
# ─────────────────────────────────────────────
@paper_group.command("sync-keywords")
@click.option("--force", is_flag=True, help="이미 키워드가 있는 논문도 재추출")
@click.option("--limit", "-n", default=0, help="최대 처리 수 (0=전체)")
@click.option("--claims/--no-claims", default=True, show_default=True, help="키워드와 함께 claim 추출/저장")
@click.option("--allow-external/--no-allow-external", default=False, show_default=True, help="외부 API 사용 허용")
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="LLM 라우팅 모드",
)
@click.pass_context
def paper_sync_keywords(ctx, force, limit, claims, allow_external, llm_mode):
    """모든 논문에서 핵심 키워드+근거 추출 → ontology relation + claim 갱신"""
    from knowledge_hub.learning.resolver import EntityResolver
    from knowledge_hub.papers.claim_extractor import (
        estimate_evidence_quality,
        extract_claim_candidates,
        score_claim_with_breakdown,
    )

    run_paper_sync_keywords(
        khub=ctx.obj["khub"],
        force=force,
        limit=limit,
        claims=claims,
        allow_external=allow_external,
        llm_mode=llm_mode,
        console=console,
        resolve_vault_papers_dir_fn=_resolve_vault_papers_dir,
        sqlite_db_fn=_sqlite_db,
        extract_note_concepts_fn=_extract_note_concepts,
        extract_summary_text_fn=_extract_summary_text,
        resolve_routed_llm_fn=_resolve_routed_llm,
        extract_keywords_with_evidence_fn=_extract_keywords_with_evidence,
        concept_id_fn=_concept_id,
        upsert_ai_concept_fn=_upsert_ai_concept,
        update_note_concepts_fn=_update_note_concepts,
        regenerate_concept_index_fn=_regenerate_concept_index,
        entity_resolver_cls=EntityResolver,
        estimate_evidence_quality_fn=estimate_evidence_quality,
        extract_claim_candidates_fn=extract_claim_candidates,
        score_claim_with_breakdown_fn=score_claim_with_breakdown,
    )

# ─────────────────────────────────────────────
# paper build-concepts
# ─────────────────────────────────────────────
@paper_group.command("build-concepts")
@click.option("--force", is_flag=True, help="기존 개념 노트도 재생성")
@click.pass_context
def paper_build_concepts(ctx, force):
    """모든 키워드에 대해 개별 개념 노트 생성 + ontology relations에 관계 저장"""
    run_paper_build_concepts(
        khub=ctx.obj["khub"],
        force=force,
        console=console,
        resolve_vault_papers_dir_fn=_resolve_vault_papers_dir,
        resolve_vault_concepts_dir_fn=_resolve_vault_concepts_dir,
        build_llm_fn=_build_llm,
        sqlite_db_fn=_sqlite_db,
        extract_note_concepts_fn=_extract_note_concepts,
        batch_describe_concepts_fn=_batch_describe_concepts,
        concept_id_fn=_concept_id,
        upsert_ai_concept_fn=_upsert_ai_concept,
        build_concept_note_fn=_build_concept_note,
        rebuild_concept_index_with_relations_fn=_rebuild_concept_index_with_relations,
    )

# ─────────────────────────────────────────────
# paper normalize-concepts
# ─────────────────────────────────────────────
@paper_group.command("normalize-concepts")
@click.option("--dry-run", is_flag=True, help="변경 없이 탐지 결과만 표시")
@click.pass_context
def paper_normalize_concepts(ctx, dry_run):
    """개념 동의어/복수형/약어 탐지 → 정규화 + 병합"""
    run_paper_normalize_concepts(
        khub=ctx.obj["khub"],
        dry_run=dry_run,
        console=console,
        build_llm_fn=_build_llm,
        resolve_vault_papers_dir_fn=_resolve_vault_papers_dir,
        resolve_vault_concepts_dir_fn=_resolve_vault_concepts_dir,
        detect_synonym_groups_fn=_detect_synonym_groups,
        sqlite_db_fn=_sqlite_db,
        concept_id_fn=_concept_id,
        upsert_ai_concept_fn=_upsert_ai_concept,
        merge_obsidian_concept_fn=_merge_obsidian_concept,
        replace_in_paper_notes_fn=_replace_in_paper_notes,
        rebuild_concept_index_with_relations_fn=_rebuild_concept_index_with_relations,
    )

# ─────────────────────────────────────────────
# paper info <arxiv_id>
# ─────────────────────────────────────────────
@paper_group.command("info")
@click.argument("arxiv_id")
@click.pass_context
def paper_info(ctx, arxiv_id):
    """논문 상세 정보"""
    run_paper_info(
        khub=ctx.obj["khub"],
        arxiv_id=arxiv_id,
        console=console,
        sqlite_db_fn=_sqlite_db,
        validate_arxiv_id_fn=_validate_arxiv_id,
        assess_summary_quality_fn=_assess_summary_quality,
    )


# ─────────────────────────────────────────────
# paper resummary-vault
# ─────────────────────────────────────────────
@paper_group.command("resummary-vault")
@click.option("--bad-only", is_flag=True, default=True, help="부실한 요약만 재요약 (기본값)")
@click.option("--all", "resummary_all", is_flag=True, help="모든 노트 재요약")
@click.option("--threshold", "-t", default=60, help="재요약 기준 점수 (기본: 60)")
@click.option("--provider", "-p", default=None, help="요약 프로바이더")
@click.option("--model", "-m", default=None, help="요약 모델")
@click.option("--limit", "-n", default=0, help="최대 처리 수 (0=전체)")
@click.option("--dry-run", is_flag=True, help="변경 없이 대상만 표시")
@click.pass_context
def paper_resummary_vault(ctx, bad_only, resummary_all, threshold, provider, model, limit, dry_run):
    """Obsidian vault 논문 노트의 부실한 요약을 재생성

    \b
    사용 예시:
      khub paper resummary-vault --dry-run          # 대상만 확인
      khub paper resummary-vault                     # 부실한 요약 재생성
      khub paper resummary-vault -p openai -m gpt-4o # 모델 지정
      khub paper resummary-vault --all               # 전체 재요약
    """
    run_paper_resummary_vault(
        khub=ctx.obj["khub"],
        bad_only=bad_only,
        resummary_all=resummary_all,
        threshold=threshold,
        provider=provider,
        model=model,
        limit=limit,
        dry_run=dry_run,
        console=console,
        assess_vault_note_quality_fn=_assess_vault_note_quality,
        resolve_vault_papers_dir_fn=_resolve_vault_papers_dir,
        build_llm_fn=_build_llm,
        collect_vault_note_text_fn=_collect_vault_note_text,
        update_vault_note_summary_fn=_update_vault_note_summary,
        log=log,
    )

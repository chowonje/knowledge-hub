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
  khub paper review-card <ID>    memory card 품질 피드백 기록
  khub paper review-card-plan    카드 품질 remediation plan 확인
  khub paper review-card-apply   안전한 remediation action 실행
  khub paper review-card-export  audit/rebuild 대상 paper id export
  khub paper canon-quality-audit AI canon 9편 전용 deterministic quality audit
  khub paper repair-source       known source contamination relink + artifact rebuild
  khub paper repair-source-queue paper source repair action queue 적재

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
from typing import Any

import click
from pathlib import Path
from rich.console import Console

from knowledge_hub.application.paper_source_freshness import audit_paper_source_freshness
from knowledge_hub.application.paper_source_repairs import queue_paper_source_repairs, repair_paper_sources
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.interfaces.cli.commands.paper_admin_runtime import (
    run_paper_info,
    run_paper_list,
    run_paper_review,
)
from knowledge_hub.interfaces.cli.commands.paper_materialization_runtime import (
    _run_structured_summary_batch_worker,
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
from knowledge_hub.papers.card_feedback import (
    CARD_QUALITY_ISSUES,
    PaperCardFeedbackLogger,
    build_card_remediation_plan,
)
from knowledge_hub.papers.canon_quality_audit import (
    DEFAULT_MANIFEST_PATH as _DEFAULT_CANON_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR as _DEFAULT_CANON_OUTPUT_DIR,
    SCHEMA_ID as _CANON_QUALITY_AUDIT_SCHEMA,
    audit_canon_quality as _audit_canon_quality,
    remediation_needs_concept_refresh as _canon_needs_concept_refresh,
    remediation_needs_memory_rebuild as _canon_needs_memory_rebuild,
    remediation_needs_source_repair as _canon_needs_source_repair,
    remediation_needs_summary_rebuild as _canon_needs_summary_rebuild,
    write_canon_audit_outputs as _write_canon_audit_outputs,
)
from knowledge_hub.papers.source_guard import review_downloaded_source, stage_source_guard
from knowledge_hub.papers.memory_runtime import build_paper_memory_builder
from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever
from knowledge_hub.papers.public_surface import build_public_summary_card as _build_public_summary_card
from knowledge_hub.papers.structured_summary import StructuredPaperSummaryService
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
    _extract_note_concepts,
    _paper_summary_parser,
    _render_structured_summary_notes,
    _resolve_vault_concepts_dir,
    _resolve_vault_papers_dir,
    _sqlite_db,
    _sync_structured_summary_view,
    _validate_arxiv_id,
    _vector_db,
)

console = Console()
log = logging.getLogger("khub.paper")


def _validate_cli_payload(config, payload: dict[str, Any], schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


@click.group("paper")
def paper_group():
    """논문 관리 (add/import-csv/download/translate/summarize/summary/evidence/memory/related/embed/list/info)"""
    pass


paper_group.add_command(paper_public_summary)
paper_group.add_command(paper_public_evidence)
paper_group.add_command(paper_public_memory)
paper_group.add_command(paper_public_related)
paper_group.add_command(paper_board_export)


def _paper_card_feedback_context(*, khub, paper_id: str) -> dict[str, Any]:
    config = khub.config
    sqlite_db = _sqlite_db(config, khub=khub)
    token = str(paper_id).strip()
    paper = sqlite_db.get_paper(token) or {}
    summary_service = StructuredPaperSummaryService(sqlite_db, config)
    summary_payload = dict(summary_service.load_artifact(paper_id=token) or {})
    summary = dict(summary_payload.get("summary") or {})
    memory_card = dict(PaperMemoryRetriever(sqlite_db).get(token, include_refs=False) or {})
    has_summary = bool(" ".join(str(summary.get(key) or "").strip() for key in ("oneLine", "coreIdea")).strip())
    has_memory = bool(
        " ".join(str(memory_card.get(key) or "").strip() for key in ("paperCore", "methodCore", "evidenceCore")).strip()
    )
    observed_warnings: list[str] = []
    if not has_summary:
        observed_warnings.append("summary_artifact_missing")
    if not has_memory:
        observed_warnings.append("memory_card_missing")
    if has_memory and not list(memory_card.get("conceptLinks") or []):
        observed_warnings.append("concept_links_missing")
    return {
        "paper": paper,
        "artifactFlags": {
            "hasPdf": bool(str(paper.get("pdf_path") or "").strip()),
            "hasSummary": has_summary,
            "hasTranslation": bool(str(paper.get("translated_path") or "").strip()),
            "isIndexed": bool(paper.get("indexed")),
            "hasMemory": has_memory,
        },
        "summarySnapshot": {
            "oneLine": str(summary.get("oneLine") or "").strip(),
            "coreIdea": str(summary.get("coreIdea") or "").strip(),
            "parserUsed": str(summary_payload.get("parserUsed") or summary_payload.get("parser_used") or "").strip(),
            "fallbackUsed": bool(summary_payload.get("fallbackUsed")),
        },
        "memorySnapshot": {
            "qualityFlag": str(memory_card.get("qualityFlag") or "").strip(),
            "paperCore": str(memory_card.get("paperCore") or "").strip(),
            "problemContext": str(memory_card.get("problemContext") or "").strip(),
            "methodCore": str(memory_card.get("methodCore") or "").strip(),
            "evidenceCore": str(memory_card.get("evidenceCore") or "").strip(),
            "limitations": str(memory_card.get("limitations") or "").strip(),
            "conceptLinks": list(memory_card.get("conceptLinks") or []),
        },
        "observedWarnings": observed_warnings,
    }


def _parse_cli_paper_ids(*, paper_ids: list[str] | tuple[str, ...], paper_id_file: Path | None = None) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in list(paper_ids or []):
        token = str(raw or "").strip()
        if not token or token.casefold() in seen:
            continue
        seen.add(token.casefold())
        items.append(token)
    if paper_id_file is not None and paper_id_file.exists():
        for line in paper_id_file.read_text(encoding="utf-8").splitlines():
            token = str(line or "").strip()
            if not token or token.casefold() in seen:
                continue
            seen.add(token.casefold())
            items.append(token)
    return items


def _logged_card_issues(logger: PaperCardFeedbackLogger, *, paper_id: str) -> list[str]:
    issues: list[str] = []
    for event in logger.load_feedback_events():
        if str(event.get("paper_id") or "").strip() != str(paper_id).strip():
            continue
        for issue in list(event.get("issues") or []):
            token = str(issue or "").strip()
            if token and token not in issues:
                issues.append(token)
    return issues


def _paper_card_plan_payload(
    *,
    khub: Any,
    paper_id: str,
    issues: list[str] | tuple[str, ...] | None = None,
    from_log: bool = True,
) -> dict[str, Any]:
    logger = PaperCardFeedbackLogger(khub.config)
    card_context = _paper_card_feedback_context(khub=khub, paper_id=paper_id)
    paper = dict(card_context.get("paper") or {})
    combined_issues = list(_logged_card_issues(logger, paper_id=paper_id)) if from_log else []
    for issue in list(issues or []):
        token = str(issue or "").strip()
        if token and token not in combined_issues:
            combined_issues.append(token)
    return {
        "status": "ok",
        "paperId": str(paper_id).strip(),
        "paperTitle": str(paper.get("title") or "").strip(),
        "issues": combined_issues,
        "artifactFlags": dict(card_context.get("artifactFlags") or {}),
        "summarySnapshot": dict(card_context.get("summarySnapshot") or {}),
        "memorySnapshot": dict(card_context.get("memorySnapshot") or {}),
        "observedWarnings": list(card_context.get("observedWarnings") or []),
        "remediationPlan": build_card_remediation_plan(
            issues=combined_issues,
            artifact_flags=dict(card_context.get("artifactFlags") or {}),
            summary_snapshot=dict(card_context.get("summarySnapshot") or {}),
            memory_snapshot=dict(card_context.get("memorySnapshot") or {}),
            observed_warnings=list(card_context.get("observedWarnings") or []),
        ),
    }


def _paper_memory_builder_context(payload: dict[str, Any], sqlite_db: Any) -> tuple[str, str]:
    title = str(payload.get("paperTitle") or "").strip()
    summary_snapshot = dict(payload.get("summarySnapshot") or {})
    memory_snapshot = dict(payload.get("memorySnapshot") or {})
    paper = sqlite_db.get_paper(str(payload.get("paperId") or "").strip()) or {}
    context_parts = [
        str(summary_snapshot.get("oneLine") or "").strip(),
        str(summary_snapshot.get("coreIdea") or "").strip(),
        str(memory_snapshot.get("paperCore") or "").strip(),
        str(paper.get("notes") or "").strip(),
    ]
    context = "\n\n".join(part for part in context_parts if part).strip()
    return title, context


def _load_review_card_target_ids(
    *,
    logger: PaperCardFeedbackLogger,
    paper_ids: list[str] | tuple[str, ...] | None = None,
    paper_id_file: Path | None = None,
    issues: list[str] | tuple[str, ...] | None = None,
    limit: int = 0,
) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        token = str(raw or "").strip()
        if not token or token in seen:
            return
        seen.add(token)
        targets.append(token)

    for token in list(paper_ids or []):
        _add(token)
    if paper_id_file is not None and paper_id_file.exists():
        for line in paper_id_file.read_text(encoding="utf-8").splitlines():
            _add(line)
    if issues:
        for item in logger.build_export_queue(issues=list(issues), limit=max(0, int(limit))):
            _add(str(item.get("paperId") or ""))
    return targets


def _execute_card_remediation_actions(
    *,
    khub: Any,
    payload: dict[str, Any],
    provider: str | None,
    model: str | None,
    allow_external: bool,
    llm_mode: str,
) -> dict[str, Any]:
    plan = dict(payload.get("remediationPlan") or {})
    actions = list(plan.get("actions") or [])
    executed_actions: list[dict[str, Any]] = []
    skipped_actions: list[dict[str, Any]] = []
    blocked_actions: list[dict[str, Any]] = []
    failed_actions: list[dict[str, Any]] = []

    if not actions:
        return {
            "status": "noop",
            "executedActions": executed_actions,
            "skippedActions": skipped_actions,
            "blockedActions": blocked_actions,
            "failedActions": failed_actions,
        }

    if bool(plan.get("requiresManualReview")):
        for action in actions:
            blocked_actions.append(
                {
                    "code": str(action.get("code") or ""),
                    "reason": "manual source repair is required before auto remediation can run",
                }
            )
        return {
            "status": "blocked",
            "executedActions": executed_actions,
            "skippedActions": skipped_actions,
            "blockedActions": blocked_actions,
            "failedActions": failed_actions,
        }

    sqlite_db = _sqlite_db(khub.config, khub=khub)
    paper_id = str(payload.get("paperId") or "").strip()
    paper_title, memory_context = _paper_memory_builder_context(payload, sqlite_db)
    memory_refreshed = False

    for action in actions:
        code = str(action.get("code") or "").strip()
        if not code:
            continue
        if not bool(action.get("autoApply")):
            blocked_actions.append({"code": code, "reason": "manual-only remediation action"})
            continue
        try:
            if code == "rebuild_structured_summary":
                run_paper_summarize(
                    khub=khub,
                    arxiv_id=paper_id,
                    provider=provider,
                    model=model,
                    quick=False,
                    allow_external=allow_external,
                    llm_mode=llm_mode,
                    console=console,
                    sqlite_db_fn=_sqlite_db,
                    structured_summary_service_factory=StructuredPaperSummaryService,
                    paper_summary_parser_fn=_paper_summary_parser,
                    sync_structured_summary_view_fn=_sync_structured_summary_view,
                )
                executed_actions.append({"code": code, "status": "ok"})
                continue

            if code in {"rebuild_paper_memory", "refresh_concept_links"}:
                if memory_refreshed:
                    skipped_actions.append(
                        {"code": code, "reason": "paper memory rebuild already refreshed concept links in this run"}
                    )
                    continue
                builder = build_paper_memory_builder(
                    sqlite_db,
                    config=khub.config,
                    allow_external=allow_external,
                    llm_mode=llm_mode,
                    query=paper_title,
                    context=memory_context,
                    source_count=1,
                )
                item = builder.build_and_store(paper_id=paper_id)
                executed_actions.append(
                    {
                        "code": code,
                        "status": "ok",
                        "paperId": str(item.get("paperId") or paper_id),
                        "qualityFlag": str(item.get("qualityFlag") or ""),
                    }
                )
                memory_refreshed = True
                continue

            blocked_actions.append({"code": code, "reason": "no safe executor is registered for this action"})
        except Exception as error:  # pragma: no cover - defensive surface for operator command
            failed_actions.append({"code": code, "error": str(error)})

    status = "ok"
    if failed_actions:
        status = "failed"
    elif blocked_actions and not executed_actions:
        status = "blocked"
    elif not executed_actions and skipped_actions:
        status = "noop"
    return {
        "status": status,
        "executedActions": executed_actions,
        "skippedActions": skipped_actions,
        "blockedActions": blocked_actions,
        "failedActions": failed_actions,
    }


def _canon_quality_action_order(issues: list[str] | tuple[str, ...]) -> list[str]:
    summary_needed = _canon_needs_summary_rebuild(issues)
    memory_needed = _canon_needs_memory_rebuild(issues) or summary_needed
    concept_only = _canon_needs_concept_refresh(issues) and not memory_needed
    actions: list[str] = []
    if _canon_needs_source_repair(issues):
        actions.append("repair_source_content")
    if summary_needed:
        actions.append("rebuild_structured_summary")
    if memory_needed:
        actions.append("rebuild_paper_memory")
    if concept_only:
        actions.append("refresh_concept_links")
    return actions


def _canon_can_continue_after_source_repair_block(
    issues: list[str] | tuple[str, ...],
    *,
    reason: str,
) -> bool:
    reason_text = str(reason or "").strip().casefold()
    if "no cleanup rule available" not in reason_text:
        return False
    hard_source_issues = {"likely_semantic_mismatch", "latex_core", "latex_problem_context", "text_starts_latex"}
    issue_set = {str(item or "").strip() for item in list(issues or []) if str(item or "").strip()}
    return not bool(issue_set & hard_source_issues)


def _execute_canon_quality_remediation(
    *,
    khub: Any,
    paper_id: str,
    paper_title: str,
    issues: list[str] | tuple[str, ...],
    provider: str,
    model: str,
    allow_external: bool,
    llm_mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    planned_actions = _canon_quality_action_order(issues)
    executed_actions: list[dict[str, Any]] = []
    skipped_actions: list[dict[str, Any]] = []
    blocked_actions: list[dict[str, Any]] = []
    failed_actions: list[dict[str, Any]] = []

    payload = {
        "paperId": str(paper_id).strip(),
        "paperTitle": str(paper_title or "").strip(),
        "issues": [str(item or "").strip() for item in list(issues or []) if str(item or "").strip()],
        "dryRun": bool(dry_run),
        "plannedActions": planned_actions,
        "executedActions": executed_actions,
        "skippedActions": skipped_actions,
        "blockedActions": blocked_actions,
        "failedActions": failed_actions,
    }
    if not planned_actions:
        payload["status"] = "noop"
        return payload
    if dry_run:
        payload["status"] = "planned"
        return payload

    sqlite_db = _sqlite_db(khub.config, khub=khub)
    summary_needed = _canon_needs_summary_rebuild(issues)
    memory_needed = _canon_needs_memory_rebuild(issues) or summary_needed
    concept_only = _canon_needs_concept_refresh(issues) and not memory_needed

    if _canon_needs_source_repair(issues):
        repair_payload = repair_paper_sources(
            sqlite_db=sqlite_db,
            config=khub.config,
            paper_ids=[paper_id],
            document_memory_parser="raw",
            allow_external=allow_external,
            llm_mode=llm_mode,
            dry_run=False,
            rebuild=False,
        )
        repair_item = dict((repair_payload.get("items") or [{}])[0] or {})
        repair_status = str(repair_item.get("repairStatus") or repair_payload.get("status") or "").strip()
        if repair_status in {"blocked", "missing"}:
            repair_reason = str(repair_item.get("resolutionReason") or repair_item.get("error") or repair_status)
            blocked_actions.append(
                {
                    "code": "repair_source_content",
                    "reason": repair_reason,
                }
            )
            if not (
                repair_status == "blocked"
                and _canon_can_continue_after_source_repair_block(issues, reason=repair_reason)
            ):
                if summary_needed:
                    skipped_actions.append(
                        {"code": "rebuild_structured_summary", "reason": "source repair did not complete"}
                    )
                if memory_needed:
                    skipped_actions.append({"code": "rebuild_paper_memory", "reason": "source repair did not complete"})
                elif concept_only:
                    skipped_actions.append({"code": "refresh_concept_links", "reason": "source repair did not complete"})
                payload["status"] = "blocked"
                return payload
        if repair_status == "failed":
            failed_actions.append(
                {
                    "code": "repair_source_content",
                    "error": str(repair_item.get("error") or "source repair failed"),
                }
            )
            if summary_needed:
                skipped_actions.append(
                    {"code": "rebuild_structured_summary", "reason": "source repair failed"}
                )
            if memory_needed:
                skipped_actions.append({"code": "rebuild_paper_memory", "reason": "source repair failed"})
            elif concept_only:
                skipped_actions.append({"code": "refresh_concept_links", "reason": "source repair failed"})
            payload["status"] = "failed"
            return payload
        if repair_status not in {"blocked", "missing"}:
            executed_actions.append(
                {
                    "code": "repair_source_content",
                    "status": "ok",
                    "repairStatus": repair_status or "ok",
                    "action": str(repair_item.get("action") or ""),
                    "canonicalPaperId": str(repair_item.get("canonicalPaperId") or ""),
                }
            )

    if summary_needed:
        try:
            run_paper_summarize(
                khub=khub,
                arxiv_id=paper_id,
                provider=provider,
                model=model,
                quick=False,
                allow_external=allow_external,
                llm_mode=llm_mode,
                console=console,
                sqlite_db_fn=_sqlite_db,
                structured_summary_service_factory=StructuredPaperSummaryService,
                paper_summary_parser_fn=_paper_summary_parser,
                sync_structured_summary_view_fn=_sync_structured_summary_view,
            )
            executed_actions.append({"code": "rebuild_structured_summary", "status": "ok"})
        except Exception as error:  # pragma: no cover - defensive operator path
            failed_actions.append({"code": "rebuild_structured_summary", "error": str(error)})
            if memory_needed:
                skipped_actions.append(
                    {"code": "rebuild_paper_memory", "reason": "structured summary rebuild failed"}
                )
            elif concept_only:
                skipped_actions.append(
                    {"code": "refresh_concept_links", "reason": "structured summary rebuild failed"}
                )
            payload["status"] = "failed"
            return payload

    if memory_needed or concept_only:
        action_code = "rebuild_paper_memory" if memory_needed else "refresh_concept_links"
        try:
            builder = build_paper_memory_builder(
                sqlite_db,
                config=khub.config,
                allow_external=allow_external,
                llm_mode=llm_mode,
                query=paper_title,
                context="",
                source_count=1,
            )
            item = builder.build_and_store(paper_id=paper_id)
            executed_actions.append(
                {
                    "code": action_code,
                    "status": "ok",
                    "paperId": str(item.get("paperId") or paper_id),
                    "qualityFlag": str(item.get("qualityFlag") or item.get("quality_flag") or ""),
                }
            )
        except Exception as error:  # pragma: no cover - defensive operator path
            failed_actions.append({"code": action_code, "error": str(error)})
            payload["status"] = "failed"
            return payload

    payload["status"] = "ok" if executed_actions else "noop"
    return payload


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


@paper_group.command("review-card")
@click.argument("paper_id")
@click.option(
    "--issue",
    "issues",
    multiple=True,
    required=True,
    type=click.Choice(CARD_QUALITY_ISSUES),
    help="카드 품질 이슈 코드",
)
@click.option("--note", default="", help="상세 메모")
@click.option("--title", default="", help="논문 제목 (DB/로그에 없을 때만 필요)")
@click.option("--source", default="manual", show_default=True, help="피드백 출처 라벨")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_review_card(ctx, paper_id, issues, note, title, source, as_json):
    """빈약한 paper memory/summary 카드 품질 피드백 기록"""
    khub = ctx.obj["khub"]
    config = khub.config
    logger = PaperCardFeedbackLogger(config)
    card_context = _paper_card_feedback_context(khub=khub, paper_id=paper_id)
    paper = dict(card_context.get("paper") or {})

    event = logger.log_feedback(
        paper_id=paper_id,
        issues=list(issues),
        source=source,
        note=note,
        title=title or str(paper.get("title") or ""),
        extra={
            key: value
            for key, value in {
                "year": paper.get("year"),
                "field": paper.get("field"),
                "authors": paper.get("authors"),
            }.items()
            if value not in {None, ""}
        },
        artifact_flags=dict(card_context.get("artifactFlags") or {}),
        summary_snapshot=dict(card_context.get("summarySnapshot") or {}),
        memory_snapshot=dict(card_context.get("memorySnapshot") or {}),
        observed_warnings=list(card_context.get("observedWarnings") or []),
    )
    payload = {
        **event,
        "remediationPlan": build_card_remediation_plan(
            issues=list(event.get("issues") or []),
            artifact_flags=dict(card_context.get("artifactFlags") or {}),
            summary_snapshot=dict(card_context.get("summarySnapshot") or {}),
            memory_snapshot=dict(card_context.get("memorySnapshot") or {}),
            observed_warnings=list(card_context.get("observedWarnings") or []),
        ),
    }

    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    console.print(
        f"[green]card quality feedback recorded[/green] "
        f"{paper_id} → [bold]{', '.join(payload.get('issues') or [])}[/bold]"
    )
    if payload.get("note"):
        console.print(f"[dim]{payload['note']}[/dim]")
    plan = dict(payload.get("remediationPlan") or {})
    if str(plan.get("primaryAction") or "") not in {"", "none"}:
        console.print(
            f"[dim]next: {plan.get('primaryAction')} "
            f"(auto={','.join(list(plan.get('autoApplyActions') or [])) or 'none'})[/dim]"
        )


@paper_group.command("review-card-export")
@click.option(
    "--issue",
    "issues",
    multiple=True,
    type=click.Choice(CARD_QUALITY_ISSUES),
    help="특정 카드 품질 이슈만 export",
)
@click.option("--limit", type=int, default=0, show_default=True, help="내보낼 최대 paper 수 (0=전체)")
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="newline-delimited paper id 파일 경로",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_review_card_export(ctx, issues, limit, output_path, as_json):
    """카드 품질 피드백 로그를 audit/rebuild 대상 paper id로 export"""
    khub = ctx.obj["khub"]
    logger = PaperCardFeedbackLogger(khub.config)
    items = logger.build_export_queue(issues=list(issues), limit=max(0, int(limit)))
    paper_ids = [str(item.get("paperId") or "") for item in items if str(item.get("paperId") or "").strip()]

    written_path = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(f"{paper_id}\n" for paper_id in paper_ids), encoding="utf-8")
        written_path = str(output_path)

    payload = {
        "status": "ok",
        "count": len(paper_ids),
        "issueFilter": list(issues),
        "paperIds": paper_ids,
        "items": items,
        "outputPath": written_path or "",
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    console.print(f"[green]card review export ready[/green] count={len(paper_ids)}")
    if written_path:
        console.print(f"[dim]{written_path}[/dim]")
    for item in items[:10]:
        console.print(
            f"- {item.get('paperId')} "
            f"issues={','.join(item.get('issues') or [])} "
            f"events={item.get('eventCount')} "
            f"plan={((item.get('remediationPlan') or {}).get('primaryAction') or 'none')}"
        )


@paper_group.command("review-card-plan")
@click.argument("paper_id")
@click.option(
    "--issue",
    "issues",
    multiple=True,
    type=click.Choice(CARD_QUALITY_ISSUES),
    help="추가로 반영할 카드 품질 이슈 코드",
)
@click.option(
    "--from-log/--no-from-log",
    default=True,
    show_default=True,
    help="기존 review-card 로그의 누적 이슈를 함께 반영",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_review_card_plan(ctx, paper_id, issues, from_log, as_json):
    """현재 카드 스냅샷과 review-card 로그를 바탕으로 remediation plan 계산"""
    khub = ctx.obj["khub"]
    payload = _paper_card_plan_payload(khub=khub, paper_id=paper_id, issues=list(issues), from_log=from_log)
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    plan = dict(payload.get("remediationPlan") or {})
    console.print(
        f"[bold]card remediation plan[/bold] paper={payload.get('paperId')} "
        f"primary={plan.get('primaryAction') or 'none'}"
    )
    for action in list(plan.get("actions") or []):
        console.print(
            f"- {action.get('code')} "
            f"auto={bool(action.get('autoApply'))} "
            f"reason={'; '.join(list(action.get('reasons') or []))}"
        )


@paper_group.command("review-card-apply")
@click.argument("paper_id")
@click.option(
    "--issue",
    "issues",
    multiple=True,
    type=click.Choice(CARD_QUALITY_ISSUES),
    help="추가로 반영할 카드 품질 이슈 코드",
)
@click.option(
    "--from-log/--no-from-log",
    default=True,
    show_default=True,
    help="기존 review-card 로그의 누적 이슈를 함께 반영",
)
@click.option("--provider", default=None, help="구조화 summary 재빌드용 provider override")
@click.option("--model", default=None, help="구조화 summary 재빌드용 model override")
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
    help="LLM 라우팅 모드",
)
@click.option("--dry-run/--no-dry-run", default=False, show_default=True, help="실행 없이 계획만 계산")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_review_card_apply(ctx, paper_id, issues, from_log, provider, model, allow_external, llm_mode, dry_run, as_json):
    """안전한 카드 remediation action 실행 (summary/memory rebuild 등)"""
    khub = ctx.obj["khub"]
    payload = _paper_card_plan_payload(khub=khub, paper_id=paper_id, issues=list(issues), from_log=from_log)
    if dry_run:
        payload.update(
            {
                "dryRun": True,
                "executedActions": [],
                "skippedActions": [],
                "blockedActions": [],
                "failedActions": [],
            }
        )
    else:
        payload.update(
            {
                "dryRun": False,
                **_execute_card_remediation_actions(
                    khub=khub,
                    payload=payload,
                    provider=provider,
                    model=model,
                    allow_external=allow_external,
                    llm_mode=llm_mode,
                ),
            }
        )
        payload["postContext"] = _paper_card_feedback_context(khub=khub, paper_id=paper_id)
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    plan = dict(payload.get("remediationPlan") or {})
    console.print(
        f"[bold]card remediation apply[/bold] paper={payload.get('paperId')} "
        f"primary={plan.get('primaryAction') or 'none'} status={payload.get('status')}"
    )
    if dry_run:
        console.print("[dim]dry-run only; no actions executed[/dim]")
    for action in list(payload.get("executedActions") or []):
        console.print(f"- executed {action.get('code')}")
    for action in list(payload.get("skippedActions") or []):
        console.print(f"- skipped {action.get('code')}: {action.get('reason')}")
    for action in list(payload.get("blockedActions") or []):
        console.print(f"- blocked {action.get('code')}: {action.get('reason')}")
    for action in list(payload.get("failedActions") or []):
        console.print(f"- failed {action.get('code')}: {action.get('error')}")


@paper_group.command("review-card-apply-batch")
@click.option("--paper-id", "paper_ids", multiple=True, help="대상 paper id (여러 번 사용 가능)")
@click.option(
    "--paper-id-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="newline-delimited paper id 파일",
)
@click.option(
    "--issue",
    "issues",
    multiple=True,
    type=click.Choice(CARD_QUALITY_ISSUES),
    help="review-card 로그에서 특정 issue를 가진 paper만 선택",
)
@click.option("--limit", type=int, default=0, show_default=True, help="issue filter로 고른 최대 paper 수 (0=전체)")
@click.option(
    "--from-log/--no-from-log",
    default=True,
    show_default=True,
    help="각 paper의 기존 review-card 로그 누적 이슈를 함께 반영",
)
@click.option("--provider", default=None, help="구조화 summary 재빌드용 provider override")
@click.option("--model", default=None, help="구조화 summary 재빌드용 model override")
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
    help="LLM 라우팅 모드",
)
@click.option("--dry-run/--no-dry-run", default=False, show_default=True, help="실행 없이 계획만 계산")
@click.option("--fail-fast/--continue-on-error", default=False, show_default=True, help="첫 실패에서 중단")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_review_card_apply_batch(
    ctx,
    paper_ids,
    paper_id_file,
    issues,
    limit,
    from_log,
    provider,
    model,
    allow_external,
    llm_mode,
    dry_run,
    fail_fast,
    as_json,
):
    """review-card queue를 안전한 auto remediation으로 일괄 적용"""
    khub = ctx.obj["khub"]
    logger = PaperCardFeedbackLogger(khub.config)
    selector_supplied = bool(list(paper_ids) or paper_id_file is not None or list(issues))
    targets = _load_review_card_target_ids(
        logger=logger,
        paper_ids=list(paper_ids),
        paper_id_file=paper_id_file,
        issues=list(issues),
        limit=limit,
    )
    if not selector_supplied:
        raise click.ClickException("at least one target selector is required (--paper-id, --paper-id-file, or --issue)")
    if not targets:
        payload = {
            "status": "ok",
            "dryRun": bool(dry_run),
            "targetCount": 0,
            "processedCount": 0,
            "selectedByIssues": list(issues),
            "paperIdFile": str(paper_id_file) if paper_id_file is not None else "",
            "counts": {"ok": 0, "blocked": 0, "failed": 0, "noop": 0},
            "stoppedEarly": False,
            "items": [],
        }
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        console.print("[yellow]card remediation batch[/yellow] no matching targets")
        return

    items: list[dict[str, Any]] = []
    counts = {"ok": 0, "blocked": 0, "failed": 0, "noop": 0}
    stopped_early = False

    for token in targets:
        item = _paper_card_plan_payload(khub=khub, paper_id=token, issues=[], from_log=from_log)
        if dry_run:
            item.update(
                {
                    "dryRun": True,
                    "executedActions": [],
                    "skippedActions": [],
                    "blockedActions": [],
                    "failedActions": [],
                }
            )
        else:
            item.update(
                {
                    "dryRun": False,
                    **_execute_card_remediation_actions(
                        khub=khub,
                        payload=item,
                        provider=provider,
                        model=model,
                        allow_external=allow_external,
                        llm_mode=llm_mode,
                    ),
                }
            )
        status = str(item.get("status") or "noop")
        counts[status] = int(counts.get(status, 0)) + 1
        items.append(item)
        if fail_fast and status == "failed":
            stopped_early = True
            break

    payload = {
        "status": "ok" if counts.get("failed", 0) == 0 else "failed",
        "dryRun": bool(dry_run),
        "targetCount": len(targets),
        "processedCount": len(items),
        "selectedByIssues": list(issues),
        "paperIdFile": str(paper_id_file) if paper_id_file is not None else "",
        "counts": counts,
        "stoppedEarly": bool(stopped_early),
        "items": items,
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    console.print(
        f"[bold]card remediation batch[/bold] processed={payload.get('processedCount')} "
        f"ok={counts.get('ok', 0)} blocked={counts.get('blocked', 0)} failed={counts.get('failed', 0)}"
    )
    if dry_run:
        console.print("[dim]dry-run only; no actions executed[/dim]")
    for item in items[:10]:
        console.print(
            f"- {item.get('paperId')} status={item.get('status')} "
            f"primary={((item.get('remediationPlan') or {}).get('primaryAction') or 'none')}"
        )


@paper_group.command("canon-quality-audit")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_CANON_MANIFEST_PATH,
    show_default=True,
    help="AI canon source-of-truth manifest CSV",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_CANON_OUTPUT_DIR,
    show_default=True,
    help="report/selector 산출물 디렉터리",
)
@click.option("--apply/--no-apply", default=False, show_default=True, help="needs_review subset에 remediation 적용")
@click.option("--provider", default="openai", show_default=True, help="structured summary rebuild provider")
@click.option("--model", default="gpt-5.4", show_default=True, help="structured summary rebuild model")
@click.option(
    "--allow-external/--no-allow-external",
    default=True,
    show_default=True,
    help="canon remediation에서 외부 strong API 사용 허용 여부",
)
@click.option(
    "--llm-mode",
    default="strong",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="remediation에 사용할 LLM routing mode",
)
@click.option("--dry-run/--no-dry-run", default=False, show_default=True, help="audit + remediation 계획만 계산")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_canon_quality_audit(ctx, manifest_path, output_dir, apply, provider, model, allow_external, llm_mode, dry_run, as_json):
    """AI canon 9편 전용 deterministic card-quality audit + optional remediation"""
    khub = ctx.obj["khub"]
    payload = _audit_canon_quality(khub, manifest_path=manifest_path, output_dir=output_dir)
    remediation_items: list[dict[str, Any]] = []
    remediation_counts = {"ok": 0, "blocked": 0, "failed": 0, "noop": 0, "planned": 0}

    if apply:
        targets = [dict(item) for item in list(payload.get("items") or []) if bool(item.get("needsReview"))]
        for item in targets:
            result = _execute_canon_quality_remediation(
                khub=khub,
                paper_id=str(item.get("paperId") or ""),
                paper_title=str(item.get("title") or ""),
                issues=list(item.get("issues") or []),
                provider=str(provider or "openai"),
                model=str(model or "gpt-5.4"),
                allow_external=bool(allow_external),
                llm_mode=str(llm_mode or "strong"),
                dry_run=bool(dry_run),
            )
            remediation_items.append(result)
            status = str(result.get("status") or "noop")
            remediation_counts[status] = int(remediation_counts.get(status, 0)) + 1

        if not dry_run:
            payload = _audit_canon_quality(khub, manifest_path=manifest_path, output_dir=output_dir)

        remediation_by_paper = {
            str(item.get("paperId") or ""): item
            for item in remediation_items
            if str(item.get("paperId") or "").strip()
        }
        for item in list(payload.get("items") or []):
            remediation = remediation_by_paper.get(str(item.get("paperId") or ""))
            if remediation is not None:
                item["remediation"] = remediation

        payload["apply"] = True
        payload["dryRun"] = bool(dry_run)
        payload["remediation"] = {
            "targetCount": len(targets),
            "processedCount": len(remediation_items),
            "counts": remediation_counts,
            "items": remediation_items,
        }
        if remediation_counts.get("failed", 0):
            payload["status"] = "failed"
        elif remediation_counts.get("blocked", 0):
            payload["status"] = "partial"
    else:
        payload["apply"] = False
        payload["dryRun"] = bool(dry_run)

    payload.update(_write_canon_audit_outputs(payload, output_dir=output_dir))
    _validate_cli_payload(khub.config, payload, _CANON_QUALITY_AUDIT_SCHEMA)

    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    counts = dict(payload.get("counts") or {})
    console.print(
        f"[bold]AI canon quality audit[/bold] "
        f"targets={payload.get('targetCount')} needs_review={payload.get('needsReviewCount')} "
        f"ok={counts.get('ok', 0)} blocked={counts.get('blocked', 0)}"
    )
    console.print(f"[dim]report: {payload.get('reportPath')}[/dim]")
    console.print(f"[dim]selector: {payload.get('selectorPath')}[/dim]")
    if apply:
        remediation = dict(payload.get("remediation") or {})
        remediation_counts = dict(remediation.get("counts") or {})
        console.print(
            f"[dim]remediation processed={remediation.get('processedCount', 0)} "
            f"ok={remediation_counts.get('ok', 0)} blocked={remediation_counts.get('blocked', 0)} "
            f"failed={remediation_counts.get('failed', 0)}[/dim]"
        )
        if dry_run:
            console.print("[dim]dry-run only; no remediation executed[/dim]")
    for item in list(payload.get("items") or [])[:10]:
        console.print(
            f"- {item.get('paperId')} needs_review={bool(item.get('needsReview'))} "
            f"issues={','.join(item.get('issues') or []) or '-'}"
        )


@paper_group.command("repair-source")
@click.option("--paper-id", "paper_ids", multiple=True, help="대상 paper id (여러 번 사용 가능)")
@click.option(
    "--paper-id-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="newline-delimited paper id 파일",
)
@click.option(
    "--document-memory-parser",
    type=click.Choice(["raw", "pymupdf", "mineru", "opendataloader"]),
    default="raw",
    show_default=True,
    help="rebuild 시 사용할 paper document-memory parser",
)
@click.option("--rebuild/--no-rebuild", default=True, show_default=True, help="source repair 뒤 memory/card 재생성")
@click.option(
    "--allow-external/--no-allow-external",
    default=None,
    help="paper-memory 재생성 시 외부 API 사용 허용 여부",
)
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="paper-memory 재생성용 LLM 라우팅 모드",
)
@click.option("--dry-run/--no-dry-run", default=False, show_default=True, help="실행 없이 repair plan만 계산")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_repair_source(ctx, paper_ids, paper_id_file, document_memory_parser, rebuild, allow_external, llm_mode, dry_run, as_json):
    """known source contamination relink와 artifact rebuild를 정식 surface로 실행"""
    targets = _parse_cli_paper_ids(paper_ids=list(paper_ids), paper_id_file=paper_id_file)
    if not targets:
        raise click.ClickException("at least one --paper-id or --paper-id-file is required")
    khub = ctx.obj["khub"]
    payload = repair_paper_sources(
        sqlite_db=_sqlite_db(khub.config, khub=khub),
        config=khub.config,
        paper_ids=targets,
        document_memory_parser=str(document_memory_parser or "raw"),
        allow_external=allow_external,
        llm_mode=llm_mode,
        dry_run=bool(dry_run),
        rebuild=bool(rebuild),
    )
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    counts = dict(payload.get("counts") or {})
    console.print(
        f"[bold]paper source repair[/bold] status={payload.get('status')} "
        f"ok={counts.get('ok', 0)} blocked={counts.get('blocked', 0)} "
        f"failed={counts.get('failed', 0)} missing={counts.get('missing', 0)}"
    )
    if dry_run:
        console.print("[dim]dry-run only; no source relink or rebuild executed[/dim]")
    for item in list(payload.get("items") or [])[:10]:
        console.print(
            f"- {item.get('paperId')} repair={item.get('repairStatus')} "
            f"action={item.get('action')} reason={item.get('resolutionReason') or '-'}"
        )


@paper_group.command("repair-source-queue")
@click.option("--paper-id", "paper_ids", multiple=True, help="대상 paper id (여러 번 사용 가능)")
@click.option(
    "--paper-id-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="newline-delimited paper id 파일",
)
@click.option(
    "--document-memory-parser",
    type=click.Choice(["raw", "pymupdf", "mineru", "opendataloader"]),
    default="raw",
    show_default=True,
    help="safe runner가 사용할 paper document-memory parser",
)
@click.option("--rebuild/--no-rebuild", default=True, show_default=True, help="queue action이 memory/card rebuild까지 실행")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_repair_source_queue(ctx, paper_ids, paper_id_file, document_memory_parser, rebuild, as_json):
    """paper source repair를 ops action queue에 적재"""
    targets = _parse_cli_paper_ids(paper_ids=list(paper_ids), paper_id_file=paper_id_file)
    if not targets:
        raise click.ClickException("at least one --paper-id or --paper-id-file is required")
    khub = ctx.obj["khub"]
    payload = queue_paper_source_repairs(
        sqlite_db=_sqlite_db(khub.config, khub=khub),
        paper_ids=targets,
        document_memory_parser=str(document_memory_parser or "raw"),
        rebuild=bool(rebuild),
    )
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    counts = dict(payload.get("counts") or {})
    console.print(
        f"[bold]paper source repair queue[/bold] "
        f"created={counts.get('created', 0)} updated={counts.get('updated', 0)} "
        f"reopened={counts.get('reopened', 0)} missing={counts.get('missing', 0)}"
    )
    for item in list(payload.get("items") or [])[:10]:
        action = dict(item.get("action") or {})
        console.print(
            f"- {item.get('paperId')} op={item.get('operation')} "
            f"actionId={action.get('action_id') or '-'}"
        )


@paper_group.command("source-freshness", hidden=True)
@click.option("--paper-id", "paper_ids", multiple=True, help="대상 paper id (여러 번 사용 가능)")
@click.option(
    "--paper-id-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="newline-delimited paper id 파일",
)
@click.option("--limit", default=100, show_default=True, help="paper id 미지정 시 검사할 최대 논문 수")
@click.option("--sample-limit", default=10, show_default=True, help="JSON/text preview에 포함할 최대 문제 항목 수")
@click.option("--apply/--no-apply", default=False, show_default=True, help="stale 후보를 실제 stale로 표시")
@click.option("--json/--no-json", "as_json", default=False, show_default=True, help="결과를 JSON으로 출력")
@click.pass_context
def paper_source_freshness(ctx, paper_ids, paper_id_file, limit, sample_limit, apply, as_json):
    """paper source file hash와 document-memory hash를 비교하는 hidden audit."""
    khub = ctx.obj["khub"]
    targets = _parse_cli_paper_ids(paper_ids=list(paper_ids), paper_id_file=paper_id_file)
    payload = audit_paper_source_freshness(
        _sqlite_db(khub.config, khub=khub),
        paper_ids=targets,
        limit=max(1, int(limit)),
        sample_limit=max(0, int(sample_limit)),
        apply=bool(apply),
    )
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    counts = dict(payload.get("counts") or {})
    console.print(
        f"[bold]paper source freshness[/bold] scanned={payload.get('scannedCount', 0)} "
        f"fresh={counts.get('fresh', 0)} staleCandidates={counts.get('staleCandidate', 0)} "
        f"missingDocumentMemory={counts.get('missingDocumentMemory', 0)} "
        f"unableToHash={counts.get('unableToHash', 0)} markedStale={payload.get('markedStaleCount', 0)}"
    )
    if not apply:
        console.print("[dim]dry-run only; pass --apply to mark stale derivatives[/dim]")
    for item in list(payload.get("sampleItems") or [])[: max(0, int(sample_limit))]:
        console.print(
            f"- {item.get('paperId')} status={item.get('status')} "
            f"source={item.get('sourceKey') or '-'} path={item.get('sourcePath') or '-'}"
        )


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
    source_guard = stage_source_guard(
        sqlite_db,
        paper_id=str(paper_data["arxiv_id"] or ""),
        title=str(paper.title or ""),
    )
    if str(source_guard.get("status") or "") == "suspected" and not download:
        console.print(
            f"[yellow]source guard pending[/yellow]: {source_guard.get('reason', 'title collision candidates detected')}"
        )

    if download:
        downloader = PaperDownloader(config.papers_dir)
        with console.status("다운로드 중..."):
            result = downloader.download_single(paper.arxiv_id, paper.title)
        if result["success"]:
            review = review_downloaded_source(
                sqlite_db,
                paper_id=str(paper_data["arxiv_id"] or ""),
                title=str(paper.title or ""),
                pdf_path=str(result.get("pdf") or ""),
                text_path=str(result.get("text") or ""),
                existing=source_guard,
            )
            source_guard = dict(review.get("guard") or {})
            if bool(review.get("blocked")):
                sqlite_db.upsert_paper(paper_data)
                console.print(f"[yellow]source guard blocked import[/yellow]: {source_guard.get('reason')}")
                console.print(f"[yellow]논문 등록 보류: {paper_data['arxiv_id']}[/yellow]")
                return
            paper_data["pdf_path"] = str(review.get("finalPdfPath") or "") or None
            paper_data["text_path"] = str(review.get("finalTextPath") or "") or None
            if str(source_guard.get("decision") or "") == "relink_to_canonical":
                console.print(
                    f"  [yellow]source guard relink[/yellow]: {source_guard.get('canonicalPaperId') or '-'}"
                )
            else:
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
    type=click.Choice(["raw", "pymupdf", "mineru", "opendataloader"]),
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
        structured_summary_service_factory=StructuredPaperSummaryService,
        paper_summary_parser_fn=_paper_summary_parser,
        sync_structured_summary_view_fn=_sync_structured_summary_view,
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
    help="LLM 라우팅 모드",
)
@click.option("--paper-timeout-sec", default=300, show_default=True, help="paper별 요약 worker timeout (초)")
@click.option("--checkpoint-file", type=click.Path(dir_okay=False, path_type=Path), default=None, help="배치 진행 checkpoint JSONL 경로")
@click.pass_context
def paper_summarize_all(
    ctx,
    limit,
    field,
    quick,
    resummary,
    bad_only,
    threshold,
    provider,
    model,
    allow_external,
    llm_mode,
    paper_timeout_sec,
    checkpoint_file,
):
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
        allow_external=allow_external,
        llm_mode=llm_mode,
        paper_timeout_sec=paper_timeout_sec,
        checkpoint_file=checkpoint_file,
        console=console,
        sqlite_db_fn=_sqlite_db,
        assess_summary_quality_fn=_assess_summary_quality,
        structured_summary_service_factory=StructuredPaperSummaryService,
        paper_summary_parser_fn=_paper_summary_parser,
        render_structured_summary_notes_fn=_render_structured_summary_notes,
        sync_structured_summary_view_fn=_sync_structured_summary_view,
        build_public_summary_card_fn=_build_public_summary_card,
        summary_batch_worker_fn=_run_structured_summary_batch_worker,
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
@click.option("--provider", default=None, help="개념 정규화용 LLM provider override")
@click.option("--model", default=None, help="개념 정규화용 LLM model override")
@click.pass_context
def paper_normalize_concepts(ctx, dry_run, provider, model):
    """개념 동의어/복수형/약어 탐지 → 정규화 + 병합"""
    run_paper_normalize_concepts(
        khub=ctx.obj["khub"],
        dry_run=dry_run,
        provider=provider,
        model=model,
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

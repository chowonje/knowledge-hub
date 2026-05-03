"""Canonical crawl CLI registration surface."""

from __future__ import annotations

import subprocess
from copy import copy
from datetime import datetime, timezone
from typing import Any

import click
from rich.console import Console

from knowledge_hub.application.ko_note_reports import build_ko_note_report
from knowledge_hub.learning import LearningCoachService
from knowledge_hub.notes import KoNoteEnricher, KoNoteMaterializer
from knowledge_hub.web import WebIngestService
from knowledge_hub.interfaces.cli.commands.crawl_support import (
    build_continuous_latest_batch,
    build_reference_seed_batch,
    build_reindex_worker_cmd as _build_reindex_worker_cmd,
    collect_to_obsidian_payload as _collect_to_obsidian_payload,
    collect_urls as _collect_urls,
    load_normalized_records_for_job as _load_normalized_records_for_job,
    metadata_audit_payload as _metadata_audit_payload,
    parse_reindex_worker_output as _parse_reindex_worker_output,
    sqlite_db as _sqlite_db,
    sync_watchlist_payload as _sync_watchlist_payload,
    validate_cli_payload as _validate_cli_payload,
)

console = Console()


def _web_ingest_service(khub):
    if hasattr(khub, "web_ingest_service"):
        return khub.web_ingest_service()
    return WebIngestService(khub.config)


def _learning_service(khub):
    if hasattr(khub, "learning_service"):
        return khub.learning_service()
    return LearningCoachService(khub.config)


@click.group("crawl")
def crawl_group():
    """웹 문서를 크롤링하고 로컬 지식베이스에 적재"""


@click.group("crawl")
def labs_crawl_group():
    """고급 crawl / ko-note 운영 commands"""


@crawl_group.command("run")
@click.option("--url", "urls", multiple=True, help="수집할 URL (반복 사용 가능)")
@click.option("--url-file", default=None, help="URL 목록 파일(.txt)")
@click.option("--topic", default="", help="학습 주제 라벨")
@click.option("--source", default="web", show_default=True, help="소스 라벨")
@click.option("--profile", type=click.Choice(["safe", "balanced", "fast"]), default="safe", show_default=True)
@click.option("--source-policy", type=click.Choice(["fixed", "hybrid", "keyword"]), default="hybrid", show_default=True)
@click.option("--limit", type=int, default=0, show_default=True, help="처리 상한(0=전체)")
@click.option("--engine", type=click.Choice(["auto", "crawl4ai", "basic"]), default="auto", show_default=True)
@click.option("--timeout", type=int, default=15, show_default=True)
@click.option("--delay", type=float, default=0.5, show_default=True)
@click.option("--index/--no-index", default=True, show_default=True)
@click.option("--extract-concepts/--no-extract-concepts", default=True, show_default=True)
@click.option("--allow-external", is_flag=True, default=False, help="외부 LLM 보강 허용(sanitized only)")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_run(
    ctx,
    urls,
    url_file,
    topic,
    source,
    profile,
    source_policy,
    limit,
    engine,
    timeout,
    delay,
    index,
    extract_concepts,
    allow_external,
    as_json,
):
    """3계층(raw/normalized/indexed) 파이프라인 실행"""
    merged_urls = _collect_urls(urls, url_file)
    if not merged_urls:
        raise click.BadParameter("최소 1개 URL이 필요합니다 (--url 또는 --url-file)")

    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.run_pipeline(
        urls=merged_urls,
        topic=topic,
        source=source,
        profile=profile,
        source_policy=source_policy,
        limit=max(0, int(limit)),
        engine=engine,
        timeout=max(1, int(timeout)),
        delay=max(0.0, float(delay)),
        index=bool(index),
        extract_concepts=bool(extract_concepts),
        allow_external=bool(allow_external),
    )
    if as_json:
        console.print_json(data=payload)
        return

    console.print(
        f"[bold]crawl run[/bold] status={payload.get('status')} job={payload.get('jobId')} "
        f"requested={payload.get('requested')} processed={payload.get('processed')} "
        f"normalized={payload.get('normalized')} indexed={payload.get('indexed')} "
        f"pendingDomain={payload.get('pendingDomain')} failed={payload.get('failed')} skipped={payload.get('skipped')}"
    )
    console.print(
        f"profile={payload.get('profile')} sourcePolicy={payload.get('sourcePolicy')} "
        f"records/min={payload.get('recordsPerMin')} p50={payload.get('p50StepLatencyMs')}ms "
        f"dedupeRate={payload.get('dedupeRate')} retryRate={payload.get('retryRate')}"
    )
    for warning in payload.get("warnings", [])[:20]:
        console.print(f"[yellow]- {warning}[/yellow]")
    if int(payload.get("pendingDomain") or 0) > 0:
        console.print(
            "[dim]Domain pending: URLs did not reach the crawler — fix domain policy first, then "
            "`crawl resume --job-id` with the job id above.[/dim]"
        )


@crawl_group.command("collect")
@click.option("--url", "urls", multiple=True, help="수집할 URL (반복 사용 가능)")
@click.option("--url-file", default=None, help="URL 목록 파일(.txt)")
@click.option("--topic", default="", help="학습 주제 라벨")
@click.option("--source", default="web", show_default=True, help="소스 라벨")
@click.option("--profile", type=click.Choice(["safe", "balanced", "fast"]), default="safe", show_default=True)
@click.option("--source-policy", type=click.Choice(["fixed", "hybrid", "keyword"]), default="hybrid", show_default=True)
@click.option("--engine", type=click.Choice(["auto", "crawl4ai", "basic"]), default="auto", show_default=True)
@click.option("--timeout", type=int, default=15, show_default=True)
@click.option("--delay", type=float, default=0.5, show_default=True)
@click.option("--index/--no-index", default=True, show_default=True)
@click.option("--extract-concepts/--no-extract-concepts", default=True, show_default=True)
@click.option("--allow-external", is_flag=True, default=False, help="외부 LLM 보강 허용(sanitized only)")
@click.option("--max-source-notes", type=int, default=0, show_default=True)
@click.option("--max-concept-notes", type=int, default=0, show_default=True)
@click.option("--enrich/--no-enrich", default=True, show_default=True, help="생성 직후 enrichment pass 수행")
@click.option("--apply/--stage-only", "apply_notes", default=False, show_default=True, help="staged ko-note를 최종 Vault에 적용")
@click.option("--only-approved/--all-staged", default=False, show_default=True, help="--apply 사용 시 approved item만 반영")
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="LLM 라우팅 모드",
)
@click.option("--local-timeout-sec", type=int, default=0, show_default=True, help="로컬 LLM 타임아웃(초)")
@click.option(
    "--api-fallback-on-timeout/--no-api-fallback-on-timeout",
    default=True,
    show_default=True,
    help="로컬 타임아웃 시 API fallback 허용",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_collect(
    ctx,
    urls,
    url_file,
    topic,
    source,
    profile,
    source_policy,
    engine,
    timeout,
    delay,
    index,
    extract_concepts,
    allow_external,
    max_source_notes,
    max_concept_notes,
    enrich,
    apply_notes,
    only_approved,
    llm_mode,
    local_timeout_sec,
    api_fallback_on_timeout,
    as_json,
):
    """URL 목록을 수집하고 ko-note로 정리해 stage/apply까지 진행"""
    merged_urls = _collect_urls(urls, url_file)
    if not merged_urls:
        raise click.BadParameter("최소 1개 URL이 필요합니다 (--url 또는 --url-file)")

    khub = ctx.obj["khub"]
    payload = _collect_to_obsidian_payload(
        khub=khub,
        urls=merged_urls,
        topic=str(topic).strip(),
        source=str(source).strip(),
        profile=str(profile),
        source_policy=str(source_policy),
        engine=str(engine),
        timeout=max(1, int(timeout)),
        delay=max(0.0, float(delay)),
        index=bool(index),
        extract_concepts=bool(extract_concepts),
        allow_external=bool(allow_external),
        max_source_notes=max(0, int(max_source_notes)),
        max_concept_notes=max(0, int(max_concept_notes)),
        llm_mode=str(llm_mode),
        local_timeout_sec=max(0, int(local_timeout_sec)),
        api_fallback_on_timeout=bool(api_fallback_on_timeout),
        enrich=bool(enrich),
        apply_notes=bool(apply_notes),
        only_approved=bool(only_approved),
        web_ingest_factory=WebIngestService,
        materializer_factory=KoNoteMaterializer,
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.crawl.collect.result.v1")

    if as_json:
        console.print_json(data=payload)
        return

    crawl_payload = payload.get("crawl") or {}
    materialize_payload = payload.get("materialize") or {}
    apply_payload = payload.get("apply") or {}
    console.print(
        f"[bold]crawl collect[/bold] status={payload.get('status')} job={payload.get('jobId') or '-'} "
        f"run={payload.get('runId') or '-'} requested={payload.get('requested')}"
    )
    console.print(
        f"crawl indexed={crawl_payload.get('indexed', 0)} normalized={crawl_payload.get('normalized', 0)} "
        f"failed={crawl_payload.get('failed', 0)} pendingDomain={crawl_payload.get('pendingDomain', 0)}"
    )
    console.print(
        f"ko-note source={materialize_payload.get('sourceGenerated', 0)}/{materialize_payload.get('sourceCandidates', 0)} "
        f"concept={materialize_payload.get('conceptGenerated', 0)}/{materialize_payload.get('conceptCandidates', 0)}"
    )
    if apply_notes:
        console.print(
            f"apply applied={apply_payload.get('applied', 0)} skipped={apply_payload.get('skipped', 0)} "
            f"conflicts={apply_payload.get('conflicts', 0)} onlyApproved={payload.get('onlyApproved')}"
        )
    else:
        console.print("apply skipped (stage-only)")
    for warning in payload.get("warnings", [])[:20]:
        console.print(f"[yellow]- {warning}[/yellow]")
    if int(crawl_payload.get("pendingDomain") or 0) > 0:
        console.print(
            "[dim]Domain pending: some URLs never reached the crawler — resolve domain policy, then "
            "`crawl resume --job-id` if you used crawl run.[/dim]"
        )


@crawl_group.command("resume")
@click.option("--job-id", required=True, help="재개할 파이프라인 job ID")
@click.option("--profile", type=click.Choice(["safe", "balanced", "fast"]), default=None)
@click.option("--source-policy", type=click.Choice(["fixed", "hybrid", "keyword"]), default=None)
@click.option("--limit", type=int, default=0, show_default=True, help="추가 처리 상한(0=제한 없음)")
@click.option("--engine", type=click.Choice(["auto", "crawl4ai", "basic"]), default="auto", show_default=True)
@click.option("--timeout", type=int, default=15, show_default=True)
@click.option("--delay", type=float, default=0.5, show_default=True)
@click.option("--index/--no-index", default=True, show_default=True)
@click.option("--extract-concepts/--no-extract-concepts", default=True, show_default=True)
@click.option("--allow-external", is_flag=True, default=False)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_resume(
    ctx,
    job_id,
    profile,
    source_policy,
    limit,
    engine,
    timeout,
    delay,
    index,
    extract_concepts,
    allow_external,
    as_json,
):
    """중단된 파이프라인 재개(run_id/job 상태 유지)"""
    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.resume_pipeline(
        job_id=str(job_id).strip(),
        profile=profile,
        source_policy=source_policy,
        limit=max(0, int(limit)),
        engine=engine,
        timeout=max(1, int(timeout)),
        delay=max(0.0, float(delay)),
        index=bool(index),
        extract_concepts=bool(extract_concepts),
        allow_external=bool(allow_external),
    )
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]crawl resume[/bold] status={payload.get('status')} job={payload.get('jobId')} "
        f"processed={payload.get('processed')} indexed={payload.get('indexed')} failed={payload.get('failed')}"
    )


@crawl_group.command("status")
@click.option("--job-id", required=True, help="조회할 파이프라인 job ID")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_status(ctx, job_id, as_json):
    """파이프라인 상태/체크포인트/메트릭 조회"""
    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.pipeline_status(str(job_id).strip())
    if as_json:
        console.print_json(data=payload)
        return
    if payload.get("status") != "ok":
        raise click.ClickException(str(payload.get("error") or "pipeline status failed"))
    counts = payload.get("counts") or {}
    console.print(
        f"[bold]crawl status[/bold] job={payload.get('jobId')} status={payload.get('jobStatus')} "
        f"total={counts.get('total', 0)} indexed={counts.get('indexed', 0)} "
        f"pendingDomain={counts.get('pending_domain', 0)} failed={counts.get('failed', 0)} "
        f"skipped={counts.get('skipped', 0)}"
    )


@crawl_group.command("latest-build")
@click.option(
    "--watchlist-file",
    default="data/curation/ai_watchlists/continuous_sources.yaml",
    show_default=True,
    help="continuous watchlist YAML 경로",
)
@click.option("--output-prefix", default="", help="출력 파일 prefix (비우면 날짜 자동)")
@click.option("--per-source-limit", type=int, default=4, show_default=True, help="소스별 최신 글 상한")
@click.option("--include-existing", is_flag=True, default=False, help="이미 수집된 URL도 포함")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_latest_build(ctx, watchlist_file, output_prefix, per_source_limit, include_existing, as_json):
    """continuous source watchlist에서 최신 글 batch를 생성"""
    khub = ctx.obj["khub"]
    payload = build_continuous_latest_batch(
        config=khub.config,
        watchlist_path=watchlist_file,
        output_prefix=str(output_prefix or "").strip(),
        per_source_limit=max(1, int(per_source_limit)),
        include_existing=bool(include_existing),
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ai-watchlist-batch.v2")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]crawl latest-build[/bold] count={payload.get('count')} "
        f"txt={payload.get('txtPath')} yaml={payload.get('yamlPath')}"
    )
    by_source = payload.get("bySource") or {}
    for source_name, count in sorted(by_source.items()):
        console.print(f"- {source_name}: {count}")
    for failed in payload.get("failedSources", [])[:10]:
        console.print(f"[yellow]- failed {failed.get('source_name')}: {failed.get('error')}[/yellow]")
    for warning in payload.get("warnings", [])[:20]:
        console.print(f"[yellow]- {warning}[/yellow]")


@crawl_group.command("continuous-sync")
@click.option(
    "--watchlist-file",
    default="data/curation/ai_watchlists/continuous_sources.yaml",
    show_default=True,
    help="continuous watchlist YAML 경로",
)
@click.option("--output-prefix", default="", help="출력 파일 prefix (비우면 날짜 자동)")
@click.option("--per-source-limit", type=int, default=4, show_default=True, help="소스별 최신 글 상한")
@click.option("--include-existing", is_flag=True, default=False, help="이미 수집된 URL도 포함")
@click.option("--topic", default="continuous-latest", show_default=True, help="crawl topic 라벨")
@click.option("--source", default="web", show_default=True, help="crawl source 라벨")
@click.option("--profile", type=click.Choice(["safe", "balanced", "fast"]), default="safe", show_default=True)
@click.option("--source-policy", type=click.Choice(["fixed", "hybrid", "keyword"]), default="fixed", show_default=True)
@click.option("--engine", type=click.Choice(["auto", "crawl4ai", "basic"]), default="auto", show_default=True)
@click.option("--timeout", type=int, default=15, show_default=True)
@click.option("--delay", type=float, default=0.5, show_default=True)
@click.option("--index/--no-index", default=True, show_default=True)
@click.option("--extract-concepts/--no-extract-concepts", default=True, show_default=True)
@click.option("--materialize/--no-materialize", default=True, show_default=True, help="ko-note 생성까지 연속 실행")
@click.option("--apply/--no-apply", "apply_notes", default=True, show_default=True, help="생성된 ko-note를 최종 Vault에 반영")
@click.option("--max-source-notes", type=int, default=20, show_default=True)
@click.option("--max-concept-notes", type=int, default=10, show_default=True)
@click.option("--allow-external", is_flag=True, default=False, help="crawl/ko-note 외부 LLM 보강 허용")
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="ko-note LLM 라우팅 모드",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_continuous_sync(
    ctx,
    watchlist_file,
    output_prefix,
    per_source_limit,
    include_existing,
    topic,
    source,
    profile,
    source_policy,
    engine,
    timeout,
    delay,
    index,
    extract_concepts,
    materialize,
    apply_notes,
    max_source_notes,
    max_concept_notes,
    allow_external,
    llm_mode,
    as_json,
):
    """continuous watchlist를 build->crawl->ko-note로 연속 실행"""
    khub = ctx.obj["khub"]
    build_payload = build_continuous_latest_batch(
        config=khub.config,
        watchlist_path=watchlist_file,
        output_prefix=str(output_prefix or "").strip(),
        per_source_limit=max(1, int(per_source_limit)),
        include_existing=bool(include_existing),
    )
    _validate_cli_payload(khub.config, build_payload, "knowledge-hub.ai-watchlist-batch.v2")
    payload = _sync_watchlist_payload(
        khub=khub,
        build_payload=build_payload,
        topic=topic,
        source=source,
        profile=profile,
        source_policy=source_policy,
        engine=engine,
        timeout=timeout,
        delay=delay,
        index=index,
        extract_concepts=extract_concepts,
        materialize=materialize,
        apply_notes=apply_notes,
        max_source_notes=max_source_notes,
        max_concept_notes=max_concept_notes,
        allow_external=allow_external,
        llm_mode=llm_mode,
        web_ingest_factory=WebIngestService,
        materializer_factory=KoNoteMaterializer,
    )
    if as_json:
        console.print_json(data=payload)
        return
    if payload.get("status") == "skipped":
        console.print("[yellow]continuous-sync skipped: no fresh urls discovered[/yellow]")
        return

    crawl_payload = payload.get("crawl") or {}
    materialize_payload = payload.get("materialize") or {}
    apply_payload = payload.get("apply") or {}

    console.print(
        f"[bold]continuous-sync[/bold] discovered={build_payload.get('count', 0)} "
        f"normalized={crawl_payload.get('normalized', 0)} indexed={crawl_payload.get('indexed', 0)}"
    )
    if materialize_payload:
        console.print(
            f"ko-note source={materialize_payload.get('sourceGenerated', 0)}/{materialize_payload.get('sourceCandidates', 0)} "
            f"concept={materialize_payload.get('conceptGenerated', 0)}/{materialize_payload.get('conceptCandidates', 0)}"
        )
    if apply_payload:
        console.print(
            f"apply applied={apply_payload.get('applied', 0)} skipped={apply_payload.get('skipped', 0)}"
        )
    for warning in list(build_payload.get("warnings", []))[:10]:
        console.print(f"[yellow]- build: {warning}[/yellow]")
    for warning in list(crawl_payload.get("warnings", []))[:10]:
        console.print(f"[yellow]- crawl: {warning}[/yellow]")
    for warning in list(materialize_payload.get("warnings", []))[:10]:
        console.print(f"[yellow]- ko-note: {warning}[/yellow]")


@crawl_group.command("reference-build")
@click.option(
    "--watchlist-file",
    default="data/curation/ai_watchlists/reference_sources.yaml",
    show_default=True,
    help="reference watchlist YAML 경로",
)
@click.option("--output-prefix", default="", help="출력 파일 prefix (비우면 날짜 자동)")
@click.option("--include-existing", is_flag=True, default=False, help="이미 수집된 URL도 포함")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_reference_build(ctx, watchlist_file, output_prefix, include_existing, as_json):
    """specialist reference watchlist에서 정적 reference batch를 생성"""
    khub = ctx.obj["khub"]
    payload = build_reference_seed_batch(
        config=khub.config,
        watchlist_path=watchlist_file,
        output_prefix=str(output_prefix or "").strip(),
        include_existing=bool(include_existing),
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ai-watchlist-batch.v2")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]crawl reference-build[/bold] count={payload.get('count')} "
        f"txt={payload.get('txtPath')} yaml={payload.get('yamlPath')}"
    )
    by_source = payload.get("bySource") or {}
    for source_name, count in sorted(by_source.items()):
        console.print(f"- {source_name}: {count}")
    for warning in payload.get("warnings", [])[:20]:
        console.print(f"[yellow]- {warning}[/yellow]")


@crawl_group.command("reference-sync")
@click.option(
    "--watchlist-file",
    default="data/curation/ai_watchlists/reference_sources.yaml",
    show_default=True,
    help="reference watchlist YAML 경로",
)
@click.option("--output-prefix", default="", help="출력 파일 prefix (비우면 날짜 자동)")
@click.option("--include-existing", is_flag=True, default=False, help="이미 수집된 URL도 포함")
@click.option("--topic", default="concept-reference", show_default=True, help="crawl topic 라벨")
@click.option("--source", default="web", show_default=True, help="crawl source 라벨")
@click.option("--profile", type=click.Choice(["safe", "balanced", "fast"]), default="safe", show_default=True)
@click.option("--source-policy", type=click.Choice(["fixed", "hybrid", "keyword"]), default="fixed", show_default=True)
@click.option("--engine", type=click.Choice(["auto", "crawl4ai", "basic"]), default="auto", show_default=True)
@click.option("--timeout", type=int, default=15, show_default=True)
@click.option("--delay", type=float, default=0.5, show_default=True)
@click.option("--index/--no-index", default=True, show_default=True)
@click.option("--extract-concepts/--no-extract-concepts", default=True, show_default=True)
@click.option("--materialize/--no-materialize", default=True, show_default=True, help="ko-note 생성까지 연속 실행")
@click.option("--apply/--no-apply", "apply_notes", default=False, show_default=True, help="생성된 ko-note를 최종 Vault에 반영")
@click.option("--max-source-notes", type=int, default=20, show_default=True)
@click.option("--max-concept-notes", type=int, default=10, show_default=True)
@click.option("--allow-external", is_flag=True, default=False, help="crawl/ko-note 외부 LLM 보강 허용")
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="ko-note LLM 라우팅 모드",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_reference_sync(
    ctx,
    watchlist_file,
    output_prefix,
    include_existing,
    topic,
    source,
    profile,
    source_policy,
    engine,
    timeout,
    delay,
    index,
    extract_concepts,
    materialize,
    apply_notes,
    max_source_notes,
    max_concept_notes,
    allow_external,
    llm_mode,
    as_json,
):
    """reference watchlist를 build->crawl->ko-note로 연속 실행"""
    khub = ctx.obj["khub"]
    build_payload = build_reference_seed_batch(
        config=khub.config,
        watchlist_path=watchlist_file,
        output_prefix=str(output_prefix or "").strip(),
        include_existing=bool(include_existing),
    )
    _validate_cli_payload(khub.config, build_payload, "knowledge-hub.ai-watchlist-batch.v2")
    payload = _sync_watchlist_payload(
        khub=khub,
        build_payload=build_payload,
        topic=topic,
        source=source,
        profile=profile,
        source_policy=source_policy,
        engine=engine,
        timeout=timeout,
        delay=delay,
        index=index,
        extract_concepts=extract_concepts,
        materialize=materialize,
        apply_notes=apply_notes,
        max_source_notes=max_source_notes,
        max_concept_notes=max_concept_notes,
        allow_external=allow_external,
        llm_mode=llm_mode,
        web_ingest_factory=WebIngestService,
        materializer_factory=KoNoteMaterializer,
    )
    if as_json:
        console.print_json(data=payload)
        return
    if payload.get("status") == "skipped":
        console.print("[yellow]reference-sync skipped: no fresh urls discovered[/yellow]")
        return

    crawl_payload = payload.get("crawl") or {}
    materialize_payload = payload.get("materialize") or {}
    apply_payload = payload.get("apply") or {}
    console.print(
        f"[bold]reference-sync[/bold] discovered={build_payload.get('count', 0)} "
        f"normalized={crawl_payload.get('normalized', 0)} indexed={crawl_payload.get('indexed', 0)}"
    )
    if materialize_payload:
        console.print(
            f"ko-note source={materialize_payload.get('sourceGenerated', 0)}/{materialize_payload.get('sourceCandidates', 0)} "
            f"concept={materialize_payload.get('conceptGenerated', 0)}/{materialize_payload.get('conceptCandidates', 0)}"
        )
    if apply_payload:
        console.print(
            f"apply applied={apply_payload.get('applied', 0)} skipped={apply_payload.get('skipped', 0)}"
        )
    for warning in list(build_payload.get("warnings", []))[:10]:
        console.print(f"[yellow]- build: {warning}[/yellow]")
    for warning in list(crawl_payload.get("warnings", []))[:10]:
        console.print(f"[yellow]- crawl: {warning}[/yellow]")
    for warning in list(materialize_payload.get("warnings", []))[:10]:
        console.print(f"[yellow]- ko-note: {warning}[/yellow]")


@crawl_group.command("metadata-audit")
@click.option("--job-id", default="", help="crawl job id (비우면 최신 job)")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_metadata_audit(ctx, job_id, as_json):
    """normalized payload 기준 메타데이터 품질 감사"""
    khub = ctx.obj["khub"]
    from knowledge_hub.infrastructure.persistence import SQLiteDatabase

    db = _sqlite_db(khub)
    try:
        effective_job_id = str(job_id or "").strip()
        if not effective_job_id:
            latest = db.get_latest_crawl_pipeline_job()
            if not latest:
                raise click.ClickException("no crawl pipeline jobs found")
            effective_job_id = str(latest.get("job_id") or "").strip()
    finally:
        db.close()

    rows = _load_normalized_records_for_job(
        khub.config,
        effective_job_id,
        sqlite_db_factory=lambda: _sqlite_db(khub),
    )
    payload = _metadata_audit_payload(job_id=effective_job_id, rows=rows)
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]crawl metadata-audit[/bold] job={payload.get('jobId')} "
        f"records={payload.get('records')} avgCompleteness={payload.get('avgCompleteness')}"
    )
    for source_name, stats in sorted((payload.get("bySource") or {}).items()):
        console.print(
            f"- {source_name}: count={stats.get('count')} "
            f"published={stats.get('publishedRatio')} author={stats.get('authorRatio')} "
            f"tags={stats.get('tagRatio')} avg={stats.get('avgCompleteness')}"
        )
    for item in payload.get("topFlags", [])[:10]:
        console.print(f"[yellow]- flag {item.get('flag')}: {item.get('count')}[/yellow]")


@crawl_group.command("ko-note-generate")
@click.option("--job-id", default="", help="대상 crawl job ID")
@click.option("--latest-job", is_flag=True, default=False, help="가장 최근 crawl job 사용")
@click.option("--max-source-notes", type=int, default=0, show_default=True)
@click.option("--max-concept-notes", type=int, default=0, show_default=True)
@click.option("--allow-external", is_flag=True, default=False, help="외부 fallback 허용(sanitized only)")
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="LLM 라우팅 모드",
)
@click.option("--local-timeout-sec", type=int, default=0, show_default=True, help="로컬 LLM 타임아웃(초)")
@click.option(
    "--api-fallback-on-timeout/--no-api-fallback-on-timeout",
    default=True,
    show_default=True,
    help="로컬 타임아웃 시 API fallback 허용",
)
@click.option("--enrich/--no-enrich", default=True, show_default=True, help="생성 직후 enrichment pass 수행")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ko_note_generate(
    ctx,
    job_id,
    latest_job,
    max_source_notes,
    max_concept_notes,
    allow_external,
    llm_mode,
    local_timeout_sec,
    api_fallback_on_timeout,
    enrich,
    as_json,
):
    """crawl 결과를 한국어 Obsidian 스테이징 노트로 생성"""
    khub = ctx.obj["khub"]
    db_job_id = str(job_id or "").strip()
    if latest_job:
        from knowledge_hub.infrastructure.persistence import SQLiteDatabase

        db = _sqlite_db(khub)
        latest = db.get_latest_crawl_pipeline_job()
        db.close()
        db_job_id = str((latest or {}).get("job_id") or "").strip()
    if not db_job_id:
        raise click.BadParameter("--job-id 또는 --latest-job 중 하나가 필요합니다.")

    materializer = KoNoteMaterializer(khub.config)
    payload = materializer.generate_for_job(
        job_id=db_job_id,
        max_source_notes=int(max_source_notes) if int(max_source_notes) > 0 else None,
        max_concept_notes=int(max_concept_notes) if int(max_concept_notes) > 0 else None,
        allow_external=bool(allow_external),
        llm_mode=str(llm_mode),
        local_timeout_sec=int(local_timeout_sec) if int(local_timeout_sec) > 0 else None,
        api_fallback_on_timeout=bool(api_fallback_on_timeout),
        enrich=bool(enrich),
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ko-note.generate.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]ko-note-generate[/bold] status={payload.get('status')} run={payload.get('runId')} "
        f"job={payload.get('crawlJobId')} source={payload.get('sourceGenerated')}/{payload.get('sourceCandidates')} "
        f"concept={payload.get('conceptGenerated')}/{payload.get('conceptCandidates')}"
    )
    for warning in payload.get("warnings", [])[:20]:
        console.print(f"[yellow]- {warning}[/yellow]")


@crawl_group.command("ko-note-enrich")
@click.option(
    "--scope",
    type=click.Choice(["new", "existing-top", "both"]),
    default="both",
    show_default=True,
    help="enrichment 대상 범위",
)
@click.option("--run-id", default="", help="신규 staging 대상 run ID(new/both에서 사용)")
@click.option("--item-type", type=click.Choice(["source", "concept", "all"]), default="all", show_default=True)
@click.option("--limit-source", type=int, default=120, show_default=True)
@click.option("--limit-concept", type=int, default=80, show_default=True)
@click.option("--allow-external/--no-allow-external", default=True, show_default=True)
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="LLM 라우팅 모드",
)
@click.option("--local-timeout-sec", type=int, default=0, show_default=True)
@click.option(
    "--api-fallback-on-timeout/--no-api-fallback-on-timeout",
    default=True,
    show_default=True,
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ko_note_enrich(
    ctx,
    scope,
    run_id,
    item_type,
    limit_source,
    limit_concept,
    allow_external,
    llm_mode,
    local_timeout_sec,
    api_fallback_on_timeout,
    as_json,
):
    """기존/new ko note를 내용 중심으로 재가공"""
    khub = ctx.obj["khub"]
    enricher = KoNoteEnricher(khub.config)
    payload = enricher.enrich(
        scope=str(scope),
        run_id=str(run_id).strip(),
        item_type=str(item_type),
        limit_source=max(1, int(limit_source)),
        limit_concept=max(1, int(limit_concept)),
        allow_external=bool(allow_external),
        llm_mode=str(llm_mode),
        local_timeout_sec=int(local_timeout_sec) if int(local_timeout_sec) > 0 else None,
        api_fallback_on_timeout=bool(api_fallback_on_timeout),
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ko-note.enrich.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]ko-note-enrich[/bold] status={payload.get('status')} run={payload.get('runId')} "
        f"scope={payload.get('scope')} source={payload.get('sourceEnriched')}/{payload.get('sourceTargets')} "
        f"concept={payload.get('conceptEnriched')}/{payload.get('conceptTargets')} skipped={payload.get('skipped')}"
    )
    for warning in payload.get("warnings", [])[:20]:
        console.print(f"[yellow]- {warning}[/yellow]")


@crawl_group.command("ko-note-remediate")
@click.option("--run-id", required=True, help="remediation 대상 ko note run ID")
@click.option("--item-type", type=click.Choice(["source", "concept", "all"]), default="all", show_default=True)
@click.option(
    "--quality-flag",
    type=click.Choice(["needs_review", "reject", "unscored", "all"]),
    default="all",
    show_default=True,
)
@click.option("--item-id", type=int, default=0, show_default=True)
@click.option("--limit", type=int, default=50, show_default=True)
@click.option("--strategy", type=click.Choice(["section", "full"]), default="section", show_default=True)
@click.option("--allow-external/--no-allow-external", default=True, show_default=True)
@click.option(
    "--llm-mode",
    default="auto",
    show_default=True,
    type=click.Choice(["fallback-only", "local", "mini", "strong", "auto"]),
    help="LLM 라우팅 모드",
)
@click.option("--local-timeout-sec", type=int, default=0, show_default=True)
@click.option(
    "--api-fallback-on-timeout/--no-api-fallback-on-timeout",
    default=True,
    show_default=True,
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ko_note_remediate(
    ctx,
    run_id,
    item_type,
    quality_flag,
    item_id,
    limit,
    strategy,
    allow_external,
    llm_mode,
    local_timeout_sec,
    api_fallback_on_timeout,
    as_json,
):
    """review queue item을 section/full 전략으로 자동 보강"""
    khub = ctx.obj["khub"]
    enricher = KoNoteEnricher(khub.config)
    payload = enricher.remediate(
        run_id=str(run_id).strip(),
        item_type=str(item_type),
        quality_flag=str(quality_flag),
        item_id=int(item_id),
        limit=max(1, int(limit)),
        strategy=str(strategy),
        allow_external=bool(allow_external),
        llm_mode=str(llm_mode),
        local_timeout_sec=int(local_timeout_sec) if int(local_timeout_sec) > 0 else None,
        api_fallback_on_timeout=bool(api_fallback_on_timeout),
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ko-note.remediate.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]ko-note-remediate[/bold] run={payload.get('runId')} "
        f"strategy={payload.get('strategy')} "
        f"attempted={payload.get('attempted')} remediated={payload.get('remediated')} "
        f"improved={payload.get('improved')} unchanged={payload.get('unchanged')} failed={payload.get('failed')}"
    )
    for item in payload.get("items", [])[:20]:
        console.print(
            f"- id={item.get('id')} type={item.get('itemType')} "
            f"{item.get('beforeQualityFlag')}->{item.get('afterQualityFlag')} improved={item.get('improved')} "
            f"strategy={item.get('strategy')} target={len(item.get('targetSections') or [])} patched={len(item.get('patchedSections') or [])}"
        )
    for warning in payload.get("warnings", [])[:20]:
        console.print(f"[yellow]- {warning}[/yellow]")


@crawl_group.command("ko-note-enrich-status")
@click.option("--run-id", required=True, help="조회할 enrichment run ID")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ko_note_enrich_status(ctx, run_id, as_json):
    """ko note enrichment 상태 조회"""
    khub = ctx.obj["khub"]
    enricher = KoNoteEnricher(khub.config)
    payload = enricher.status(run_id=str(run_id).strip())
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ko-note.enrich.status.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    counts = payload.get("counts") or {}
    console.print(
        f"[bold]ko-note-enrich-status[/bold] run={payload.get('runId')} status={payload.get('status')} "
        f"source={counts.get('sourceEnriched', 0)}/{counts.get('sourceTargets', 0)} "
        f"concept={counts.get('conceptEnriched', 0)}/{counts.get('conceptTargets', 0)} "
        f"total={counts.get('total', 0)}"
    )
    for warning in payload.get("warnings", [])[:20]:
        console.print(f"[yellow]- {warning}[/yellow]")


@crawl_group.command("ko-note-status")
@click.option("--run-id", required=True, help="조회할 ko note run ID")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ko_note_status(ctx, run_id, as_json):
    """ko note staging 상태 조회"""
    khub = ctx.obj["khub"]
    materializer = KoNoteMaterializer(khub.config)
    payload = materializer.status(run_id=str(run_id).strip())
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ko-note.status.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    counts = payload.get("counts") or {}
    paths = payload.get("paths") or {}
    console.print(
        f"[bold]ko-note-status[/bold] run={payload.get('runId')} status={payload.get('status')} "
        f"staged={counts.get('staged', 0)} approved={counts.get('approved', 0)} "
        f"applied={counts.get('applied', 0)} rejected={counts.get('rejected', 0)} "
        f"source={counts.get('source', 0)} concept={counts.get('concept', 0)}"
    )
    if paths.get("stagingRoot"):
        console.print(f"staging: {paths.get('stagingRoot')}")


@crawl_group.command("ko-note-report")
@click.option("--run-id", required=True, help="조회할 ko note run ID")
@click.option("--recent-runs", type=int, default=10, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ko_note_report(ctx, run_id, recent_runs, as_json):
    """ko note 운영 리포트 조회"""
    khub = ctx.obj["khub"]
    materializer = KoNoteMaterializer(khub.config)
    payload = build_ko_note_report(
        materializer.sqlite_db,
        run_id=str(run_id).strip(),
        recent_runs=max(1, int(recent_runs)),
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ko-note.report.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    run_payload = payload.get("run") or {}
    counts = run_payload.get("counts") or {}
    approvals = run_payload.get("autoApproved") or {}
    review_queue = (run_payload.get("reviewQueue") or {}).get("combined") or {}
    console.print(
        f"[bold]ko-note-report[/bold] run={payload.get('runId')} status={payload.get('status')} "
        f"staged={counts.get('staged', 0)} approved={counts.get('approved', 0)} "
        f"applied={counts.get('applied', 0)} rejected={counts.get('rejected', 0)} "
        f"autoApproved={approvals.get('total', 0)} reviewQueued={review_queue.get('total', 0)}"
    )
    for alert in list(payload.get("alerts") or [])[:5]:
        console.print(f"[yellow]! {alert.get('severity')} {alert.get('code')}: {alert.get('summary')}[/yellow]")
    for action in list(payload.get("recommendedActions") or [])[:3]:
        command = " ".join([str(action.get("command") or ""), *[str(item) for item in (action.get("args") or [])]]).strip()
        console.print(f"[cyan]> {action.get('summary')}[/cyan]")
        if command:
            console.print(f"[dim]  {command}[/dim]")
    for warning in list(payload.get("warnings") or [])[:10]:
        console.print(f"[yellow]- {warning}[/yellow]")


@crawl_group.command("ko-note-review-list")
@click.option("--run-id", required=True, help="조회할 ko note run ID")
@click.option("--item-type", type=click.Choice(["source", "concept", "all"]), default="all", show_default=True)
@click.option(
    "--quality-flag",
    type=click.Choice(["needs_review", "reject", "unscored", "all"]),
    default="all",
    show_default=True,
)
@click.option("--limit", type=int, default=50, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ko_note_review_list(ctx, run_id, item_type, quality_flag, limit, as_json):
    """review queue에 올라간 ko note 목록 조회"""
    khub = ctx.obj["khub"]
    materializer = KoNoteMaterializer(khub.config)
    payload = materializer.review_list(
        run_id=str(run_id).strip(),
        item_type=str(item_type),
        quality_flag=str(quality_flag),
        limit=max(1, int(limit)),
    )
    if as_json:
        console.print_json(data=payload)
        return
    counts = payload.get("counts") or {}
    console.print(
        f"[bold]ko-note-review-list[/bold] run={payload.get('runId')} "
        f"total={counts.get('total', 0)} source={counts.get('source', 0)} concept={counts.get('concept', 0)} "
        f"needs_review={counts.get('needs_review', 0)} reject={counts.get('reject', 0)}"
    )
    for item in payload.get("items", [])[:20]:
        console.print(
            f"- id={item.get('id')} type={item.get('itemType')} status={item.get('status')} "
            f"quality={item.get('qualityFlag')} title={item.get('titleKo') or item.get('titleEn')}"
        )
        for reason in list(item.get("reviewReasons") or [])[:2]:
            console.print(f"  [yellow]reason[/yellow] {reason}")
        for hint in list(item.get("reviewPatchHints") or [])[:2]:
            console.print(f"  [cyan]hint[/cyan] {hint}")


@crawl_group.command("ko-note-review-approve")
@click.option("--item-id", required=True, type=int, help="승인할 ko note item ID")
@click.option("--reviewer", default="cli-user", show_default=True)
@click.option("--note", default="", help="선택적 review 메모")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ko_note_review_approve(ctx, item_id, reviewer, note, as_json):
    """review queue item을 승인 처리"""
    khub = ctx.obj["khub"]
    materializer = KoNoteMaterializer(khub.config)
    payload = materializer.review_approve(item_id=int(item_id), reviewer=str(reviewer), note=str(note))
    if as_json:
        console.print_json(data=payload)
        return
    if payload.get("status") != "ok":
        raise click.ClickException("; ".join(payload.get("warnings") or ["review approve failed"]))
    console.print(
        f"[green]ko-note-review-approve[/green] item={payload.get('itemId')} "
        f"type={payload.get('itemType')} quality={payload.get('qualityFlag')}"
    )


@crawl_group.command("ko-note-review-reject")
@click.option("--item-id", required=True, type=int, help="거절할 ko note item ID")
@click.option("--reviewer", default="cli-user", show_default=True)
@click.option("--note", default="", help="선택적 review 메모")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ko_note_review_reject(ctx, item_id, reviewer, note, as_json):
    """review queue item을 거절 처리"""
    khub = ctx.obj["khub"]
    materializer = KoNoteMaterializer(khub.config)
    payload = materializer.review_reject(item_id=int(item_id), reviewer=str(reviewer), note=str(note))
    if as_json:
        console.print_json(data=payload)
        return
    if payload.get("status") != "ok":
        raise click.ClickException("; ".join(payload.get("warnings") or ["review reject failed"]))
    console.print(
        f"[yellow]ko-note-review-reject[/yellow] item={payload.get('itemId')} "
        f"type={payload.get('itemType')} quality={payload.get('qualityFlag')}"
    )


@crawl_group.command("ko-note-apply")
@click.option("--run-id", required=True, help="적용할 ko note run ID")
@click.option("--item-type", type=click.Choice(["source", "concept", "all"]), default="all", show_default=True)
@click.option("--limit", type=int, default=0, show_default=True)
@click.option("--only-approved/--all-staged", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ko_note_apply(ctx, run_id, item_type, limit, only_approved, as_json):
    """staging ko note를 최종 Vault로 승격"""
    khub = ctx.obj["khub"]
    materializer = KoNoteMaterializer(khub.config)
    payload = materializer.apply(
        run_id=str(run_id).strip(),
        item_type=str(item_type),
        limit=max(0, int(limit)),
        only_approved=bool(only_approved),
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.ko-note.apply.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]ko-note-apply[/bold] run={payload.get('runId')} applied={payload.get('applied')} "
        f"skipped={payload.get('skipped')} conflicts={payload.get('conflicts')}"
    )
    for warning in payload.get("warnings", [])[:20]:
        console.print(f"[yellow]- {warning}[/yellow]")


@crawl_group.command("ko-note-reject")
@click.option("--run-id", required=True, help="거절할 ko note run ID")
@click.option("--item-type", type=click.Choice(["source", "concept", "all"]), default="all", show_default=True)
@click.option("--limit", type=int, default=0, show_default=True)
@click.pass_context
def crawl_ko_note_reject(ctx, run_id, item_type, limit):
    """staging ko note를 reject 처리"""
    khub = ctx.obj["khub"]
    materializer = KoNoteMaterializer(khub.config)
    payload = materializer.reject(
        run_id=str(run_id).strip(),
        item_type=str(item_type),
        limit=max(0, int(limit)),
    )
    console.print(
        f"[bold]ko-note-reject[/bold] run={payload.get('runId')} "
        f"rejected={payload.get('rejected')} itemType={payload.get('itemType')}"
    )


@crawl_group.command("benchmark")
@click.option("--url", "urls", multiple=True, help="벤치마크 대상 URL (반복 사용 가능)")
@click.option("--url-file", default=None, help="URL 목록 파일(.txt)")
@click.option("--sample", type=int, default=20, show_default=True, help="샘플 수")
@click.option("--topic", default="", help="학습 주제 라벨")
@click.option("--profile", type=click.Choice(["safe", "balanced", "fast"]), default="safe", show_default=True)
@click.option("--source-policy", type=click.Choice(["fixed", "hybrid", "keyword"]), default="hybrid", show_default=True)
@click.option("--engine", type=click.Choice(["auto", "crawl4ai", "basic"]), default="auto", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_benchmark(ctx, urls, url_file, sample, topic, profile, source_policy, engine, as_json):
    """성능 스모크 벤치마크(records/min, p50 latency, retry/dedupe율)"""
    merged_urls = _collect_urls(urls, url_file)
    if not merged_urls:
        raise click.BadParameter("최소 1개 URL이 필요합니다 (--url 또는 --url-file)")

    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.benchmark_pipeline(
        urls=merged_urls,
        sample=max(1, int(sample)),
        profile=profile,
        source_policy=source_policy,
        topic=topic,
        engine=engine,
    )
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]crawl benchmark[/bold] status={payload.get('status')} sample={payload.get('sample')} "
        f"records/min={payload.get('recordsPerMin')} p50={payload.get('p50StepLatencyMs')}ms "
        f"retryRate={payload.get('retryRate')} dedupeRate={payload.get('dedupeRate')} "
        f"memoryPeak={payload.get('memoryPeakRatio')}"
    )


@crawl_group.group("domain-policy")
def crawl_domain_policy_group():
    """hybrid 소스 정책용 도메인 승인 큐"""


@crawl_domain_policy_group.command("list")
@click.option("--status", default="", type=click.Choice(["", "approved", "pending", "rejected"]))
@click.option("--limit", type=int, default=200, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_domain_policy_list(ctx, status, limit, as_json):
    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.list_domain_policy(status=status, limit=max(1, int(limit)))
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]domain-policy[/bold] count={payload.get('count', 0)} status={status or 'all'}")
    for item in payload.get("items", [])[:limit]:
        console.print(
            f"- {item.get('domain')} status={item.get('status')} updated={item.get('updated_at')} reason={item.get('reason', '')}"
        )


@crawl_domain_policy_group.command("approve")
@click.option("--domain", required=True, help="승인할 도메인")
@click.option("--reason", default="", help="승인 사유")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_domain_policy_approve(ctx, domain, reason, as_json):
    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.apply_domain_policy(domain=domain, reason=reason)
    if as_json:
        console.print_json(data=payload)
        return
    item = payload.get("item") or {}
    console.print(f"[green]approved[/green] {item.get('domain')} ({item.get('status')})")


@crawl_domain_policy_group.command("reject")
@click.option("--domain", required=True, help="거절할 도메인")
@click.option("--reason", default="", help="거절 사유")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_domain_policy_reject(ctx, domain, reason, as_json):
    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.reject_domain_policy(domain=domain, reason=reason)
    if as_json:
        console.print_json(data=payload)
        return
    item = payload.get("item") or {}
    console.print(f"[yellow]rejected[/yellow] {item.get('domain')} ({item.get('status')})")


@crawl_group.command("ingest")
@click.option("--url", "urls", multiple=True, help="수집할 URL (반복 사용 가능)")
@click.option("--url-file", default=None, help="URL 목록 파일(.txt)")
@click.option("--topic", default="", help="학습 주제 라벨 (선택)")
@click.option("--engine", type=click.Choice(["auto", "crawl4ai", "basic"]), default="auto", show_default=True)
@click.option("--timeout", type=int, default=15, show_default=True)
@click.option("--delay", type=float, default=0.5, show_default=True)
@click.option("--index/--no-index", default=True, show_default=True, help="벡터 인덱싱 여부")
@click.option("--save-raw/--no-save-raw", default=False, show_default=True, help="원본 HTML(raw) 저장")
@click.option("--raw-dir", default=None, help="원본 HTML/raw 메타 저장 경로 (미지정 시 sqlite 부모/web_raw)")
@click.option("--extract-concepts/--no-extract-concepts", default=True, show_default=True, help="수집 직후 웹 온톨로지 자동 추출")
@click.option("--allow-external", is_flag=True, default=False, help="외부 LLM 보강 단계 허용 (sanitized payload only)")
@click.option("--writeback/--no-writeback", default=False, show_default=True, help="LearningHub 03/04 파일 갱신")
@click.option("--concept-threshold", type=float, default=0.78, show_default=True, help="개념 즉시 반영 임계값")
@click.option("--relation-threshold", type=float, default=0.75, show_default=True, help="관계 즉시 반영 임계값")
@click.option("--quality-first/--no-quality-first", default=False, show_default=True, help="품질 승인 문서만 온톨로지/임베딩에 반영")
@click.option("--quality-threshold", type=float, default=0.62, show_default=True, help="문서 품질 점수 승인 임계값")
@click.option("--quality-min-tokens", type=int, default=80, show_default=True, help="품질 승인 최소 토큰 수")
@click.option("--quality-sample-size", type=int, default=12, show_default=True, help="샘플 게이트 평가 문서 수")
@click.option("--quality-sample-min-pass-rate", type=float, default=0.70, show_default=True, help="샘플 게이트 최소 통과율")
@click.option("--incremental/--no-incremental", default=True, show_default=True, help="동일 content hash는 재처리 생략")
@click.option("--emit-ontology-graph/--no-emit-ontology-graph", default=False, show_default=True, help="온톨로지 그래프 TTL 생성")
@click.option("--ontology-ttl-path", default=None, help="RDF/TTL 출력 경로(미지정 시 자동 경로)")
@click.option("--validate-ontology-graph", is_flag=True, default=False, help="pySHACL 유효성 검사 수행 (옵션)")
@click.option("--learn-map", is_flag=True, default=False, help="수집 후 학습 trunk map 갱신")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_ingest(
    ctx,
    urls,
    url_file,
    topic,
    engine,
    timeout,
    delay,
    index,
    save_raw,
    raw_dir,
    extract_concepts,
    allow_external,
    writeback,
    concept_threshold,
    relation_threshold,
    quality_first,
    quality_threshold,
    quality_min_tokens,
    quality_sample_size,
    quality_sample_min_pass_rate,
    incremental,
    emit_ontology_graph,
    ontology_ttl_path,
    validate_ontology_graph,
    learn_map,
    as_json,
):
    """crawl4ai(또는 기본 크롤러)로 웹 콘텐츠를 수집/저장/인덱싱"""
    merged_urls = _collect_urls(urls, url_file)
    if not merged_urls:
        raise click.BadParameter("최소 1개 URL이 필요합니다 (--url 또는 --url-file)")

    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.crawl_and_ingest(
        urls=merged_urls,
        topic=topic,
        engine=engine,
        timeout=max(1, timeout),
        delay=max(0.0, delay),
        index=index,
        save_raw=save_raw,
        raw_dir=raw_dir,
        extract_concepts=extract_concepts,
        allow_external=allow_external,
        writeback=writeback,
        concept_threshold=max(0.0, min(1.0, concept_threshold)),
        relation_threshold=max(0.0, min(1.0, relation_threshold)),
        quality_first=quality_first,
        quality_threshold=max(0.0, min(1.0, quality_threshold)),
        quality_min_tokens=max(10, int(quality_min_tokens)),
        quality_sample_size=max(1, int(quality_sample_size)),
        quality_sample_min_pass_rate=max(0.0, min(1.0, quality_sample_min_pass_rate)),
        incremental=incremental,
        emit_ontology_graph=emit_ontology_graph,
        ontology_ttl_path=ontology_ttl_path,
        validate_ontology_graph=validate_ontology_graph,
    )

    if learn_map and topic:
        learn_svc = _learning_service(khub)
        map_result = learn_svc.map(
            topic=topic,
            source="all",
            days=180,
            top_k=12,
            writeback=False,
            allow_external=False,
        )
        payload["learningMap"] = {
            "status": map_result.get("status"),
            "trunkCount": len(map_result.get("trunks") or []),
            "branchCount": len(map_result.get("branches") or []),
            "schema": map_result.get("schema"),
        }

    if as_json:
        console.print_json(data=payload)
        return

    console.print(
        f"[bold]web ingest[/bold] requested={payload.get('requested')} crawled={payload.get('crawled')} "
        f"stored={payload.get('stored')} rawStored={payload.get('rawStored', 0)} "
        f"indexedChunks={payload.get('indexedChunks')} unchanged={payload.get('unchanged', 0)}"
    )
    console.print(
        f"engine: {payload.get('engine')} | topic: {payload.get('topic') or '-'}"
        f" | runId: {payload.get('runId')}"
    )

    ontology = payload.get("ontology") or {}
    ontology_graph = payload.get("ontologyGraph") or {}
    quality = payload.get("quality") or {}
    console.print(
        "ontology: "
        f"conceptsAccepted={ontology.get('conceptsAccepted', 0)} "
        f"relationsAccepted={ontology.get('relationsAccepted', 0)} "
        f"pending={ontology.get('pendingCount', 0)} "
        f"aliasesAdded={ontology.get('aliasesAdded', 0)}"
    )
    if quality:
        sample_gate = quality.get("sampleGate") if isinstance(quality.get("sampleGate"), dict) else {}
        console.print(
            "quality: "
            f"approved={quality.get('approvedCount', 0)} "
            f"rejected={quality.get('rejectedCount', 0)} "
            f"duplicates={quality.get('duplicateCount', 0)} "
            f"gateAllowed={quality.get('gateAllowed', True)} "
            f"samplePassRate={sample_gate.get('passRate', '-')}"
        )
    if ontology_graph:
        console.print(
            f"[cyan]ontology graph:[/cyan] status={ontology_graph.get('status')} "
            f"concepts={ontology_graph.get('conceptCount', 0)} "
            f"relations={ontology_graph.get('relationCount', 0)} "
            f"ttl={ontology_graph.get('turtlePath') or '-'}"
        )

    warnings = payload.get("warnings") or []
    for warning in warnings:
        console.print(f"[yellow]- {warning}[/yellow]")

    failed = payload.get("failed") or []
    if failed:
        console.print("[red]failed:[/red]")
        for item in failed[:20]:
            console.print(f"- {item.get('url')}: {item.get('error')}")

    if payload.get("learningMap"):
        lm = payload["learningMap"]
        console.print(
            "[cyan]learning map:[/cyan] "
            f"status={lm.get('status')} trunks={lm.get('trunkCount')} branches={lm.get('branchCount')}"
        )

    raw_dir_value = str(payload.get("rawDir") or "").strip()
    if raw_dir_value:
        console.print(f"[cyan]raw archive:[/cyan] {raw_dir_value}")

    writeback_paths = payload.get("writebackPaths") or []
    if writeback_paths:
        console.print("[cyan]writeback:[/cyan]")
        for path in writeback_paths:
            console.print(f"- {path}")


@labs_crawl_group.command("youtube-ingest")
@click.option("--url", "urls", multiple=True, help="수집할 YouTube URL (반복 사용 가능)")
@click.option("--url-file", default=None, help="YouTube URL 목록 파일(.txt)")
@click.option("--topic", default="", help="학습 주제 라벨 (선택)")
@click.option("--timeout", type=int, default=30, show_default=True)
@click.option("--delay", type=float, default=0.0, show_default=True)
@click.option("--index/--no-index", default=True, show_default=True, help="벡터 인덱싱 여부")
@click.option("--extract-concepts/--no-extract-concepts", default=True, show_default=True, help="수집 직후 웹 온톨로지 자동 추출")
@click.option("--allow-external", is_flag=True, default=False, help="외부 LLM 보강 단계 허용 (sanitized payload only)")
@click.option("--writeback/--no-writeback", default=False, show_default=True, help="LearningHub 03/04 파일 갱신")
@click.option("--transcript-language", default=None, help="선호 자막/전사 언어 코드 (예: ko, en)")
@click.option("--asr-model", default="tiny", show_default=True, help="local Whisper fallback model")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_youtube_ingest(
    ctx,
    urls,
    url_file,
    topic,
    timeout,
    delay,
    index,
    extract_concepts,
    allow_external,
    writeback,
    transcript_language,
    asr_model,
    as_json,
):
    """YouTube URL을 caption-first/local-ASR-fallback으로 수집/저장/인덱싱"""
    merged_urls = _collect_urls(urls, url_file)
    if not merged_urls:
        raise click.BadParameter("최소 1개 URL이 필요합니다 (--url 또는 --url-file)")

    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.crawl_and_ingest(
        urls=merged_urls,
        topic=topic,
        engine="youtube",
        timeout=max(5, timeout),
        delay=max(0.0, delay),
        index=index,
        extract_concepts=extract_concepts,
        allow_external=allow_external,
        writeback=writeback,
        input_source="youtube",
        transcript_language=str(transcript_language or "").strip() or None,
        asr_model=str(asr_model or "tiny").strip() or "tiny",
        index_autofix_mode="youtube_single_retry",
    )

    if as_json:
        console.print_json(data=payload)
        return

    console.print(
        f"[bold]youtube ingest[/bold] requested={payload.get('requested')} crawled={payload.get('crawled')} "
        f"stored={payload.get('stored')} indexedChunks={payload.get('indexedChunks')}"
    )
    console.print(f"engine: {payload.get('engine')} | topic: {payload.get('topic') or '-'} | runId: {payload.get('runId')}")
    index_diagnostics = payload.get("indexDiagnostics") if isinstance(payload.get("indexDiagnostics"), dict) else {}
    if index_diagnostics:
        console.print(
            "index: "
            f"{index_diagnostics.get('status', '')} "
            f"initial={index_diagnostics.get('initialIndexedChunks', 0)} "
            f"final={index_diagnostics.get('finalIndexedChunks', 0)}"
        )
    for warning in payload.get("warnings") or []:
        console.print(f"[yellow]- {warning}[/yellow]")
    for item in payload.get("failed") or []:
        console.print(f"[red]- {item.get('url')}: {item.get('error')}[/red]")


@crawl_group.group("pending")
def crawl_pending_group():
    """웹 온톨로지 pending 큐 조회/승인/거절"""


@crawl_pending_group.command("list")
@click.option("--topic", default="", help="주제 필터")
@click.option("--limit", type=int, default=50, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_pending_list(ctx, topic, limit, as_json):
    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.list_pending(topic=topic, limit=max(1, limit))
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]pending[/bold] count={payload.get('count', 0)} topic={topic or '-'}")
    for item in payload.get("items", [])[:limit]:
        reason = item.get("reason_json") if isinstance(item.get("reason_json"), dict) else {}
        kind = reason.get("kind", "relation")
        console.print(
            f"- id={item.get('id')} kind={kind} relation={item.get('relation_norm')} "
            f"conf={float(item.get('confidence') or 0):.3f} status={item.get('status')}"
        )


@crawl_pending_group.command("apply")
@click.option("--id", "pending_id", type=int, required=True, help="pending 항목 ID")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_pending_apply(ctx, pending_id, as_json):
    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.apply_pending(pending_id=int(pending_id))
    if as_json:
        console.print_json(data=payload)
        return
    if payload.get("status") != "ok":
        raise click.ClickException(str(payload.get("error") or "pending apply failed"))
    item = payload.get("item") or {}
    console.print(f"[green]applied[/green] id={item.get('id')} status={item.get('status')}")


@crawl_pending_group.command("reject")
@click.option("--id", "pending_id", type=int, required=True, help="pending 항목 ID")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_pending_reject(ctx, pending_id, as_json):
    khub = ctx.obj["khub"]
    service = _web_ingest_service(khub)
    payload = service.reject_pending(pending_id=int(pending_id))
    if as_json:
        console.print_json(data=payload)
        return
    if payload.get("status") != "ok":
        raise click.ClickException(str(payload.get("error") or "pending reject failed"))
    item = payload.get("item") or {}
    console.print(f"[yellow]rejected[/yellow] id={item.get('id')} status={item.get('status')}")


@crawl_group.command("reindex-approved")
@click.option("--topic", default="", help="주제 필터(선택)")
@click.option("--limit", type=int, default=0, show_default=True, help="처리 상한(0=전체)")
@click.option("--workers", type=int, default=1, show_default=True, help="병렬 워커 수(현재 각 워커는 별도 프로세스)")
@click.option("--include-unrated", is_flag=True, default=False, help="quality 메타가 없는 기존 노트도 포함")
@click.option(
    "--shard-index",
    type=int,
    default=0,
    show_default=True,
    help="수동 샤드 인덱스(개별 실행)",
)
@click.option(
    "--shard-total",
    type=int,
    default=1,
    show_default=True,
    help="수동 샤드 수(개별 실행)",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def crawl_reindex_approved(
    ctx,
    topic,
    limit,
    workers,
    include_unrated,
    shard_index,
    shard_total,
    as_json,
):
    """품질 승인된 웹 문서만 다시 임베딩."""
    khub = ctx.obj["khub"]
    workers_n = max(1, int(workers))
    shard_index_n = max(0, int(shard_index))
    shard_total_n = max(1, int(shard_total))
    topic_safe = str(topic or "").strip()

    if workers_n > 1 and (shard_index_n != 0 or shard_total_n != 1):
        raise click.BadParameter(
            "--workers 모드에서는 --shard-index/--shard-total를 직접 지정할 수 없습니다."
        )

    service = _web_ingest_service(khub)

    if workers_n == 1:
        payload = service.reindex_approved(
            topic=topic_safe,
            limit=max(0, int(limit)),
            include_unrated=bool(include_unrated),
            shard_index=shard_index_n,
            shard_total=shard_total_n,
        )
    else:
        worker_results: list[dict[str, Any]] = []
        worker_diagnostics: list[str] = []
        processes: list[tuple[int, list[str], subprocess.Popen]] = []
        cmd_base_limit = max(0, int(limit))

        for i in range(workers_n):
            cmd = _build_reindex_worker_cmd(
                base_ctx=ctx,
                worker_index=i,
                worker_total=workers_n,
                topic=topic_safe,
                limit=cmd_base_limit,
                include_unrated=bool(include_unrated),
            )
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append((i, cmd, process))

        for i, cmd, process in processes:
            out, err = process.communicate()
            completed = subprocess.CompletedProcess(cmd, process.returncode or 0, out, err)
            shard_payload, diag = _parse_reindex_worker_output(
                worker_index=i,
                worker_total=workers_n,
                topic=topic_safe,
                result=completed,
            )
            worker_results.append(shard_payload)
            if diag:
                worker_diagnostics.append(f"worker[{i}] {diag}")

        status = "ok"
        failed = []
        for item in worker_results:
            failed.extend(item.get("failed") or [])
            if item.get("status") != "ok":
                status = "partial"

        payload = {
            "schema": "knowledge-hub.crawl.reindex-approved.result.v1",
            "status": status,
            "topic": topic_safe,
            "scanned": sum(int(item.get("scanned", 0)) for item in worker_results),
            "selected": sum(int(item.get("selected", 0)) for item in worker_results),
            "indexedChunks": sum(int(item.get("indexedChunks", 0)) for item in worker_results),
            "includeUnrated": bool(include_unrated),
            "shardIndex": 0,
            "shardTotal": workers_n,
            "workers": workers_n,
            "failed": failed,
            "workerResults": worker_results,
            "workerDiagnostics": worker_diagnostics,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    if as_json:
        console.print_json(data=payload)
        return
    if workers_n > 1:
        console.print(
            f"[bold]reindex-approved[/bold] status={payload.get('status')} workers={workers_n} "
            f"scanned={payload.get('scanned')} selected={payload.get('selected')} "
            f"indexedChunks={payload.get('indexedChunks')}"
        )
        for item in payload.get("workerResults", []):
            console.print(
                f"  - worker={item.get('shardIndex')}/{payload.get('shardTotal')} "
                f"status={item.get('status')} scanned={item.get('scanned')} "
                f"selected={item.get('selected')} indexed={item.get('indexedChunks')}"
            )
        for diagnostic in payload.get("workerDiagnostics", []):
            console.print(f"[yellow]  ! {diagnostic}[/yellow]")
        return

    console.print(
        f"[bold]reindex-approved[/bold] status={payload.get('status')} "
        f"scanned={payload.get('scanned')} selected={payload.get('selected')} "
        f"indexedChunks={payload.get('indexedChunks')}"
    )
    for item in payload.get("failed", [])[:10]:
        console.print(f"[red]- {item.get('url')}: {item.get('error')}[/red]")


DEFAULT_CRAWL_COMMAND_NAMES = {
    "ingest",
    "ko-note-apply",
    "ko-note-generate",
    "ko-note-status",
    "resume",
    "run",
    "status",
}

LABS_CRAWL_COMMAND_NAMES = {
    "benchmark",
    "continuous-sync",
    "domain-policy",
    "ko-note-enrich",
    "ko-note-enrich-status",
    "ko-note-report",
    "ko-note-reject",
    "ko-note-remediate",
    "ko-note-review-list",
    "ko-note-review-approve",
    "ko-note-review-reject",
    "latest-build",
    "metadata-audit",
    "pending",
    "reference-build",
    "reference-sync",
    "reindex-approved",
}

for _command_name in sorted(LABS_CRAWL_COMMAND_NAMES):
    _command = crawl_group.commands.pop(_command_name, None)
    if _command is not None:
        _legacy_alias = copy(_command)
        _legacy_alias.hidden = True
        crawl_group.add_command(_legacy_alias, _command_name)
        labs_crawl_group.add_command(_command, _command_name)

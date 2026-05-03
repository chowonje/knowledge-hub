"""
khub health - 진단/트러블슈팅 명령어
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knowledge_hub.infrastructure.config import Config, mask_secret, resolve_api_key
from knowledge_hub.application.paper_reports import build_paper_source_ops_report
from knowledge_hub.providers import registry

console = Console()


def _sqlite_db(khub_ctx):
    if hasattr(khub_ctx, "sqlite_db"):
        return khub_ctx.sqlite_db()
    from knowledge_hub.infrastructure.persistence import SQLiteDatabase

    return SQLiteDatabase(khub_ctx.config.sqlite_path)


def _vector_db(khub_ctx):
    if hasattr(khub_ctx, "vector_db"):
        return khub_ctx.vector_db()
    from knowledge_hub.infrastructure.persistence import VectorDatabase

    config = khub_ctx.config
    return VectorDatabase(config.vector_db_path, config.collection_name)


def _check_api_key_status(config: Config, provider: str, selected_model: str) -> tuple[str, str, str]:
    """(상태, 출처, 상세) 반환"""
    _ = selected_model
    config_value = ""
    raw_providers = config._data.get("providers", {}) if hasattr(config, "_data") else {}
    raw_provider_cfg = raw_providers.get(provider, {}) if isinstance(raw_providers, dict) else {}
    if isinstance(raw_provider_cfg, dict):
        config_value = str(raw_provider_cfg.get("api_key", ""))
    config_value = config_value.strip()

    info = registry.get_provider_info(provider)
    if not (info and info.requires_api_key):
        return "OK", "불필요", "-"

    if config_value.startswith("${") and config_value.endswith("}"):
        env_key = config_value[2:-1]
        actual = resolve_api_key(provider, config_value)
        if actual:
            return "OK", f"env:{env_key}", mask_secret(actual, 6)
        return "WARN", f"env:{env_key}", "환경변수 미설정"

    actual = resolve_api_key(provider, config_value)
    if actual:
        return "OK", "직접입력", mask_secret(actual, 4)

    return "WARN", "미설정", "번역/요약/임베딩 실행 시 실패 가능"


def _count_non_empty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _event_integrity_rows(config: Config) -> tuple[list[tuple[str, str, str]], bool]:
    rows: list[tuple[str, str, str]] = []
    ok = True
    try:
        from knowledge_hub.infrastructure.persistence import SQLiteDatabase

        sqlite_db = SQLiteDatabase(config.sqlite_path)
        try:
            jsonl_path = Path(config.sqlite_path).parent / "ontology_events.jsonl"
            jsonl_count = _count_non_empty_lines(jsonl_path)
            sql_count_row = sqlite_db.conn.execute("SELECT COUNT(*) AS cnt FROM ontology_events").fetchone()
            sql_count = int(sql_count_row["cnt"]) if sql_count_row else 0
            count_status = "OK" if jsonl_count == sql_count else "WARN"
            if count_status != "OK":
                ok = False
            rows.append(("이벤트 로그 수 (JSONL vs SQLite)", count_status, f"{jsonl_count} vs {sql_count}"))

            snapshot = (
                sqlite_db.event_store.snapshot_at(datetime.now(timezone.utc).isoformat())
                if sqlite_db.event_store
                else {"entity_count": 0}
            )
            snapshot_entity_count = int(snapshot.get("entity_count", 0))
            db_entity_row = sqlite_db.conn.execute("SELECT COUNT(*) AS cnt FROM ontology_entities").fetchone()
            db_entity_count = int(db_entity_row["cnt"]) if db_entity_row else 0
            entity_status = "OK" if snapshot_entity_count == db_entity_count else "WARN"
            if entity_status != "OK":
                ok = False
            rows.append(
                (
                    "엔티티 수 (snapshot vs ontology_entities)",
                    entity_status,
                    f"{snapshot_entity_count} vs {db_entity_count}",
                )
            )

            orphan_count_row = sqlite_db.conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM ontology_entities e
                LEFT JOIN ontology_events oe ON oe.entity_id = e.entity_id
                WHERE oe.entity_id IS NULL
                """
            ).fetchone()
            orphan_count = int(orphan_count_row["cnt"]) if orphan_count_row else 0
            orphan_rows = sqlite_db.conn.execute(
                """
                SELECT e.entity_id
                FROM ontology_entities e
                LEFT JOIN ontology_events oe ON oe.entity_id = e.entity_id
                WHERE oe.entity_id IS NULL
                ORDER BY e.entity_id
                LIMIT 5
                """
            ).fetchall()
            orphan_ids = [str(row["entity_id"]) for row in orphan_rows if row and row["entity_id"]]
            orphan_status = "OK" if orphan_count == 0 else "WARN"
            if orphan_status != "OK":
                ok = False
            orphan_message = str(orphan_count)
            if orphan_ids:
                orphan_message += f" sample={', '.join(orphan_ids)}"
            rows.append(("이벤트 없이 남은 엔티티", orphan_status, orphan_message))
        finally:
            sqlite_db.close()
    except Exception as error:
        rows.append(("이벤트 정합성", "WARN", str(error)))
        ok = False
    return rows, ok


def _pipeline_integrity_rows(config: Config) -> tuple[list[tuple[str, str, str]], bool]:
    rows: list[tuple[str, str, str]] = []
    ok = True
    try:
        from knowledge_hub.infrastructure.persistence import SQLiteDatabase

        sqlite_db = SQLiteDatabase(config.sqlite_path)
        try:
            latest = sqlite_db.get_latest_crawl_pipeline_job()
            if not latest:
                rows.append(("최근 파이프라인 잡", "OK", "실행 이력이 없습니다"))
                return rows, True

            job_id = str(latest.get("job_id", ""))
            counts = sqlite_db.count_crawl_pipeline_records(job_id)
            indexed_rows = sqlite_db.list_crawl_pipeline_records(job_id, state="indexed", limit=200000)
            storage_root_token = str(latest.get("storage_root", "") or "").strip()
            if not storage_root_token:
                rows.append(("스토리지 루트", "WARN", f"job={job_id} storage_root 누락"))
                ok = False
            else:
                storage_root = Path(storage_root_token)
                if storage_root.exists():
                    rows.append(("스토리지 루트", "OK", f"job={job_id} root={storage_root}"))
                    indexed_manifest_count = 0
                    missing_manifest = 0
                    for item in indexed_rows:
                        manifest = str(item.get("indexed_path", "") or "").strip()
                        if manifest and Path(manifest).exists():
                            indexed_manifest_count += 1
                        else:
                            missing_manifest += 1

                    status = "OK" if int(counts.get("indexed", 0)) == indexed_manifest_count else "WARN"
                    if status != "OK":
                        ok = False
                    rows.append(
                        (
                            "indexed 레코드 vs indexed manifest",
                            status,
                            f"{int(counts.get('indexed', 0))} vs {indexed_manifest_count} (missing={missing_manifest})",
                        )
                    )
                else:
                    rows.append(
                        (
                            "스토리지 루트",
                            "SKIP",
                            f"job={job_id} root unavailable: {storage_root}",
                        )
                    )
                    rows.append(
                        (
                            "indexed 레코드 vs indexed manifest",
                            "SKIP",
                            f"storage_root unavailable: {storage_root}",
                        )
                    )

            total = int(counts.get("total", 0))
            failed = int(counts.get("failed", 0))
            failed_ratio = (failed / total) if total else 0.0
            threshold = float(config.get_nested("pipeline", "health", "failed_ratio_threshold", default=0.4) or 0.4)
            ratio_status = "OK" if failed_ratio <= threshold else "WARN"
            if ratio_status != "OK":
                ok = False
            rows.append(("실패율 임계치", ratio_status, f"{failed_ratio:.3f} <= {threshold:.3f}"))

            checkpoints = sqlite_db.list_crawl_pipeline_checkpoints(job_id)
            inconsistent = 0
            for checkpoint in checkpoints:
                last_record_id = str(checkpoint.get("last_record_id", "") or "").strip()
                if not last_record_id:
                    continue
                record = sqlite_db.get_crawl_pipeline_record(job_id, last_record_id)
                if not record:
                    inconsistent += 1
            checkpoint_status = "OK" if inconsistent == 0 else "WARN"
            if checkpoint_status != "OK":
                ok = False
            rows.append(
                (
                    "체크포인트 정합성",
                    checkpoint_status,
                    f"job={job_id} checkpoints={len(checkpoints)} inconsistent={inconsistent}",
                )
            )
        finally:
            sqlite_db.close()
    except Exception as error:
        rows.append(("파이프라인 정합성", "WARN", str(error)))
        ok = False
    return rows, ok


def _paper_source_integrity_rows(config: Config) -> tuple[list[tuple[str, str, str]], bool]:
    rows: list[tuple[str, str, str]] = []
    ok = True
    try:
        from knowledge_hub.infrastructure.persistence import SQLiteDatabase

        sqlite_db = SQLiteDatabase(config.sqlite_path)
        try:
            report = build_paper_source_ops_report(sqlite_db, limit=20)
            counts = dict(report.get("counts") or {})
            known_rule_count = int(counts.get("knownRuleCount") or 0)
            present = int(counts.get("presentInStore") or counts.get("tracked") or 0)
            missing = int(counts.get("missingFromStore") or counts.get("missingKnownIds") or 0)
            pending = int(counts.get("repairablePending") or counts.get("repairEligible") or 0)
            blocked_manual = int(counts.get("blockedManual") or counts.get("manualFixRequired") or 0)
            blocked_missing_canonical = int(counts.get("blockedMissingCanonical") or counts.get("canonicalMissing") or 0)
            reviewed_keep = int(counts.get("keepCurrentReviewed") or counts.get("keepCurrent") or 0)
            aligned = int(counts.get("alreadyAligned") or 0)

            rows.append(("known cleanup rules", "OK", f"{known_rule_count} tracked rules"))
            rows.append(("known-rule rows in store", "OK", f"present={present} missing={missing}"))

            pending_status = "OK" if pending == 0 else "WARN"
            if pending_status != "OK":
                ok = False
            rows.append(("pending canonical relinks", pending_status, str(pending)))

            manual_status = "OK" if blocked_manual == 0 else "WARN"
            if manual_status != "OK":
                ok = False
            rows.append(("manual source fixes required", manual_status, str(blocked_manual)))

            canonical_status = "OK" if blocked_missing_canonical == 0 else "WARN"
            if canonical_status != "OK":
                ok = False
            rows.append(("missing canonical rows", canonical_status, str(blocked_missing_canonical)))

            rows.append(("reviewed keep-current rows", "OK", str(reviewed_keep)))
            rows.append(("already aligned aliases", "OK", str(aligned)))
        finally:
            sqlite_db.close()
    except Exception as error:
        rows.append(("paper source integrity", "WARN", str(error)))
        ok = False
    return rows, ok


def run_health(khub_ctx, check_events: bool = False, check_pipeline: bool = False, check_paper_sources: bool = False):
    config = khub_ctx.config
    event_ok = True
    pipeline_ok = True
    paper_source_ok = True

    lines = [
        "[bold]Knowledge Hub Health Check[/bold]",
        f"Config: {config.config_path or '(기본값)'}",
    ]
    console.print(Panel("\n".join(lines), title="요약", border_style="blue"))

    table = Table(title="환경/폴더")
    table.add_column("항목", style="cyan")
    table.add_column("상태", justify="center")
    table.add_column("설명")

    checks = []

    config_dir = Path(config.config_path).parent if config.config_path else Path.home() / ".khub"
    if config_dir.exists() or (config.config_path is None or Path(config_dir).exists()):
        checks.append(("설정 저장 경로", "OK", str(config_dir)))
    else:
        checks.append(("설정 저장 경로", "WARN", f"경로 없음: {config_dir}"))

    for label, path in [
        ("논문 저장 디렉토리", config.papers_dir),
        ("벡터DB 경로", config.vector_db_path),
        ("SQLite 경로", config.sqlite_path),
    ]:
        p = Path(path)
        parent = p.parent
        if parent.exists():
            checks.append((label, "OK", str(path)))
        else:
            checks.append((label, "WARN", f"부모 디렉토리 없음: {parent}"))

    for item in checks:
        table.add_row(*item)
    console.print(table)

    providers = registry.list_providers()
    provider_table = Table(title="프로바이더/키 상태")
    provider_table.add_column("타입", style="cyan")
    provider_table.add_column("프로바이더", style="magenta")
    provider_table.add_column("모델")
    provider_table.add_column("설치", justify="center")
    provider_table.add_column("API 키", justify="center")
    provider_table.add_column("키 출처", style="green")
    provider_table.add_column("상세", max_width=50)

    roles = [
        ("번역", config.translation_provider, config.translation_model),
        ("요약", config.summarization_provider, config.summarization_model),
        ("임베딩", config.embedding_provider, config.embedding_model),
    ]

    for role, provider_name, model in roles:
        info = providers.get(provider_name)
        installed = "OK" if info else "NO"
        key_status, key_source, key_detail = _check_api_key_status(config, provider_name, model)

        if not info:
            installed = "NO"
            key_status = "WARN"
            key_source = "-"
            key_detail = f"{provider_name} 모듈을 찾을 수 없음"
        elif key_status == "WARN":
            pass
        provider_table.add_row(role, provider_name, model, installed, key_status, key_source, key_detail)

    console.print(provider_table)

    run_test_table = Table(title="런타임 점검")
    run_test_table.add_column("항목")
    run_test_table.add_column("상태", justify="center")
    run_test_table.add_column("메시지")

    try:
        sqlite_db = _sqlite_db(khub_ctx)
        stats = sqlite_db.get_stats()
        run_test_table.add_row("SQLite", "OK", f"논문 {stats.get('papers', 0)}개")
    except Exception as e:
        run_test_table.add_row("SQLite", "FAIL", str(e))

    try:
        config.validate(
            require_providers=list(
                dict.fromkeys([config.translation_provider, config.summarization_provider, config.embedding_provider])
            )
        )
        run_test_table.add_row("설정 검증", "OK", "필수 값/키 누락 없음")
    except Exception as e:
        run_test_table.add_row("설정 검증", "WARN", str(e).replace("\n", " "))

    try:
        vector_db = _vector_db(khub_ctx)
        run_test_table.add_row("Vector DB", "OK", f"컬렉션: {config.collection_name} ({vector_db.count()})")
    except Exception as e:
        run_test_table.add_row("Vector DB", "WARN", str(e))

    console.print(run_test_table)

    if check_events:
        event_table = Table(title="이벤트 정합성 (--check-events)")
        event_table.add_column("항목")
        event_table.add_column("상태", justify="center")
        event_table.add_column("메시지")
        rows, event_ok = _event_integrity_rows(config)
        for item, status, message in rows:
            event_table.add_row(item, status, message)
        console.print(event_table)

    if check_pipeline:
        pipeline_table = Table(title="파이프라인 정합성 (--check-pipeline)")
        pipeline_table.add_column("항목")
        pipeline_table.add_column("상태", justify="center")
        pipeline_table.add_column("메시지")
        rows, pipeline_ok = _pipeline_integrity_rows(config)
        for item, status, message in rows:
            pipeline_table.add_row(item, status, message)
        console.print(pipeline_table)

    if check_paper_sources:
        paper_table = Table(title="논문 source 정합성 (--check-paper-sources)")
        paper_table.add_column("항목")
        paper_table.add_column("상태", justify="center")
        paper_table.add_column("메시지")
        rows, paper_source_ok = _paper_source_integrity_rows(config)
        for item, status, message in rows:
            paper_table.add_row(item, status, message)
        console.print(paper_table)

    console.print("\n[dim]진단 요약: 키 경고가 있으면 khub init 또는 khub config set providers.<provider>.api_key 로 업데이트하세요.[/dim]")
    return {"event_ok": event_ok, "pipeline_ok": pipeline_ok, "paper_source_ok": paper_source_ok}


@click.command("health")
@click.option("--check-events", is_flag=True, help="ontology_events JSONL/SQLite/snapshot 정합성 점검")
@click.option("--check-pipeline", is_flag=True, help="crawl pipeline 상태/manifest/checkpoint 정합성 점검")
@click.option("--check-paper-sources", is_flag=True, help="known paper source cleanup rule 기준 relink/manual-fix 상태 점검")
@click.pass_context
def health_cmd(ctx, check_events, check_pipeline, check_paper_sources):
    """설치/설정/연결 상태를 한 번에 진단합니다."""
    result = run_health(
        ctx.obj["khub"],
        check_events=check_events,
        check_pipeline=check_pipeline,
        check_paper_sources=check_paper_sources,
    )
    if check_events and not bool(result.get("event_ok", True)):
        raise click.ClickException("health check failed: event integrity mismatch")
    if check_pipeline and not bool(result.get("pipeline_ok", True)):
        raise click.ClickException("health check failed: pipeline integrity mismatch")
    if check_paper_sources and not bool(result.get("paper_source_ok", True)):
        raise click.ClickException("health check failed: paper source integrity mismatch")

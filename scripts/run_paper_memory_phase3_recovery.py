#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from knowledge_hub.application.runtime_diagnostics import parser_runtime_status

from knowledge_hub.application.context import AppContextFactory
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.source_cleanup import (
    apply_source_cleanup_plan,
    build_source_cleanup_plan,
    load_source_cleanup_queue,
    write_source_cleanup_artifacts,
)
from knowledge_hub.papers.structured_summary import StructuredPaperSummaryService
from paper_memory_audit import build_paper_memory_audit_rows, summarize_paper_memory_audit


def _load_rows(path_value: str | Path) -> list[dict[str, Any]]:
    path = Path(path_value).expanduser()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_id_file(path: Path, paper_ids: list[str]) -> str:
    path.write_text("\n".join(paper_ids) + ("\n" if paper_ids else ""), encoding="utf-8")
    return str(path)


def _audit_snapshot(*, sqlite_db, paper_ids: list[str], artifact_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        card = sqlite_db.get_paper_memory_card(paper_id)
        if not card:
            continue
        paper = sqlite_db.get_paper(paper_id) or {}
        records.append(
            {
                **dict(card),
                "paper_id": paper_id,
                "title": card.get("title") or paper.get("title") or "",
                "pdf_path": paper.get("pdf_path") or "",
                "text_path": paper.get("text_path") or "",
                "translated_path": paper.get("translated_path") or "",
            }
        )
    rows = build_paper_memory_audit_rows(records)
    summary = summarize_paper_memory_audit(rows)
    rows_path = artifact_dir / "paper_memory_quality_rows.jsonl"
    rows_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    summary_path = artifact_dir / "paper_memory_quality_summary.json"
    _dump_json(summary_path, summary)
    problem_csv = artifact_dir / "paper_memory_problem_cards.csv"
    with problem_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {
        "summary": summary,
        "rows": rows,
        "artifactPaths": {
            "rows": str(rows_path),
            "summary": str(summary_path),
            "problemCardsCsv": str(problem_csv),
        },
    }


def _detect_parsers() -> dict[str, bool]:
    return {
        "mineru": bool(parser_runtime_status("mineru").get("available")),
        "opendataloader": bool(parser_runtime_status("opendataloader").get("available")),
        "raw": True,
    }


def _lane_split(rows: list[dict[str, Any]], *, failed_apply_ids: set[str]) -> dict[str, list[str]]:
    lane_a: list[str] = []
    lane_b: list[str] = []
    for row in rows:
        paper_id = str(row.get("paperId") or "").strip()
        reasons = list(row.get("reviewReasons") or [])
        if not paper_id:
            continue
        if reasons == ["text_starts_latex"]:
            lane_a.append(paper_id)
            continue
        if any(
            bool(row.get(key))
            for key in ("emptyProblemContext", "emptyMethod", "emptyEvidence", "likelySemanticMismatch", "latexProblemContext")
        ):
            lane_b.append(paper_id)
    lane_c = sorted(set(failed_apply_ids))
    return {
        "lane_a_parser_only": sorted(dict.fromkeys(lane_a)),
        "lane_b_parser_plus_card": sorted(dict.fromkeys(lane_b)),
        "lane_c_manual_or_skip": lane_c,
    }


def _retry_parsers(
    *,
    sqlite_db,
    config,
    parser_available: dict[str, bool],
    lane_a_ids: list[str],
    lane_b_ids: list[str],
    applied_cleanup_ids: set[str],
) -> list[dict[str, Any]]:
    summary_service = StructuredPaperSummaryService(sqlite_db, config)
    memory_builder = PaperMemoryBuilder(sqlite_db)
    results: list[dict[str, Any]] = []
    parser_order = ["mineru", "opendataloader", "raw"]
    for paper_id in [*lane_a_ids, *lane_b_ids]:
        lane = "lane_a_parser_only" if paper_id in set(lane_a_ids) else "lane_b_parser_plus_card"
        chosen_parser = "raw"
        parser_attempts: list[dict[str, Any]] = []
        build_payload: dict[str, Any] = {}
        for parser in parser_order:
            available = bool(parser_available.get(parser))
            if not available:
                parser_attempts.append({"parser": parser, "status": "unavailable"})
                continue
            chosen_parser = parser
            try:
                build_payload = summary_service.build(
                    paper_id=paper_id,
                    paper_parser=parser,
                    refresh_parse=True,
                    quick=True,
                )
            except Exception as error:
                parser_attempts.append({"parser": parser, "status": "error", "warning": str(error)})
                continue
            status = str(build_payload.get("status") or "")
            warnings = [str(item) for item in list(build_payload.get("warnings") or [])[:5]]
            parser_attempts.append({"parser": parser, "status": status or "ok", "warnings": warnings})
            if status != "blocked" or parser == "raw":
                break
        rebuilt = False
        rebuild_reason = ""
        if lane == "lane_b_parser_plus_card":
            memory_builder.build_and_store(paper_id=paper_id)
            rebuilt = True
            rebuild_reason = "lane_b_always_rebuild_after_parser"
        elif paper_id in applied_cleanup_ids and chosen_parser == "raw":
            memory_builder.build_and_store(paper_id=paper_id)
            rebuilt = True
            rebuild_reason = "lane_a_rebuild_after_source_cleanup"
        results.append(
            {
                "paperId": paper_id,
                "lane": lane,
                "chosenParser": chosen_parser,
                "parserAttempts": json.dumps(parser_attempts, ensure_ascii=False),
                "parserAttemptCount": len(parser_attempts),
                "parserBlocked": all(str(item.get("status")) in {"unavailable", "blocked"} for item in parser_attempts),
                "parserWarnings": " | ".join(
                    warning
                    for item in parser_attempts
                    for warning in list(item.get("warnings") or [])
                    if str(warning).strip()
                ),
                "rebuiltPaperMemory": rebuilt,
                "rebuildReason": rebuild_reason,
                "summaryStatus": str(build_payload.get("status") or ""),
            }
        )
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run phase 3 paper-memory source/parser recovery.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--audit-rows",
        default="eval/knowledgeos/runs/paper_memory_gpt54_rebuild/audit_round2_post_seq/paper_memory_quality_rows.jsonl",
    )
    parser.add_argument(
        "--cleanup-queue",
        default="eval/knowledgeos/runs/paper_memory_gpt54_rebuild/audit_round2_post_seq/paper_memory_source_cleanup_queue.csv",
    )
    parser.add_argument(
        "--run-root",
        default="eval/knowledgeos/runs/paper_memory_gpt54_rebuild/phase3",
    )
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args()

    factory = AppContextFactory(config_path=args.config)
    sqlite_db = factory.get_sqlite_db()
    config = factory.build().config
    run_root = Path(args.run_root).expanduser()
    run_root.mkdir(parents=True, exist_ok=True)

    source_rows = _load_rows(args.audit_rows)
    failed_apply_ids = {"2510.04852"}
    lanes = _lane_split(source_rows, failed_apply_ids=failed_apply_ids)
    lane_paths = {
        "laneA": _write_id_file(run_root / "lane_a_parser_only_ids.txt", lanes["lane_a_parser_only"]),
        "laneB": _write_id_file(run_root / "lane_b_parser_plus_card_ids.txt", lanes["lane_b_parser_plus_card"]),
        "laneC": _write_id_file(run_root / "lane_c_manual_queue_ids.txt", lanes["lane_c_manual_or_skip"]),
    }

    target_ids = sorted(dict.fromkeys([*lanes["lane_a_parser_only"], *lanes["lane_b_parser_plus_card"], *lanes["lane_c_manual_or_skip"]]))
    pre_stage = _audit_snapshot(sqlite_db=sqlite_db, paper_ids=target_ids, artifact_dir=run_root / "pre_stage")

    cleanup_rows = load_source_cleanup_queue(args.cleanup_queue)
    cleanup_decisions = build_source_cleanup_plan(cleanup_rows, sqlite_db=sqlite_db)
    cleanup_apply = apply_source_cleanup_plan(sqlite_db=sqlite_db, decisions=cleanup_decisions)
    cleanup_artifacts = write_source_cleanup_artifacts(
        artifact_dir=run_root / "source_cleanup",
        decisions=cleanup_decisions,
        pass_b_ids=target_ids,
    )
    applied_cleanup_ids = {
        str(item.get("paperId") or "")
        for item in cleanup_decisions
        if str(item.get("action") or "") == "relink_to_canonical" and str(item.get("status") or "") == "resolved"
    }
    post_cleanup = _audit_snapshot(sqlite_db=sqlite_db, paper_ids=target_ids, artifact_dir=run_root / "post_source_cleanup")

    parser_available = _detect_parsers()
    parser_results = _retry_parsers(
        sqlite_db=sqlite_db,
        config=config,
        parser_available=parser_available,
        lane_a_ids=lanes["lane_a_parser_only"],
        lane_b_ids=lanes["lane_b_parser_plus_card"],
        applied_cleanup_ids=applied_cleanup_ids,
    )
    parser_csv_path = _write_csv(run_root / "phase3_parser_retry_results.csv", parser_results)
    post_parser = _audit_snapshot(sqlite_db=sqlite_db, paper_ids=target_ids, artifact_dir=run_root / "post_parser_retry")
    post_rebuild = _audit_snapshot(sqlite_db=sqlite_db, paper_ids=target_ids, artifact_dir=run_root / "post_memory_rebuild")

    final_rows = list(post_rebuild["rows"])
    source_warning_only = sorted(
        row["paperId"]
        for row in final_rows
        if list(row.get("reviewReasons") or []) == ["text_starts_latex"] and row["paperId"] not in failed_apply_ids
    )
    manual_queue = []
    for row in final_rows:
        paper_id = str(row.get("paperId") or "")
        reasons = list(row.get("reviewReasons") or [])
        if paper_id in failed_apply_ids or any(reason != "text_starts_latex" for reason in reasons):
            manual_queue.append(
                {
                    "paperId": paper_id,
                    "title": row.get("title") or "",
                    "reviewReasons": ",".join(reasons),
                    "recommendedParser": row.get("recommendedParser") or "",
                    "qualityFlag": row.get("qualityFlag") or "",
                }
            )
    manual_queue.sort(key=lambda item: (item["paperId"], item["title"]))
    manual_queue_path = _write_csv(run_root / "phase3_manual_queue.csv", manual_queue)
    residual_external_ids = sorted(dict.fromkeys(item["paperId"] for item in manual_queue))
    residual_external_path = _write_id_file(run_root / "phase3_residual_external_rebuild_ids.txt", residual_external_ids)

    summary = {
        "schema": "knowledge-hub.paper-memory.phase3-recovery.v1",
        "status": "ok",
        "laneCounts": {key: len(value) for key, value in lanes.items()},
        "parserAvailability": parser_available,
        "cleanup": {
            "queueCount": len(cleanup_rows),
            "decisionCount": len(cleanup_decisions),
            "applySummary": cleanup_apply,
            "artifacts": cleanup_artifacts,
        },
        "stageSummaries": {
            "preStage": pre_stage["summary"],
            "postSourceCleanup": post_cleanup["summary"],
            "postParserRetry": post_parser["summary"],
            "postMemoryRebuild": post_rebuild["summary"],
        },
        "resultBuckets": {
            "sourceWarningOnlyCount": len(source_warning_only),
            "manualQueueCount": len(manual_queue),
            "residualExternalRebuildCount": len(residual_external_ids),
        },
        "artifactPaths": {
            **lane_paths,
            "parserRetryResultsCsv": parser_csv_path,
            "manualQueueCsv": manual_queue_path,
            "residualExternalRebuildIds": residual_external_path,
            "preStageSummary": pre_stage["artifactPaths"]["summary"],
            "postSourceCleanupSummary": post_cleanup["artifactPaths"]["summary"],
            "postParserRetrySummary": post_parser["artifactPaths"]["summary"],
            "postMemoryRebuildSummary": post_rebuild["artifactPaths"]["summary"],
        },
        "notes": [
            "text_starts_latex-only rows are treated as source-warning-only residuals unless other field failures remain.",
            "mineru and opendataloader retries are skipped when unavailable on the current workstation.",
            "2510.04852 remains in the manual/residual bucket because the second external apply failed for that paper.",
        ],
    }
    summary_path = run_root / "phase3_recovery_summary.json"
    _dump_json(summary_path, summary)
    if args.as_json:
        print(json.dumps({**summary, "summaryPath": str(summary_path)}, ensure_ascii=False, indent=2))
    else:
        print(f"phase3 recovery complete: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

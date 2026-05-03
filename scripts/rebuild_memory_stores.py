#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from knowledge_hub.application.context import AppContextFactory
from knowledge_hub.document_memory import DocumentMemoryBuilder
from knowledge_hub.document_memory.extraction import DocumentMemorySchemaExtractor
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_extraction import PaperMemorySchemaExtractor
from knowledge_hub.providers.registry import get_llm
from knowledge_hub.web.card_v2_builder import WebCardV2Builder
from knowledge_hub.web.ingest import make_web_note_id
try:
    from scripts.paper_memory_audit import (
        build_paper_memory_audit_rows,
        compare_paper_memory_audit_summaries,
        load_paper_memory_records,
        summarize_paper_memory_audit,
        write_paper_memory_audit_artifacts,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from paper_memory_audit import (
        build_paper_memory_audit_rows,
        compare_paper_memory_audit_summaries,
        load_paper_memory_records,
        summarize_paper_memory_audit,
        write_paper_memory_audit_artifacts,
    )


def _parse_note_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        payload = json.loads(raw or "{}")
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _normalize_targets(raw: str) -> list[str]:
    requested = [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]
    allowed = {"vault", "paper", "web", "paper-memory", "web-cards"}
    invalid = [item for item in requested if item not in allowed]
    if invalid:
        raise ValueError(f"unsupported targets: {', '.join(invalid)}")
    return requested or ["vault", "paper", "web"]


def _web_urls(notes: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for note in notes:
        metadata = _parse_note_metadata(note.get("metadata"))
        for candidate in (
            metadata.get("canonical_url"),
            metadata.get("url"),
            metadata.get("source_url"),
        ):
            token = str(candidate or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            urls.append(token)
            break
    return urls


def _load_query_strings(path_value: str | None) -> list[str]:
    token = str(path_value or "").strip()
    if not token:
        return []
    path = Path(token).expanduser()
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row and not row.startswith("#")]


_WEB_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9._+-]+|[가-힣]+")
_WEB_QUERY_STOPWORDS = {
    "web",
    "card",
    "v2",
    "ask",
    "latest",
    "update",
    "source",
    "sources",
    "guide",
    "reference",
    "최근",
    "최신",
    "업데이트",
    "질문",
    "필드",
    "필드를",
    "어떤",
    "이유",
    "무엇",
    "왜",
    "역할",
}


def _web_query_forms(query: str) -> list[str]:
    forms: list[str] = []
    raw = str(query or "").strip()
    if raw:
        forms.append(raw)
    for candidate in _WEB_QUERY_TOKEN_RE.findall(raw):
        token = str(candidate or "").strip()
        if not token:
            continue
        lowered = token.casefold()
        if lowered in _WEB_QUERY_STOPWORDS:
            continue
        if len(token) < 3 and not any(ord(ch) > 127 for ch in token):
            continue
        if any(existing.casefold() == lowered for existing in forms):
            continue
        forms.append(token)
        if len(forms) >= 6:
            break
    return forms


def _web_urls_from_queries(sqlite_db, *, queries: list[str], limit_per_query: int = 12) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for query in queries:
        for form in _web_query_forms(query):
            for row in sqlite_db.search_notes(form, limit=max(limit_per_query, 6)):
                if str(row.get("source_type") or "").strip() != "web":
                    continue
                metadata = _parse_note_metadata(row.get("metadata"))
                url = str(metadata.get("canonical_url") or metadata.get("url") or metadata.get("source_url") or "").strip()
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)
            for row in sqlite_db.search_document_memory_units(form, limit=max(limit_per_query, 6), unit_types=["document_summary"]):
                if str(row.get("source_type") or "").strip() != "web":
                    continue
                url = str(row.get("source_ref") or "").strip()
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)
    return urls


def _load_selected_paper_ids(path_value: str | None) -> set[str]:
    token = str(path_value or "").strip()
    if not token:
        return set()
    path = Path(token).expanduser()
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return {row for row in rows if row and not row.startswith("#")}


def _sample_failures(items: list[dict[str, str]], *, limit: int = 10) -> list[dict[str, str]]:
    return items[: max(1, int(limit))]


def _load_json_payload(path_value: str | None) -> dict[str, Any]:
    token = str(path_value or "").strip()
    if not token:
        return {}
    path = Path(token).expanduser()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _build_paper_memory_builder(
    sqlite_db,
    factory: AppContextFactory,
    *,
    extraction_mode: str,
    extractor_provider: str,
    extractor_model: str,
    extractor_timeout: float,
) -> PaperMemoryBuilder:
    mode = str(extraction_mode or "deterministic").strip().lower()
    if mode == "deterministic":
        return PaperMemoryBuilder(sqlite_db)
    cfg = dict(factory.config.get_provider_config(extractor_provider) or {})
    cfg["timeout"] = float(extractor_timeout)
    cfg.setdefault("temperature", 0.0)
    llm = get_llm(extractor_provider, model=extractor_model, **cfg)
    extractor = PaperMemorySchemaExtractor(llm, model=extractor_model)
    return PaperMemoryBuilder(sqlite_db, schema_extractor=extractor, extraction_mode=mode)


def _build_document_memory_builder(
    sqlite_db,
    factory: AppContextFactory,
    *,
    extraction_mode: str,
    extractor_provider: str,
    extractor_model: str,
    extractor_timeout: float,
) -> DocumentMemoryBuilder:
    mode = str(extraction_mode or "deterministic").strip().lower()
    if mode == "deterministic":
        return DocumentMemoryBuilder(sqlite_db, config=factory.config)
    cfg = dict(factory.config.get_provider_config(extractor_provider) or {})
    cfg["timeout"] = float(extractor_timeout)
    cfg.setdefault("temperature", 0.0)
    llm = get_llm(extractor_provider, model=extractor_model, **cfg)
    extractor = DocumentMemorySchemaExtractor(llm, model=extractor_model)
    return DocumentMemoryBuilder(
        sqlite_db,
        config=factory.config,
        schema_extractor=extractor,
        extraction_mode=mode,
    )


def _summarize_paper_memory_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    extractor_models = Counter()
    warning_counter: Counter[str] = Counter()
    attempted = 0
    applied = 0
    fallback_used = 0
    coverage_counter: Counter[str] = Counter()
    total_latency_ms = 0.0
    timed_rows = 0
    for row in rows:
        if bool(row.get("attempted")):
            attempted += 1
        if bool(row.get("applied")):
            applied += 1
        if bool(row.get("fallbackUsed")):
            fallback_used += 1
        try:
            latency_ms = float(row.get("latencyMs") or 0.0)
        except Exception:
            latency_ms = 0.0
        if latency_ms > 0:
            total_latency_ms += latency_ms
            timed_rows += 1
        model = str(row.get("extractorModel") or "").strip()
        if model:
            extractor_models[model] += 1
        for warning in list(row.get("warnings") or []):
            token = str(warning or "").strip()
            if token:
                warning_counter[token] += 1
        for status in dict(row.get("coverageByField") or {}).values():
            token = str(status or "").strip()
            if token:
                coverage_counter[token] += 1
    return {
        "attempted": attempted,
        "applied": applied,
        "fallbackUsed": fallback_used,
        "appliedRate": round((applied / attempted), 4) if attempted else 0.0,
        "fallbackRate": round((fallback_used / attempted), 4) if attempted else 0.0,
        "validPayloadRate": round(((attempted - fallback_used) / attempted), 4) if attempted else 0.0,
        "avgLatencyMs": round((total_latency_ms / timed_rows), 3) if timed_rows else 0.0,
        "extractorModels": dict(extractor_models),
        "warningCounts": dict(warning_counter),
        "coverageStatusCounts": dict(coverage_counter),
    }


def _paper_memory_gate(
    summary: dict[str, Any],
    *,
    mode: str = "deterministic",
    avg_latency_threshold_ms: float = 20000.0,
    audit_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attempted = int(summary.get("attempted") or 0)
    applied_rate = float(summary.get("appliedRate") or 0.0)
    fallback_rate = float(summary.get("fallbackRate") or 0.0)
    valid_payload_rate = float(summary.get("validPayloadRate") or 0.0)
    avg_latency_ms = float(summary.get("avgLatencyMs") or 0.0)
    warning_counts = dict(summary.get("warningCounts") or {})
    reasons: list[str] = []
    passed = True
    normalized_mode = str(mode or "deterministic").strip().lower()
    if attempted == 0:
        return {
            "passed": False,
            "reasons": ["no_attempts"],
            "thresholds": {"successRate": 0.8, "fallbackRate": 0.2, "avgLatencyMs": avg_latency_threshold_ms},
        }
    success_rate = valid_payload_rate if normalized_mode == "shadow" else applied_rate
    if success_rate < 0.8:
        passed = False
        reasons.append("success_rate_below_threshold")
    if fallback_rate > 0.2:
        passed = False
        reasons.append("fallback_rate_above_threshold")
    if avg_latency_ms > float(avg_latency_threshold_ms):
        passed = False
        reasons.append("average_latency_too_high")
    if warning_counts:
        top_warning, top_count = max(warning_counts.items(), key=lambda item: item[1])
        if top_warning.startswith("extractor_error:") and top_count / max(1, attempted) >= 0.8:
            passed = False
            reasons.append("extractor_error_dominates")
    audit = dict(audit_summary or {})
    weak_card_count = int(audit.get("needsReviewCount") or 0)
    if weak_card_count > 0:
        passed = False
        reasons.append("weak_cards_present")
    return {
        "passed": passed,
        "reasons": reasons,
        "thresholds": {
            "successRate": 0.8,
            "fallbackRate": 0.2,
            "avgLatencyMs": avg_latency_threshold_ms,
            "weakCardCount": 0,
        },
        "mode": normalized_mode,
        "successRate": success_rate,
        "weakCardCount": weak_card_count,
        "weakCardRate": float(audit.get("weakCardRate") or 0.0),
    }


def _summarize_document_memory_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    extractor_models = Counter()
    warning_counter: Counter[str] = Counter()
    coverage_counter: Counter[str] = Counter()
    attempted = 0
    applied = 0
    fallback_used = 0
    total_latency_ms = 0.0
    timed_rows = 0
    for row in rows:
        if bool(row.get("attempted")):
            attempted += 1
        if bool(row.get("applied")):
            applied += 1
        if bool(row.get("fallbackUsed")):
            fallback_used += 1
        try:
            latency_ms = float(row.get("latencyMs") or 0.0)
        except Exception:
            latency_ms = 0.0
        if latency_ms > 0:
            total_latency_ms += latency_ms
            timed_rows += 1
        model = str(row.get("extractorModel") or "").strip()
        if model:
            extractor_models[model] += 1
        for warning in list(row.get("warnings") or []):
            token = str(warning or "").strip()
            if token:
                warning_counter[token] += 1
        for status in dict(row.get("coverageByField") or {}).values():
            token = str(status or "").strip()
            if token:
                coverage_counter[token] += 1
    return {
        "attempted": attempted,
        "applied": applied,
        "fallbackUsed": fallback_used,
        "appliedRate": round((applied / attempted), 4) if attempted else 0.0,
        "fallbackRate": round((fallback_used / attempted), 4) if attempted else 0.0,
        "validPayloadRate": round(((attempted - fallback_used) / attempted), 4) if attempted else 0.0,
        "avgLatencyMs": round((total_latency_ms / timed_rows), 3) if timed_rows else 0.0,
        "extractorModels": dict(extractor_models),
        "warningCounts": dict(warning_counter),
        "coverageStatusCounts": dict(coverage_counter),
    }


def _document_memory_gate(
    summary: dict[str, Any],
    *,
    mode: str = "deterministic",
    source_kind: str = "",
) -> dict[str, Any]:
    normalized_source = str(source_kind or "").strip().lower()
    avg_latency_threshold_ms = 20000.0
    if normalized_source == "paper" and str(mode or "deterministic").strip().lower() == "shadow":
        avg_latency_threshold_ms = 25000.0
    gate = _paper_memory_gate(
        summary,
        mode=mode,
        avg_latency_threshold_ms=avg_latency_threshold_ms,
    )
    gate["sourceKind"] = normalized_source or "unknown"
    return gate


def _default_paper_memory_artifact_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.home() / ".khub" / "runs" / "paper-memory-extraction" / stamp


def _default_document_memory_artifact_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.home() / ".khub" / "runs" / "document-memory-extraction" / stamp


def _write_paper_memory_artifacts(
    *,
    artifact_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    audit_rows: list[dict[str, Any]] | None = None,
    audit_summary: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows_path = artifact_dir / "paper_memory_extraction_rows.jsonl"
    summary_path = artifact_dir / "paper_memory_extraction_summary.json"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"rows": str(rows_path), "summary": str(summary_path)}
    if audit_rows is not None and audit_summary is not None:
        audit_paths = write_paper_memory_audit_artifacts(
            artifact_dir=artifact_dir,
            rows=audit_rows,
            summary=audit_summary,
            comparison=comparison,
        )
        result.update(
            {
                "auditRows": audit_paths.get("rows", ""),
                "auditSummary": audit_paths.get("summary", ""),
                "problemCardsCsv": audit_paths.get("problemCardsCsv", ""),
                "top20": audit_paths.get("top20", ""),
            }
        )
        if audit_paths.get("comparison"):
            result["comparison"] = str(audit_paths["comparison"])
    return result


def _write_document_memory_artifacts(*, artifact_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows_path = artifact_dir / "document_memory_extraction_rows.jsonl"
    summary_path = artifact_dir / "document_memory_extraction_summary.json"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"rows": str(rows_path), "summary": str(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild paper/document memory stores using existing builders.")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument(
        "--targets",
        default="vault,paper,web",
        help="Comma-separated targets to rebuild: vault,paper,web,paper-memory,web-cards",
    )
    parser.add_argument("--vault-limit", type=int, default=200000, help="Max vault notes to rebuild")
    parser.add_argument("--vault-offset", type=int, default=0, help="Vault note offset for batched rebuilds")
    parser.add_argument("--paper-limit", type=int, default=5000, help="Max papers to rebuild")
    parser.add_argument(
        "--paper-id-file",
        default=None,
        help="Optional newline-delimited paper id file. When set, only matching papers are processed.",
    )
    parser.add_argument("--web-limit", type=int, default=200000, help="Max web notes to inspect")
    parser.add_argument("--web-offset", type=int, default=0, help="Web note offset for batched rebuilds")
    parser.add_argument(
        "--web-query-file",
        default=None,
        help="Optional newline-delimited query file. When set with web-cards, only URLs matched from these queries are backfilled.",
    )
    parser.add_argument(
        "--web-query-limit-per-query",
        type=int,
        default=12,
        help="Max web note/document-memory candidates to inspect per query when --web-query-file is used.",
    )
    parser.add_argument(
        "--paper-parser",
        default="raw",
        choices=["raw", "mineru", "opendataloader"],
        help="Document-memory paper parser mode",
    )
    parser.add_argument("--refresh-parse", action="store_true", help="Force parser artifact refresh for paper docs")
    parser.add_argument(
        "--paper-memory-extraction-mode",
        default="deterministic",
        choices=["deterministic", "shadow", "schema"],
        help="Paper-memory extraction mode. Keeps stored PaperMemoryCard shape unchanged.",
    )
    parser.add_argument(
        "--paper-memory-extractor-provider",
        default="ollama",
        help="Provider for paper-memory schema extraction when mode is shadow/schema",
    )
    parser.add_argument(
        "--paper-memory-extractor-model",
        default="exaone3.5:7.8b",
        help="Model for paper-memory schema extraction when mode is shadow/schema",
    )
    parser.add_argument(
        "--paper-memory-extractor-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for paper-memory schema extraction calls",
    )
    parser.add_argument(
        "--paper-memory-artifact-dir",
        default=None,
        help="Optional directory for paper-memory per-paper extraction artifacts (auto-created for shadow/schema runs)",
    )
    parser.add_argument(
        "--paper-memory-compare-to-summary",
        default=None,
        help="Optional previous paper-memory quality summary JSON path for staged comparison",
    )
    parser.add_argument(
        "--document-memory-paper-extraction-mode",
        default="deterministic",
        choices=["deterministic", "shadow", "schema"],
        help="Document-memory extraction mode for paper documents only.",
    )
    parser.add_argument(
        "--document-memory-paper-extractor-provider",
        default="ollama",
        help="Provider for document-memory paper schema extraction when mode is shadow/schema",
    )
    parser.add_argument(
        "--document-memory-paper-extractor-model",
        default="exaone3.5:7.8b",
        help="Model for document-memory paper schema extraction when mode is shadow/schema",
    )
    parser.add_argument(
        "--document-memory-paper-extractor-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for document-memory paper schema extraction calls",
    )
    parser.add_argument(
        "--document-memory-paper-artifact-dir",
        default=None,
        help="Optional directory for document-memory paper extraction artifacts (auto-created for shadow/schema runs)",
    )
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON summary")
    args = parser.parse_args()

    targets = _normalize_targets(args.targets)
    factory = AppContextFactory(config_path=args.config)
    sqlite_db = factory.get_sqlite_db()
    document_builder = DocumentMemoryBuilder(sqlite_db, config=factory.config)
    web_card_builder = WebCardV2Builder(sqlite_db)
    document_paper_builder = _build_document_memory_builder(
        sqlite_db,
        factory,
        extraction_mode=str(args.document_memory_paper_extraction_mode),
        extractor_provider=str(args.document_memory_paper_extractor_provider),
        extractor_model=str(args.document_memory_paper_extractor_model),
        extractor_timeout=float(args.document_memory_paper_extractor_timeout),
    )
    paper_builder = _build_paper_memory_builder(
        sqlite_db,
        factory,
        extraction_mode=str(args.paper_memory_extraction_mode),
        extractor_provider=str(args.paper_memory_extractor_provider),
        extractor_model=str(args.paper_memory_extractor_model),
        extractor_timeout=float(args.paper_memory_extractor_timeout),
    )

    payload: dict[str, Any] = {
        "status": "ok",
        "targets": targets,
        "documentMemory": {},
        "paperMemory": {},
        "warnings": [],
    }

    if "vault" in targets:
        notes = sqlite_db.list_notes(
            source_type="vault",
            limit=max(1, int(args.vault_limit)),
            offset=max(0, int(args.vault_offset)),
        )
        failures: list[dict[str, str]] = []
        built = 0
        for note in notes:
            note_id = str(note.get("id") or "").strip()
            if not note_id:
                continue
            try:
                document_builder.build_and_store_note(note_id=note_id)
                built += 1
            except Exception as error:
                failures.append({"target": note_id, "error": str(error)})
        payload["documentMemory"]["vault"] = {
            "requested": len(notes),
            "rebuilt": built,
            "failed": len(failures),
            "offset": max(0, int(args.vault_offset)),
            "limit": max(1, int(args.vault_limit)),
            "sampleFailures": _sample_failures(failures),
        }

    papers: list[dict[str, Any]] = []
    if "paper" in targets or "paper-memory" in targets:
        papers = sqlite_db.list_papers(limit=max(1, int(args.paper_limit)))
        selected_ids = _load_selected_paper_ids(args.paper_id_file)
        if selected_ids:
            papers = [paper for paper in papers if str(paper.get("arxiv_id") or "").strip() in selected_ids]

    if "paper" in targets:
        doc_failures: list[dict[str, str]] = []
        doc_built = 0
        doc_extraction_rows: list[dict[str, Any]] = []
        for paper in papers:
            paper_id = str(paper.get("arxiv_id") or "").strip()
            if not paper_id:
                continue
            try:
                document_paper_builder.build_and_store_paper(
                    paper_id=paper_id,
                    paper_parser=str(args.paper_parser),
                    refresh_parse=bool(args.refresh_parse),
                )
                doc_built += 1
            except Exception as error:
                doc_failures.append({"target": paper_id, "error": str(error)})
            finally:
                doc_extraction_rows.append(
                    {
                        "documentId": f"paper:{paper_id}",
                        "paperId": paper_id,
                        "title": str(paper.get("title") or ""),
                        **document_paper_builder.get_last_extraction_diagnostics(f"paper:{paper_id}"),
                    }
                )
        paper_payload = {
            "requested": len(papers),
            "rebuilt": doc_built,
            "failed": len(doc_failures),
            "sampleFailures": _sample_failures(doc_failures),
        }
        if str(args.document_memory_paper_extraction_mode) in {"shadow", "schema"}:
            extraction_summary = _summarize_document_memory_diagnostics(doc_extraction_rows)
            gate = _document_memory_gate(
                extraction_summary,
                mode=str(args.document_memory_paper_extraction_mode),
                source_kind="paper",
            )
            paper_payload["extractionMode"] = str(args.document_memory_paper_extraction_mode)
            paper_payload["extractorProvider"] = str(args.document_memory_paper_extractor_provider)
            paper_payload["extractorModel"] = str(args.document_memory_paper_extractor_model)
            paper_payload["extractionDiagnostics"] = extraction_summary
            paper_payload["promotionGate"] = gate
            artifact_dir = (
                Path(str(args.document_memory_paper_artifact_dir)).expanduser()
                if args.document_memory_paper_artifact_dir
                else _default_document_memory_artifact_dir()
            )
            paper_payload["artifactPaths"] = _write_document_memory_artifacts(
                artifact_dir=artifact_dir,
                rows=doc_extraction_rows,
                summary={"documentMemory": {"paper": paper_payload}},
            )
        payload["documentMemory"]["paper"] = paper_payload

    if "paper" in targets or "paper-memory" in targets:
        card_failures: list[dict[str, str]] = []
        card_built = 0
        extraction_rows: list[dict[str, Any]] = []
        for paper in papers:
            paper_id = str(paper.get("arxiv_id") or "").strip()
            if not paper_id:
                continue
            try:
                paper_builder.build_and_store(paper_id=paper_id)
                card_built += 1
            except Exception as error:
                card_failures.append({"target": paper_id, "error": str(error)})
            finally:
                extraction_rows.append(
                    {
                        "paperId": paper_id,
                        "title": str(paper.get("title") or ""),
                        **paper_builder.get_last_extraction_diagnostics(paper_id),
                    }
                )
        extraction_summary = _summarize_paper_memory_diagnostics(extraction_rows)
        paper_ids = {str(paper.get("arxiv_id") or "").strip() for paper in papers if str(paper.get("arxiv_id") or "").strip()}
        audit_rows = build_paper_memory_audit_rows(
            load_paper_memory_records(sqlite_db, paper_ids=paper_ids or None),
            diagnostics_by_paper_id={str(row.get("paperId") or ""): row for row in extraction_rows},
        )
        audit_summary = summarize_paper_memory_audit(audit_rows)
        previous_summary = _load_json_payload(args.paper_memory_compare_to_summary)
        comparison = compare_paper_memory_audit_summaries(audit_summary, previous_summary) if previous_summary else None
        gate = _paper_memory_gate(
            extraction_summary,
            mode=str(args.paper_memory_extraction_mode),
            audit_summary=audit_summary,
        )
        payload["paperMemory"] = {
            "requested": len(papers),
            "rebuilt": card_built,
            "failed": len(card_failures),
            "sampleFailures": _sample_failures(card_failures),
            "extractionMode": str(args.paper_memory_extraction_mode),
            "extractorProvider": str(args.paper_memory_extractor_provider),
            "extractorModel": str(args.paper_memory_extractor_model),
            "extractionDiagnostics": extraction_summary,
            "qualityAudit": audit_summary,
            "promotionGate": gate,
        }
        if comparison:
            payload["paperMemory"]["comparison"] = comparison
        if str(args.paper_memory_extraction_mode) in {"shadow", "schema"}:
            artifact_dir = (
                Path(str(args.paper_memory_artifact_dir)).expanduser()
                if args.paper_memory_artifact_dir
                else _default_paper_memory_artifact_dir()
            )
            payload["paperMemory"]["artifactPaths"] = _write_paper_memory_artifacts(
                artifact_dir=artifact_dir,
                rows=extraction_rows,
                summary={"paperMemory": payload["paperMemory"]},
                audit_rows=audit_rows,
                audit_summary=audit_summary,
                comparison=comparison,
            )

    if "web" in targets:
        notes = sqlite_db.list_notes(
            source_type="web",
            limit=max(1, int(args.web_limit)),
            offset=max(0, int(args.web_offset)),
        )
        urls = _web_urls(notes)
        failures: list[dict[str, str]] = []
        built = 0
        for canonical_url in urls:
            try:
                document_builder.build_and_store_web(canonical_url=canonical_url)
                built += 1
            except Exception as error:
                failures.append({"target": canonical_url, "error": str(error)})
        payload["documentMemory"]["web"] = {
            "requested": len(urls),
            "rebuilt": built,
            "failed": len(failures),
            "offset": max(0, int(args.web_offset)),
            "limit": max(1, int(args.web_limit)),
            "sampleFailures": _sample_failures(failures),
        }

    if "web-cards" in targets:
        query_rows = _load_query_strings(args.web_query_file)
        if query_rows:
            urls = _web_urls_from_queries(
                sqlite_db,
                queries=query_rows,
                limit_per_query=max(1, int(args.web_query_limit_per_query)),
            )
        else:
            notes = sqlite_db.list_notes(
                source_type="web",
                limit=max(1, int(args.web_limit)),
                offset=max(0, int(args.web_offset)),
            )
            urls = _web_urls(notes)
        failures: list[dict[str, str]] = []
        built = 0
        memory_rebuilt = 0
        for canonical_url in urls:
            try:
                document_id = make_web_note_id(canonical_url)
                if not sqlite_db.get_document_memory_summary(document_id):
                    document_builder.build_and_store_web(canonical_url=canonical_url)
                    memory_rebuilt += 1
                web_card_builder.build_and_store(canonical_url=canonical_url)
                built += 1
            except Exception as error:
                failures.append({"target": canonical_url, "error": str(error)})
        payload["webCards"] = {
            "requested": len(urls),
            "rebuilt": built,
            "failed": len(failures),
            "offset": max(0, int(args.web_offset)),
            "limit": max(1, int(args.web_limit)),
            "queryFile": str(args.web_query_file or ""),
            "queryCount": len(query_rows),
            "memoryRebuilt": memory_rebuilt,
            "sampleFailures": _sample_failures(failures),
        }

    counts = Counter()
    try:
        counts["document_memory_docs"] = int(
            sqlite_db.conn.execute("select count(distinct document_id) from document_memory_units").fetchone()[0]
        )
        counts["document_memory_units"] = int(sqlite_db.conn.execute("select count(*) from document_memory_units").fetchone()[0])
        counts["paper_memory_cards"] = int(sqlite_db.conn.execute("select count(*) from paper_memory_cards").fetchone()[0])
        counts["web_cards_v2"] = int(sqlite_db.conn.execute("select count(*) from web_cards_v2").fetchone()[0])
        counts["memory_relations"] = int(sqlite_db.conn.execute("select count(*) from memory_relations").fetchone()[0])
    except Exception as error:
        payload["warnings"].append(f"count query failed: {error}")
    payload["counts"] = dict(counts)

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("memory rebuild summary")
    for category, items in payload["documentMemory"].items():
        print(
            f"- document:{category} requested={items.get('requested', 0)} "
            f"rebuilt={items.get('rebuilt', 0)} failed={items.get('failed', 0)}"
        )
    if payload.get("paperMemory"):
        items = payload["paperMemory"]
        print(
            f"- paper-memory requested={items.get('requested', 0)} "
            f"rebuilt={items.get('rebuilt', 0)} failed={items.get('failed', 0)} mode={items.get('extractionMode')}"
        )
        extraction = items.get("extractionDiagnostics") or {}
        if extraction:
            print(f"- paper-memory extraction {extraction}")
        audit = items.get("qualityAudit") or {}
        if audit:
            print(f"- paper-memory audit {audit}")
        gate = items.get("promotionGate") or {}
        if gate:
            print(f"- paper-memory gate {gate}")
    if payload.get("webCards"):
        items = payload["webCards"]
        print(
            f"- web-cards requested={items.get('requested', 0)} "
            f"rebuilt={items.get('rebuilt', 0)} failed={items.get('failed', 0)} "
            f"memory_rebuilt={items.get('memoryRebuilt', 0)}"
        )
    if payload.get("counts"):
        print(f"- counts {payload['counts']}")
    if payload.get("warnings"):
        print(f"- warnings {payload['warnings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

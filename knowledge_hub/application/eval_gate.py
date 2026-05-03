"""Shared eval report + gate helpers for retrieval and memory promotion.

This module keeps evaluation logic inspectable and reusable across CLI entry
surfaces without changing runtime ranking behavior.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from knowledge_hub.ai.retrieval_eval import build_eval_report
from knowledge_hub.application.ask_v2_eval import build_ask_v2_eval_report
from knowledge_hub.application.runtime_diagnostics import build_runtime_diagnostics
from knowledge_hub.document_memory import DocumentMemoryRetriever
from knowledge_hub.knowledge.synthesis import ClaimSynthesisService
from knowledge_hub.papers.memory_eval import PaperMemoryEvalCase, PaperMemoryEvalHarness
from knowledge_hub.papers.raw_summary import build_raw_summary_artifact
from knowledge_hub.papers.structured_summary import StructuredPaperSummaryService

POSITIVE_LABELS = {"good", "1", "pass", "relevant"}
PARTIAL_LABELS = {"partial"}
NEGATIVE_LABELS = {"bad", "0", "fail", "irrelevant"}


def _read_queries(path: str | Path) -> list[str]:
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in rows if line.strip() and not line.strip().startswith("#")]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value or "").strip())
    except Exception:
        return default


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _normalize_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in POSITIVE_LABELS:
        return "positive"
    if raw in PARTIAL_LABELS:
        return "partial"
    if raw in NEGATIVE_LABELS:
        return "negative"
    return ""


def _normalize_boolish(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def export_document_memory_eval_template(
    sqlite_db,
    *,
    db_path: str,
    queries_path: str | Path,
    out_path: str | Path,
    top_k: int = 3,
) -> dict[str, Any]:
    queries_file = Path(queries_path)
    queries = _read_queries(queries_file)
    if not queries:
        raise ValueError(f"No queries found: {queries_file}")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    retriever = DocumentMemoryRetriever(sqlite_db)
    row_count = 0

    with out_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query",
                "rank",
                "document_id",
                "document_title",
                "source_type",
                "matched_unit_title",
                "matched_unit_type",
                "matched_summary",
                "matched_segment_anchor",
                "matched_segment_titles",
                "matched_segment_text",
                "document_thesis",
                "related_unit_titles",
                "strategy",
                "title_match",
                "source_type_boost",
                "generic_title_penalty",
                "placeholder_penalty",
                "label",
                "notes",
            ],
        )
        writer.writeheader()

        for query in queries:
            results = retriever.search(query, limit=max(1, int(top_k)))
            for rank, item in enumerate(results, start=1):
                matched = item.get("matchedUnit") or {}
                related = item.get("relatedUnits") or []
                matched_segment = item.get("matchedSegment") or {}
                segment_units = matched_segment.get("units") or []
                retrieval_signals = item.get("retrievalSignals") or {}
                writer.writerow(
                    {
                        "query": query,
                        "rank": rank,
                        "document_id": item.get("documentId", ""),
                        "document_title": item.get("documentTitle", ""),
                        "source_type": item.get("sourceType", ""),
                        "matched_unit_title": matched.get("title", ""),
                        "matched_unit_type": matched.get("unitType", ""),
                        "matched_summary": matched.get("contextualSummary", ""),
                        "matched_segment_anchor": matched_segment.get("anchorUnitId", ""),
                        "matched_segment_titles": " | ".join(
                            str(unit.get("title") or "") for unit in segment_units if str(unit.get("title") or "").strip()
                        ),
                        "matched_segment_text": str(matched_segment.get("segmentText") or ""),
                        "document_thesis": item.get("documentThesis", ""),
                        "related_unit_titles": " | ".join(
                            str(unit.get("title") or "") for unit in related if str(unit.get("title") or "").strip()
                        ),
                        "strategy": retrieval_signals.get("strategy", ""),
                        "title_match": retrieval_signals.get("titleMatch", ""),
                        "source_type_boost": retrieval_signals.get("sourceTypeBoost", ""),
                        "generic_title_penalty": retrieval_signals.get("genericTitlePenalty", ""),
                        "placeholder_penalty": retrieval_signals.get("placeholderPenalty", ""),
                        "label": "",
                        "notes": "",
                    }
                )
                row_count += 1

    return {
        "schema": "knowledge-hub.document-memory.eval.prepare.result.v1",
        "status": "ok",
        "dbPath": str(db_path),
        "queriesPath": str(queries_file),
        "outPath": str(out_file),
        "queryCount": len(queries),
        "rowCount": row_count,
        "topK": int(top_k),
        "warnings": [],
    }


def build_document_memory_eval_report(
    csv_path: str | Path,
    *,
    label_col: str = "label",
    query_col: str = "query",
    rank_col: str = "rank",
) -> dict[str, Any]:
    path = Path(csv_path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    unknown_labels: set[str] = set()

    for row in rows:
        query = str(row.get(query_col, "")).strip()
        if query:
            grouped[query].append(row)
        label_value = str(row.get(label_col, "")).strip()
        if label_value and not _normalize_label(label_value):
            unknown_labels.add(label_value)

    query_count = len(grouped)
    labeled_query_count = 0
    top1_good = 0
    top3_good_hit = 0
    top1_partial = 0
    top1_bad = 0

    for query, items in grouped.items():
        _ = query
        ordered = sorted(items, key=lambda row: _safe_int(row.get(rank_col), default=0))
        labels = [_normalize_label(row.get(label_col, "")) for row in ordered]
        if not any(label for label in labels):
            continue
        labeled_query_count += 1

        top1_label = labels[0] if labels else ""
        if top1_label == "positive":
            top1_good += 1
        elif top1_label == "partial":
            top1_partial += 1
        elif top1_label == "negative":
            top1_bad += 1

        if any(label == "positive" for label in labels[:3]):
            top3_good_hit += 1

    warnings: list[str] = []
    if labeled_query_count < query_count:
        warnings.append(f"only {labeled_query_count}/{query_count} queries have labeled rows")
    if unknown_labels:
        warnings.append(f"ignored unknown labels: {', '.join(sorted(unknown_labels))}")

    metrics = {
        "queryCount": query_count,
        "labeledQueryCount": labeled_query_count,
        "top1GoodRate": _rate(top1_good, labeled_query_count),
        "top3GoodHitRate": _rate(top3_good_hit, labeled_query_count),
        "top1PartialRate": _rate(top1_partial, labeled_query_count),
        "top1BadRate": _rate(top1_bad, labeled_query_count),
    }
    return {
        "schema": "knowledge-hub.document-memory.eval.report.v1",
        "status": "ok" if labeled_query_count > 0 else "warning",
        "dataset": {
            "csvPath": str(path),
            "queryCol": query_col,
            "rankCol": rank_col,
            "labelCol": label_col,
            "labelMapping": {
                "positive": sorted(POSITIVE_LABELS),
                "partial": sorted(PARTIAL_LABELS),
                "negative": sorted(NEGATIVE_LABELS),
            },
        },
        "metrics": metrics,
        "warnings": warnings,
    }


def build_memory_router_eval_report(
    csv_path: str | Path,
    *,
    baseline_csv: str | Path | None = None,
    label_col: str = "label",
    query_col: str = "query",
    rank_col: str = "rank",
    no_result_col: str = "no_result",
    temporal_col: str = "temporal_query",
    wrong_era_col: str = "wrong_era",
) -> dict[str, Any]:
    path = Path(csv_path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    unknown_labels: set[str] = set()

    for row in rows:
        query = str(row.get(query_col, "")).strip()
        if query:
            grouped[query].append(row)
        label_value = str(row.get(label_col, "")).strip()
        if label_value and not _normalize_label(label_value):
            unknown_labels.add(label_value)

    query_count = len(grouped)
    labeled_query_count = 0
    top1_good = 0
    top3_good_hit = 0
    no_result_count = 0
    temporal_query_count = 0
    labeled_temporal_query_count = 0
    wrong_era_top1_count = 0

    for _query, items in grouped.items():
        ordered = sorted(items, key=lambda row: _safe_int(row.get(rank_col), default=0))
        labels = [_normalize_label(row.get(label_col, "")) for row in ordered]
        top1_row = ordered[0] if ordered else {}

        if _normalize_boolish(top1_row.get(no_result_col, "")):
            no_result_count += 1

        is_temporal = _normalize_boolish(top1_row.get(temporal_col, "")) or any(
            _normalize_boolish(row.get(temporal_col, "")) for row in ordered
        )
        if is_temporal:
            temporal_query_count += 1
            if _normalize_boolish(top1_row.get(wrong_era_col, "")):
                wrong_era_top1_count += 1

        if not any(label for label in labels):
            continue
        labeled_query_count += 1
        if is_temporal:
            labeled_temporal_query_count += 1

        if labels and labels[0] == "positive":
            top1_good += 1
        if any(label == "positive" for label in labels[:3]):
            top3_good_hit += 1

    warnings: list[str] = []
    if labeled_query_count < query_count:
        warnings.append(f"only {labeled_query_count}/{query_count} queries have labeled rows")
    if unknown_labels:
        warnings.append(f"ignored unknown labels: {', '.join(sorted(unknown_labels))}")

    metrics = {
        "queryCount": query_count,
        "labeledQueryCount": labeled_query_count,
        "top1GoodRate": _rate(top1_good, labeled_query_count),
        "top3GoodHitRate": _rate(top3_good_hit, labeled_query_count),
        "noResultRate": _rate(no_result_count, query_count),
        "temporalQueryCount": temporal_query_count,
        "labeledTemporalQueryCount": labeled_temporal_query_count,
        "wrongEraHitRate": _rate(wrong_era_top1_count, temporal_query_count),
    }
    baseline: dict[str, Any] | None = None
    delta: dict[str, Any] | None = None
    if baseline_csv:
        baseline = build_memory_router_eval_report(
            baseline_csv,
            baseline_csv=None,
            label_col=label_col,
            query_col=query_col,
            rank_col=rank_col,
            no_result_col=no_result_col,
            temporal_col=temporal_col,
            wrong_era_col=wrong_era_col,
        )
        base_metrics = dict(baseline.get("metrics") or {})
        base_wrong_era = float(base_metrics.get("wrongEraHitRate", 0.0) or 0.0)
        current_wrong_era = float(metrics.get("wrongEraHitRate", 0.0) or 0.0)
        wrong_era_reduction = 0.0
        if base_wrong_era > 0:
            wrong_era_reduction = (base_wrong_era - current_wrong_era) / base_wrong_era
        delta = {
            "top1GoodRate": round(float(metrics["top1GoodRate"]) - float(base_metrics.get("top1GoodRate", 0.0) or 0.0), 6),
            "noResultRate": round(float(metrics["noResultRate"]) - float(base_metrics.get("noResultRate", 0.0) or 0.0), 6),
            "wrongEraHitRate": round(current_wrong_era - base_wrong_era, 6),
            "wrongEraReductionRate": round(wrong_era_reduction, 6),
        }

    payload = {
        "schema": "knowledge-hub.memory-router.eval.report.v1",
        "status": "ok" if labeled_query_count > 0 else "warning",
        "dataset": {
            "csvPath": str(path),
            "baselineCsvPath": str(baseline_csv) if baseline_csv else "",
            "queryCol": query_col,
            "rankCol": rank_col,
            "labelCol": label_col,
            "noResultCol": no_result_col,
            "temporalCol": temporal_col,
            "wrongEraCol": wrong_era_col,
        },
        "metrics": metrics,
        "warnings": warnings,
    }
    if baseline is not None:
        payload["baselineMetrics"] = dict(baseline.get("metrics") or {})
    if delta is not None:
        payload["deltaVsBaseline"] = delta
    return payload
    return {
        "schema": "knowledge-hub.document-memory.eval.report.v1",
        "status": "ok" if labeled_query_count > 0 else "warning",
        "dataset": {
            "csvPath": str(path),
            "queryCol": query_col,
            "rankCol": rank_col,
            "labelCol": label_col,
            "labelMapping": {
                "positive": sorted(POSITIVE_LABELS),
                "partial": sorted(PARTIAL_LABELS),
                "negative": sorted(NEGATIVE_LABELS),
            },
        },
        "metrics": metrics,
        "warnings": warnings,
    }


def _load_paper_memory_cases(path: str | Path) -> list[PaperMemoryEvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("paper-memory cases file must contain a JSON array")
    return [PaperMemoryEvalCase(**item) for item in raw]


def build_paper_memory_eval_report(sqlite_db, cases_path: str | Path) -> dict[str, Any]:
    path = Path(cases_path)
    cases = _load_paper_memory_cases(path)
    report = PaperMemoryEvalHarness(sqlite_db).evaluate_cases(cases)
    summary = dict(report.get("summary") or {})
    paper_memory = dict(summary.get("paperMemory") or {})
    return {
        "schema": "knowledge-hub.paper-memory.eval.report.v1",
        "status": "ok",
        "casesPath": str(path),
        "caseCount": int(summary.get("caseCount", 0) or 0),
        "summary": summary,
        "metrics": {
            "top1MatchRate": paper_memory.get("top1MatchRate", 0.0),
            "top3MatchRate": paper_memory.get("top3MatchRate", 0.0),
            "noResultRate": paper_memory.get("noResultRate", 0.0),
            "weakCardRate": paper_memory.get("weakCardRate", 0.0),
            "top1LiftVsSearchPapers": paper_memory.get("top1LiftVsSearchPapers", 0.0),
            "top1LiftVsLookup": paper_memory.get("top1LiftVsLookup", 0.0),
        },
        "cases": list(report.get("cases") or []),
        "warnings": [],
    }


def export_claim_synthesis_eval_template(
    sqlite_db,
    config: Any,
    *,
    out_path: str | Path,
    claim_ids: list[str] | None = None,
    paper_ids: list[str] | None = None,
    task: str = "",
    dataset: str = "",
    metric: str = "",
    limit: int = 200,
) -> dict[str, Any]:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    synthesis = ClaimSynthesisService(sqlite_db, config).synthesize(
        claim_ids=list(claim_ids or []),
        paper_ids=list(paper_ids or []),
        task=str(task or "").strip(),
        dataset=str(dataset or "").strip(),
        metric=str(metric or "").strip(),
        limit=max(1, int(limit)),
    )
    rows: list[dict[str, Any]] = []
    for item in list(synthesis.get("comparisonReport") or []):
        rows.append(
            {
                "item_type": "comparison_report",
                "item_id": str(item.get("reportId") or ""),
                "task": str(item.get("task") or ""),
                "metric": str(item.get("metric") or ""),
                "reason": str(item.get("reason") or ""),
                "summary": str(item.get("summary") or ""),
                "evidence_refs": " | ".join(str(claim.get("claimId") or "") for claim in list(item.get("claims") or [])[:5]),
                "label": "",
                "notes": "",
            }
        )
    for item in list(synthesis.get("commonLimitationSummary") or []):
        rows.append(
            {
                "item_type": "limitation_summary",
                "item_id": str(item.get("summaryId") or ""),
                "task": "",
                "metric": "",
                "reason": "limitation_summary",
                "summary": str(item.get("summary") or item.get("limitation") or ""),
                "evidence_refs": " | ".join(str(ev.get("claimId") or "") for ev in list(item.get("evidence") or [])[:5]),
                "label": "",
                "notes": "",
            }
        )
    for item in list(synthesis.get("conflictExplanations") or []):
        rows.append(
            {
                "item_type": "conflict_explanation",
                "item_id": str(item.get("conflictId") or ""),
                "task": str(item.get("task") or ""),
                "metric": str(item.get("metric") or ""),
                "reason": str(item.get("reason") or ""),
                "summary": str(item.get("summary") or ""),
                "evidence_refs": " | ".join(str(claim.get("claimId") or "") for claim in list(item.get("claims") or [])[:5]),
                "label": "",
                "notes": "",
            }
        )

    with out_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_type", "item_id", "task", "metric", "reason", "summary", "evidence_refs", "label", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        "schema": "knowledge-hub.claim-synthesis.eval.prepare.result.v1",
        "status": "ok",
        "outPath": str(out_file),
        "rowCount": len(rows),
        "filters": {
            "task": str(task or "").strip(),
            "dataset": str(dataset or "").strip(),
            "metric": str(metric or "").strip(),
            "claimIds": list(claim_ids or []),
            "paperIds": list(paper_ids or []),
        },
        "warnings": [],
    }


def export_paper_summary_eval_template(
    sqlite_db,
    config: Any,
    *,
    out_path: str | Path,
    paper_ids: list[str] | None = None,
    paper_ids_path: str | Path | None = None,
) -> dict[str, Any]:
    selected = [str(item).strip() for item in list(paper_ids or []) if str(item).strip()]
    if paper_ids_path:
        selected.extend(_read_queries(Path(paper_ids_path)))
    ordered_paper_ids: list[str] = []
    seen: set[str] = set()
    for item in selected:
        if item in seen:
            continue
        seen.add(item)
        ordered_paper_ids.append(item)
    if not ordered_paper_ids:
        raise ValueError("No paper ids provided for paper-summary eval template")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary_service = StructuredPaperSummaryService(sqlite_db, config)
    warnings: list[str] = []
    rows: list[dict[str, str]] = []
    row_fields = [
        "paperId",
        "rawSummaryPath",
        "structuredSummaryPath",
        "baselineGeneratedAt",
        "structuredGeneratedAt",
        "parserUsed",
        "llmRoute",
        "faithfulness",
        "sectionCoverage",
        "limitationsCoverage",
        "resultSpecificity",
        "provenanceQuality",
        "overallWinner",
        "reviewNotes",
    ]

    for paper_id in ordered_paper_ids:
        paper = sqlite_db.get_paper(paper_id)
        if not paper:
            warnings.append(f"paper not found: {paper_id}")
            continue

        artifact_dir = summary_service.artifact_dir_for(paper_id=paper_id)
        summary_json_path = artifact_dir / "summary.json"
        if not summary_json_path.exists():
            build_payload = summary_service.build(
                paper_id=paper_id,
                paper_parser="auto",
                refresh_parse=False,
                quick=False,
            )
            if str(build_payload.get("status") or "") == "blocked":
                warnings.extend(str(item) for item in list(build_payload.get("warnings") or []) if str(item).strip())

        structured_payload = summary_service.load_artifact(paper_id=paper_id) or {}
        structured_generated_at = ""
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                structured_generated_at = str(manifest_payload.get("built_at") or "")
            except Exception:
                structured_generated_at = ""
        structured_path = artifact_dir / "summary.md"
        structured_summary_path = str(structured_path) if structured_path.exists() else ""
        if not structured_summary_path:
            warnings.append(f"structured summary artifact missing: {paper_id}")
        raw_payload = build_raw_summary_artifact(
            sqlite_db,
            config,
            paper_id=paper_id,
            quick=False,
            allow_external=True,
        )
        raw_summary_path = str(((raw_payload.get("paths") or {}).get("markdownPath") or "").strip())
        if not raw_summary_path:
            warnings.append(f"raw summary artifact missing: {paper_id}")

        rows.append(
            {
                "paperId": paper_id,
                "rawSummaryPath": raw_summary_path,
                "structuredSummaryPath": structured_summary_path,
                "baselineGeneratedAt": str(raw_payload.get("builtAt") or ""),
                "structuredGeneratedAt": structured_generated_at,
                "parserUsed": str(structured_payload.get("parserUsed") or ""),
                "llmRoute": str(structured_payload.get("llmRoute") or ""),
                "faithfulness": "",
                "sectionCoverage": "",
                "limitationsCoverage": "",
                "resultSpecificity": "",
                "provenanceQuality": "",
                "overallWinner": "",
                "reviewNotes": "",
            }
        )

    with out_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=row_fields,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        "schema": "knowledge-hub.paper-summary.eval.prepare.result.v1",
        "status": "ok",
        "outPath": str(out_file),
        "paperCount": len(rows),
        "rowCount": len(rows),
        "paperIds": ordered_paper_ids,
        "rowFields": row_fields,
        "warnings": list(dict.fromkeys(warnings)),
    }


def build_claim_synthesis_eval_report(csv_path: str | Path, *, label_col: str = "label", item_type_col: str = "item_type") -> dict[str, Any]:
    path = Path(csv_path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    unknown_labels: set[str] = set()
    labeled_count = 0
    positive = 0
    partial = 0
    negative = 0
    item_type_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        item_type = str(row.get(item_type_col, "")).strip()
        if item_type:
            item_type_counts[item_type] += 1
        label = _normalize_label(row.get(label_col, ""))
        if not label and str(row.get(label_col, "")).strip():
            unknown_labels.add(str(row.get(label_col, "")).strip())
            continue
        if not label:
            continue
        labeled_count += 1
        if label == "positive":
            positive += 1
        elif label == "partial":
            partial += 1
        elif label == "negative":
            negative += 1

    row_count = len(rows)
    warnings: list[str] = []
    if labeled_count < row_count:
        warnings.append(f"only {labeled_count}/{row_count} synthesis rows have labels")
    if unknown_labels:
        warnings.append(f"ignored unknown labels: {', '.join(sorted(unknown_labels))}")
    return {
        "schema": "knowledge-hub.claim-synthesis.eval.report.v1",
        "status": "ok" if labeled_count > 0 else "warning",
        "dataset": {
            "csvPath": str(path),
            "labelCol": label_col,
            "itemTypeCol": item_type_col,
        },
        "metrics": {
            "rowCount": row_count,
            "labeledRowCount": labeled_count,
            "positiveRate": _rate(positive, labeled_count),
            "partialRate": _rate(partial, labeled_count),
            "negativeRate": _rate(negative, labeled_count),
        },
        "itemTypeCounts": dict(item_type_counts),
        "warnings": warnings,
    }


def _check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    status: str,
    summary: str,
    report: str,
    observed: Any,
    threshold: str,
) -> None:
    checks.append(
        {
            "name": name,
            "status": status,
            "summary": summary,
            "report": report,
            "observed": observed,
            "threshold": threshold,
        }
    )


def _gate_status(checks: list[dict[str, Any]]) -> str:
    statuses = {str(item.get("status") or "") for item in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _gate_summary(checks: list[dict[str, Any]]) -> str:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in checks:
        status = str(item.get("status") or "pass")
        counts[status] = counts.get(status, 0) + 1
    return f"pass={counts['pass']} warn={counts['warn']} fail={counts['fail']}"


def _recommendation(profile: str, status: str) -> str:
    if status == "pass":
        return f"{profile} clears the current eval gate."
    if profile == "ask-v2":
        return "Use the ask-v2 manual scorecard to inspect source and bucket failures before treating this lane as stable."
    if profile == "memory-promotion":
        return "Keep memory surfaces as supporting capabilities until the failing checks are resolved."
    if profile == "retrieval-core":
        return "Do not treat current retrieval-core metrics as promotion-ready until the failing checks are resolved."
    return "Do not promote supporting capabilities or retrieval-core assumptions until the failing checks are resolved."


def _build_retrieval_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = dict(report.get("metrics") or {})
    runtime = dict(report.get("runtimeDiagnostics") or {})
    checks: list[dict[str, Any]] = []

    _check(
        checks,
        name="retrieval_runtime",
        status="fail" if bool(runtime.get("degraded")) else "pass",
        summary="runtime diagnostics must not be degraded",
        report="retrievalReport",
        observed=bool(runtime.get("degraded")),
        threshold="degraded == false",
    )

    labeled_query_count = int(metrics.get("labeledQueryCount", 0) or 0)
    _check(
        checks,
        name="retrieval_labeled_queries",
        status="fail" if labeled_query_count < 5 else "pass",
        summary="retrieval eval needs enough labeled queries",
        report="retrievalReport",
        observed=labeled_query_count,
        threshold=">= 5",
    )

    hit_at_3 = float(metrics.get("hitAt3", 0.0) or 0.0)
    _check(
        checks,
        name="retrieval_hit_at_3",
        status="fail" if hit_at_3 < 0.60 else "pass",
        summary="retrieval hit@3 should stay above the minimum gate",
        report="retrievalReport",
        observed=hit_at_3,
        threshold=">= 0.60",
    )

    precision = float(metrics.get("precisionAt5", 0.0) or 0.0)
    _check(
        checks,
        name="retrieval_precision_at_5",
        status="warn" if precision < 0.45 else "pass",
        summary="retrieval precision@5 is below the target range",
        report="retrievalReport",
        observed=precision,
        threshold=">= 0.45",
    )

    vault_hit_ratio = float(metrics.get("vaultHitRatio", 0.0) or 0.0)
    _check(
        checks,
        name="retrieval_vault_hit_ratio",
        status="warn" if vault_hit_ratio < 0.25 else "pass",
        summary="vault hit ratio is below the target range",
        report="retrievalReport",
        observed=vault_hit_ratio,
        threshold=">= 0.25",
    )
    return checks


def _build_document_memory_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = dict(report.get("metrics") or {})
    checks: list[dict[str, Any]] = []
    labeled_query_count = int(metrics.get("labeledQueryCount", 0) or 0)
    top3_good = float(metrics.get("top3GoodHitRate", 0.0) or 0.0)
    top1_bad = float(metrics.get("top1BadRate", 0.0) or 0.0)
    top1_good = float(metrics.get("top1GoodRate", 0.0) or 0.0)
    top1_partial = float(metrics.get("top1PartialRate", 0.0) or 0.0)

    _check(
        checks,
        name="document_memory_labeled_queries",
        status="fail" if labeled_query_count < 5 else "pass",
        summary="document-memory eval needs enough labeled queries",
        report="documentMemoryReport",
        observed=labeled_query_count,
        threshold=">= 5",
    )
    _check(
        checks,
        name="document_memory_top3_good_hit_rate",
        status="fail" if top3_good < 0.60 else "pass",
        summary="document-memory top3 good-hit rate is below the minimum gate",
        report="documentMemoryReport",
        observed=top3_good,
        threshold=">= 0.60",
    )
    _check(
        checks,
        name="document_memory_top1_bad_rate",
        status="fail" if top1_bad > 0.50 else "pass",
        summary="document-memory top1 bad rate is too high",
        report="documentMemoryReport",
        observed=top1_bad,
        threshold="<= 0.50",
    )
    _check(
        checks,
        name="document_memory_top1_good_rate",
        status="warn" if top1_good < 0.35 else "pass",
        summary="document-memory top1 good rate is below the target range",
        report="documentMemoryReport",
        observed=top1_good,
        threshold=">= 0.35",
    )
    _check(
        checks,
        name="document_memory_top1_partial_rate",
        status="warn" if top1_partial > 0.30 else "pass",
        summary="document-memory top1 partial rate is above the target range",
        report="documentMemoryReport",
        observed=top1_partial,
        threshold="<= 0.30",
    )
    return checks


def _build_paper_memory_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = dict(report.get("metrics") or {})
    checks: list[dict[str, Any]] = []
    case_count = int(report.get("caseCount", 0) or 0)
    top1_match = float(metrics.get("top1MatchRate", 0.0) or 0.0)
    no_result_rate = float(metrics.get("noResultRate", 0.0) or 0.0)
    weak_card_rate = float(metrics.get("weakCardRate", 0.0) or 0.0)
    lift_search = float(metrics.get("top1LiftVsSearchPapers", 0.0) or 0.0)
    lift_lookup = float(metrics.get("top1LiftVsLookup", 0.0) or 0.0)

    _check(
        checks,
        name="paper_memory_case_count",
        status="fail" if case_count < 3 else "pass",
        summary="paper-memory eval needs enough cases",
        report="paperMemoryReport",
        observed=case_count,
        threshold=">= 3",
    )
    _check(
        checks,
        name="paper_memory_top1_match_rate",
        status="fail" if top1_match < 0.70 else "pass",
        summary="paper-memory top1 match rate is below the minimum gate",
        report="paperMemoryReport",
        observed=top1_match,
        threshold=">= 0.70",
    )
    _check(
        checks,
        name="paper_memory_no_result_rate",
        status="fail" if no_result_rate > 0.20 else "pass",
        summary="paper-memory no-result rate is too high",
        report="paperMemoryReport",
        observed=no_result_rate,
        threshold="<= 0.20",
    )
    _check(
        checks,
        name="paper_memory_weak_card_rate",
        status="warn" if weak_card_rate > 0.20 else "pass",
        summary="paper-memory weak-card rate is above the target range",
        report="paperMemoryReport",
        observed=weak_card_rate,
        threshold="<= 0.20",
    )
    _check(
        checks,
        name="paper_memory_baseline_lift",
        status="warn" if lift_search <= 0.0 and lift_lookup <= 0.0 else "pass",
        summary="paper-memory should improve top1 retrieval over current baselines",
        report="paperMemoryReport",
        observed={"vsSearchPapers": lift_search, "vsLookup": lift_lookup},
        threshold="one of the lifts must be > 0",
    )
    return checks


def _build_claim_synthesis_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = dict(report.get("metrics") or {})
    checks: list[dict[str, Any]] = []
    labeled_rows = int(metrics.get("labeledRowCount", 0) or 0)
    negative_rate = float(metrics.get("negativeRate", 0.0) or 0.0)
    positive_rate = float(metrics.get("positiveRate", 0.0) or 0.0)
    _check(
        checks,
        name="claim_synthesis_labeled_rows",
        status="fail" if labeled_rows < 3 else "pass",
        summary="claim-synthesis eval needs enough labeled rows",
        report="claimSynthesisReport",
        observed=labeled_rows,
        threshold=">= 3",
    )
    _check(
        checks,
        name="claim_synthesis_negative_rate",
        status="fail" if negative_rate > 0.30 else "pass",
        summary="claim-synthesis negative rate is too high",
        report="claimSynthesisReport",
        observed=negative_rate,
        threshold="<= 0.30",
    )
    _check(
        checks,
        name="claim_synthesis_positive_rate",
        status="warn" if positive_rate < 0.40 else "pass",
        summary="claim-synthesis positive rate is below the target range",
        report="claimSynthesisReport",
        observed=positive_rate,
        threshold=">= 0.40",
    )
    return checks


def _build_memory_router_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = dict(report.get("metrics") or {})
    baseline_metrics = dict(report.get("baselineMetrics") or {})
    delta = dict(report.get("deltaVsBaseline") or {})
    checks: list[dict[str, Any]] = []

    labeled_query_count = int(metrics.get("labeledQueryCount", 0) or 0)
    _check(
        checks,
        name="memory_router_labeled_queries",
        status="fail" if labeled_query_count < 5 else "pass",
        summary="memory-router eval needs enough labeled queries",
        report="memoryRouterReport",
        observed=labeled_query_count,
        threshold=">= 5",
    )

    top1_delta = float(delta.get("top1GoodRate", 0.0) or 0.0)
    _check(
        checks,
        name="memory_router_top1_good_delta",
        status="fail" if top1_delta < 0.05 else "pass",
        summary="memory-router top1 good-hit rate should improve over baseline",
        report="memoryRouterReport",
        observed=top1_delta,
        threshold=">= 0.05",
    )

    no_result_delta = float(delta.get("noResultRate", 0.0) or 0.0)
    _check(
        checks,
        name="memory_router_no_result_delta",
        status="fail" if no_result_delta > 0.02 else "pass",
        summary="memory-router must not degrade no-result rate materially",
        report="memoryRouterReport",
        observed=no_result_delta,
        threshold="<= 0.02",
    )

    temporal_query_count = int(metrics.get("temporalQueryCount", 0) or 0)
    baseline_wrong_era = float(baseline_metrics.get("wrongEraHitRate", 0.0) or 0.0)
    wrong_era_reduction = float(delta.get("wrongEraReductionRate", 0.0) or 0.0)
    if temporal_query_count <= 0 or baseline_wrong_era <= 0.0:
        status = "warn"
        summary = "memory-router temporal reduction could not be evaluated from the provided dataset"
        observed: Any = {
            "temporalQueryCount": temporal_query_count,
            "baselineWrongEraHitRate": baseline_wrong_era,
        }
        threshold = "temporal queries > 0 and baseline wrong-era rate > 0"
    else:
        status = "fail" if wrong_era_reduction < 0.50 else "pass"
        summary = "memory-router temporal queries should reduce wrong-era hits"
        observed = wrong_era_reduction
        threshold = ">= 0.50"
    _check(
        checks,
        name="memory_router_wrong_era_reduction",
        status=status,
        summary=summary,
        report="memoryRouterReport",
        observed=observed,
        threshold=threshold,
    )
    return checks


def _build_ask_v2_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = dict(report.get("metrics") or {})
    by_source = dict(report.get("bySource") or {})
    checks: list[dict[str, Any]] = []

    labeled_query_count = int(metrics.get("labeledQueryCount", 0) or 0)
    _check(
        checks,
        name="ask_v2_labeled_queries",
        status="warn" if labeled_query_count < 8 else "pass",
        summary="ask-v2 manual gate needs enough labeled rows to be actionable",
        report="askV2Report",
        observed=labeled_query_count,
        threshold=">= 8",
    )

    pass_rate = float(metrics.get("passRate", 0.0) or 0.0)
    _check(
        checks,
        name="ask_v2_pass_rate",
        status="warn" if pass_rate < 0.70 else "pass",
        summary="ask-v2 pass rate is below the current manual-review target",
        report="askV2Report",
        observed=pass_rate,
        threshold=">= 0.70",
    )

    wrong_source_rate = float(metrics.get("wrongSourceRate", 0.0) or 0.0)
    _check(
        checks,
        name="ask_v2_wrong_source_rate",
        status="warn" if wrong_source_rate > 0.20 else "pass",
        summary="ask-v2 wrong-source rate is above the manual-review target",
        report="askV2Report",
        observed=wrong_source_rate,
        threshold="<= 0.20",
    )

    wrong_era_rate = float(metrics.get("wrongEraRate", 0.0) or 0.0)
    _check(
        checks,
        name="ask_v2_wrong_era_rate",
        status="warn" if wrong_era_rate > 0.20 else "pass",
        summary="ask-v2 temporal wrong-era rate is above the manual-review target",
        report="askV2Report",
        observed=wrong_era_rate,
        threshold="<= 0.20",
    )

    abstain_correct_rate = float(metrics.get("abstainCorrectRate", 0.0) or 0.0)
    abstention_reviewed = int(metrics.get("abstentionReviewedCount", 0) or 0)
    _check(
        checks,
        name="ask_v2_abstention_correctness",
        status="warn" if abstention_reviewed > 0 and abstain_correct_rate < 0.75 else "pass",
        summary="ask-v2 abstention correctness is below the current manual-review target",
        report="askV2Report",
        observed={"rate": abstain_correct_rate, "reviewed": abstention_reviewed},
        threshold=">= 0.75 when abstention rows are reviewed",
    )

    weak_without_fallback = float(metrics.get("weakEvidenceWithoutFallbackRate", 0.0) or 0.0)
    _check(
        checks,
        name="ask_v2_weak_without_fallback_rate",
        status="warn" if weak_without_fallback > 0.20 else "pass",
        summary="ask-v2 weak-evidence answers should usually degrade to fallback",
        report="askV2Report",
        observed=weak_without_fallback,
        threshold="<= 0.20",
    )

    project_metrics = dict((by_source.get("project") or {}).get("metrics") or {})
    project_wrong_source_rate = float(project_metrics.get("wrongSourceRate", 0.0) or 0.0)
    project_labeled = int(project_metrics.get("labeledQueryCount", 0) or 0)
    _check(
        checks,
        name="ask_v2_project_wrong_source_rate",
        status="warn" if project_labeled > 0 and project_wrong_source_rate > 0.20 else "pass",
        summary="project ask-v2 rows should not frequently route to the wrong source surface",
        report="askV2Report",
        observed={"rate": project_wrong_source_rate, "labeled": project_labeled},
        threshold="<= 0.20 when labeled project rows exist",
    )
    return checks


def build_eval_gate_result(
    *,
    config: Any,
    sqlite_db,
    profile: str,
    retrieval_csv: str | None = None,
    retrieval_label_col: str = "label_context",
    retrieval_source_col: str = "context_source",
    document_memory_csv: str = "docs/experiments/document_memory_eval_template.csv",
    paper_memory_cases: str = "tests/fixtures/paper_memory_eval/cases.json",
    claim_synthesis_csv: str | None = None,
    memory_router_csv: str | None = None,
    memory_router_baseline_csv: str | None = None,
    memory_router_label_col: str = "label",
    memory_router_no_result_col: str = "no_result",
    memory_router_temporal_col: str = "temporal_query",
    memory_router_wrong_era_col: str = "wrong_era",
    ask_v2_csv: str | None = None,
    ask_v2_label_col: str = "label",
    ask_v2_wrong_source_col: str = "wrong_source",
    ask_v2_no_result_col: str = "no_result",
    ask_v2_should_abstain_col: str = "should_abstain",
    ask_v2_wrong_era_col: str = "wrong_era",
    searcher: Any = None,
    searcher_error: str = "",
    runtime_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_name = str(profile or "").strip() or "all"
    if profile_name not in {"retrieval-core", "memory-promotion", "memory-router-v1", "ask-v2", "all"}:
        raise ValueError(f"unsupported eval profile: {profile_name}")

    reports: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    if profile_name in {"retrieval-core", "memory-router-v1", "all"}:
        if not str(retrieval_csv or "").strip():
            raise ValueError("--retrieval-csv is required for retrieval-core and all profiles")
        runtime = dict(runtime_diagnostics or build_runtime_diagnostics(config, searcher=searcher, searcher_error=searcher_error))
        retrieval_report = build_eval_report(
            retrieval_csv,
            label_col=retrieval_label_col,
            source_col=retrieval_source_col,
            runtime_diagnostics=runtime,
        )
        reports["retrievalReport"] = retrieval_report
        warnings.extend(str(item) for item in (retrieval_report.get("warnings") or []))
        checks.extend(_build_retrieval_checks(retrieval_report))

    if profile_name in {"memory-promotion", "memory-router-v1", "all"}:
        document_memory_report = build_document_memory_eval_report(document_memory_csv)
        paper_memory_report = build_paper_memory_eval_report(sqlite_db, paper_memory_cases)
        reports["documentMemoryReport"] = document_memory_report
        reports["paperMemoryReport"] = paper_memory_report
        warnings.extend(str(item) for item in (document_memory_report.get("warnings") or []))
        warnings.extend(str(item) for item in (paper_memory_report.get("warnings") or []))
        checks.extend(_build_document_memory_checks(document_memory_report))
        checks.extend(_build_paper_memory_checks(paper_memory_report))
        if str(claim_synthesis_csv or "").strip():
            claim_synthesis_report = build_claim_synthesis_eval_report(str(claim_synthesis_csv))
            reports["claimSynthesisReport"] = claim_synthesis_report
            warnings.extend(str(item) for item in (claim_synthesis_report.get("warnings") or []))
            checks.extend(_build_claim_synthesis_checks(claim_synthesis_report))

    if profile_name == "memory-router-v1":
        if not str(memory_router_csv or "").strip():
            raise ValueError("--memory-router-csv is required for memory-router-v1 profile")
        if not str(memory_router_baseline_csv or "").strip():
            raise ValueError("--memory-router-baseline-csv is required for memory-router-v1 profile")
        memory_router_report = build_memory_router_eval_report(
            str(memory_router_csv),
            baseline_csv=str(memory_router_baseline_csv),
            label_col=str(memory_router_label_col),
            no_result_col=str(memory_router_no_result_col),
            temporal_col=str(memory_router_temporal_col),
            wrong_era_col=str(memory_router_wrong_era_col),
        )
        reports["memoryRouterReport"] = memory_router_report
        warnings.extend(str(item) for item in (memory_router_report.get("warnings") or []))
        checks.extend(_build_memory_router_checks(memory_router_report))

    if profile_name == "ask-v2" or (profile_name == "all" and str(ask_v2_csv or "").strip()):
        if profile_name == "ask-v2" and not str(ask_v2_csv or "").strip():
            raise ValueError("--ask-v2-csv is required for ask-v2 profile")
        ask_v2_report = build_ask_v2_eval_report(
            str(ask_v2_csv),
            label_col=str(ask_v2_label_col),
            wrong_source_col=str(ask_v2_wrong_source_col),
            no_result_col=str(ask_v2_no_result_col),
            should_abstain_col=str(ask_v2_should_abstain_col),
            wrong_era_col=str(ask_v2_wrong_era_col),
        )
        reports["askV2Report"] = ask_v2_report
        warnings.extend(str(item) for item in (ask_v2_report.get("warnings") or []))
        checks.extend(_build_ask_v2_checks(ask_v2_report))

    deduped_warnings: list[str] = []
    for warning in warnings:
        if warning not in deduped_warnings:
            deduped_warnings.append(warning)

    gate_status = _gate_status(checks)
    payload = {
        "schema": "knowledge-hub.eval.gate.result.v1",
        "status": gate_status,
        "profile": profile_name,
        "gate": {
            "status": gate_status,
            "summary": _gate_summary(checks),
            "checks": checks,
            "recommendation": _recommendation(profile_name, gate_status),
        },
        "reports": reports,
        "warnings": deduped_warnings,
    }
    return payload


__all__ = [
    "build_claim_synthesis_eval_report",
    "build_ask_v2_eval_report",
    "build_document_memory_eval_report",
    "build_eval_gate_result",
    "build_memory_router_eval_report",
    "build_paper_memory_eval_report",
    "export_claim_synthesis_eval_template",
    "export_document_memory_eval_template",
    "export_paper_summary_eval_template",
]

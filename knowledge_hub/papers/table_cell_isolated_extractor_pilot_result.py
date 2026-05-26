"""Report-only isolated TableCell extractor pilot result.

This helper can run an optional extractor only when explicitly approved and
already importable in the active environment.  It never installs packages.  The
output remains diagnostic-only: no extractor choice, verified cell pairing,
source spans, table-cell evidence, parser routing, SQLite/index mutation, or
canonical parsed-artifact writes are created here.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable


TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-isolated-extractor-pilot-result.v1"
)
TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-isolated-extractor-pilot-plan.v1"
)

ExtractorProbeLoader = Callable[[str | Path, int], dict[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _bbox(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except Exception:
            return None
    return None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _default_pdfplumber_probe(source_pdf: str | Path, page_number: int) -> dict[str, Any]:
    try:
        import pdfplumber  # type: ignore
    except Exception as exc:
        return {"status": "blocked_extractor_unavailable", "failureReason": str(exc), "tables": []}
    path = Path(str(source_pdf)).expanduser()
    if not path.exists():
        return {"status": "blocked_source_pdf_missing", "failureReason": str(path), "tables": []}
    try:
        with pdfplumber.open(str(path)) as document:
            if page_number < 1 or page_number > len(document.pages):
                return {"status": "blocked_page_out_of_range", "failureReason": str(page_number), "tables": []}
            page = document.pages[page_number - 1]
            tables = []
            for index, table in enumerate(page.find_tables() or [], start=1):
                rows = table.extract() or []
                cells = [_bbox(cell) for cell in list(getattr(table, "cells", []) or [])]
                tables.append(
                    {
                        "table_index": index,
                        "bbox": _bbox(getattr(table, "bbox", None)),
                        "row_count": len(rows),
                        "column_count": max((len(row or []) for row in rows), default=0),
                        "cell_bbox_count": sum(1 for cell in cells if cell is not None),
                        "cell_bboxes_sample": [cell for cell in cells[:8] if cell is not None],
                        "extracted_rows": [[str(cell or "") for cell in list(row or [])] for row in rows],
                    }
                )
            return {"status": "ok", "failureReason": "", "tables": tables}
    except Exception as exc:
        return {"status": "failed_extractor_probe", "failureReason": str(exc), "tables": []}


def _schema_violations(plan: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if plan.get("schema") != TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID:
        violations.append("table_cell_isolated_extractor_pilot_plan_schema_mismatch")
    if plan.get("status") not in {"approval_required"}:
        violations.append("table_cell_isolated_extractor_pilot_plan_not_approval_required")
    return violations


def _unsafe_plan_flags(plan: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(plan.get("counts") or {})
    gate = dict(plan.get("gate") or {})
    policy = dict(plan.get("policy") or {})
    for key in (
        "plannedPilotRuns",
        "packagesInstalled",
        "globalPackagesInstalled",
        "tableCellEvidenceCreatedRows",
        "tableCellCitationGradeRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"{key}_nonzero")
    for key in (
        "pilotExecuted",
        "extractorChoiceMade",
        "cellBboxTextPairingVerified",
        "cellSourceSpansCreated",
        "cellSourceHashLinked",
        "tableCellEvidenceReady",
        "tableCellCitationGradeReady",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            flags.append(f"{key}_true")
    for key in (
        "globalPackageInstall",
        "isolatedPackageInstall",
        "pilotRunExecuted",
        "tableCellEvidenceCreated",
        "tableCellCitationGradeEvidenceCreated",
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(policy.get(key)):
            flags.append(f"{key}_true")
    return list(dict.fromkeys(flags))


def _selected_table(tables: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tables:
        return None
    return max(tables, key=lambda item: (_safe_int(item.get("cell_bbox_count")), _safe_int(item.get("row_count"))))


def _cell_text_count(table: dict[str, Any] | None) -> int:
    if not table:
        return 0
    count = 0
    for row in list(table.get("extracted_rows") or []):
        if isinstance(row, list):
            count += sum(1 for cell in row if _clean_text(cell))
    return count


def _result_row(
    index: int,
    target: dict[str, Any],
    *,
    extractor: str,
    approved_to_run: bool,
    can_run: bool,
    blocked_reason: str,
    probe_loader: ExtractorProbeLoader,
) -> dict[str, Any]:
    source_pdf = str(target.get("source_pdf_path") or "")
    page = _safe_int(target.get("page"))
    probe = {"status": "not_run_approval_required", "failureReason": "", "tables": []}
    probe_attempted = False
    if approved_to_run and can_run and source_pdf and page > 0:
        probe_attempted = True
        probe = probe_loader(source_pdf, page)
    elif approved_to_run and not can_run:
        probe = {"status": blocked_reason, "failureReason": blocked_reason, "tables": []}
    tables = [dict(item) for item in list(probe.get("tables") or []) if isinstance(item, dict)]
    selected = _selected_table(tables)
    status = str(probe.get("status") or "")
    if status == "ok" and not tables:
        status = "no_tables_detected"
    elif status == "ok" and selected and _safe_int(selected.get("cell_bbox_count")) > 0:
        status = "cell_bbox_candidates_detected_non_strict"
    elif status == "ok":
        status = "tables_detected_without_cell_bboxes"
    return {
        "result_row_id": f"table-cell-isolated-extractor-pilot-result:{index:04d}",
        "paper_id": str(target.get("paper_id") or ""),
        "table_label": _clean_text(target.get("table_label")),
        "page": page or None,
        "source_pdf_path": source_pdf,
        "source_pdf_exists": bool(target.get("source_pdf_exists")),
        "source_table_region_candidate_id": str(target.get("source_table_region_candidate_id") or ""),
        "extractor": extractor,
        "approved_to_run": approved_to_run,
        "probe_attempted": probe_attempted,
        "probe_status": status,
        "probe_failure_reason": str(probe.get("failureReason") or ""),
        "detected_table_count": len(tables),
        "selected_table_index": _safe_int((selected or {}).get("table_index")),
        "selected_table_bbox": (selected or {}).get("bbox"),
        "selected_table_row_count": _safe_int((selected or {}).get("row_count")),
        "selected_table_column_count": _safe_int((selected or {}).get("column_count")),
        "selected_table_cell_bbox_count": _safe_int((selected or {}).get("cell_bbox_count")),
        "selected_table_cell_text_count": _cell_text_count(selected),
        "sample_cell_bboxes": list((selected or {}).get("cell_bboxes_sample") or []),
        "cell_bbox_text_pairing_verified": False,
        "cell_source_spans_created": 0,
        "cell_source_hash_linkages_created": 0,
        "table_cell_evidence_created": False,
        "table_cell_citation_grade": False,
        "evidence_tier": "table_cell_isolated_extractor_pilot_diagnostic_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": [
            "isolated_extractor_pilot_result_only",
            "cell_bbox_text_pairing_not_verified",
            "cell_source_spans_not_created",
            "table_cell_source_hash_linkage_not_created",
            "strict_promotion_requires_explicit_later_tranche",
            status,
        ],
        "non_strict_reason": "extractor output is diagnostic only and is not linked to verified cell source spans",
    }


def _counts(rows: list[dict[str, Any]], schema_issues: list[str], unsafe_flags: list[str]) -> dict[str, Any]:
    return {
        "targetRows": len(rows),
        "probeAttemptedRows": sum(1 for item in rows if item.get("probe_attempted")),
        "approvalRequiredRows": sum(1 for item in rows if item.get("probe_status") == "not_run_approval_required"),
        "blockedRows": sum(1 for item in rows if str(item.get("probe_status") or "").startswith("blocked")),
        "tableDetectedRows": sum(1 for item in rows if _safe_int(item.get("detected_table_count")) > 0),
        "cellBboxCandidateRows": sum(1 for item in rows if _safe_int(item.get("selected_table_cell_bbox_count")) > 0),
        "selectedTableCellBboxCandidates": sum(_safe_int(item.get("selected_table_cell_bbox_count")) for item in rows),
        "selectedTableCellTextCandidates": sum(_safe_int(item.get("selected_table_cell_text_count")) for item in rows),
        "cellBboxTextPairingVerifiedRows": 0,
        "cellSourceSpanCreatedRows": 0,
        "cellSourceHashLinkedRows": 0,
        "tableCellEvidenceCreatedRows": 0,
        "tableCellCitationGradeRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_issues),
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byProbeStatus": dict(Counter(str(item.get("probe_status") or "") for item in rows)),
    }


def build_table_cell_isolated_extractor_pilot_result(
    *,
    table_cell_isolated_extractor_pilot_plan: str | Path,
    extractor: str = "pdfplumber",
    approved_to_run: bool = False,
    extractor_probe_loader: ExtractorProbeLoader | None = None,
    extractor_available: bool | None = None,
) -> dict[str, Any]:
    plan_path = Path(str(table_cell_isolated_extractor_pilot_plan)).expanduser()
    plan = _read_json(plan_path)
    schema_issues = _schema_violations(plan)
    unsafe_flags = _unsafe_plan_flags(plan)
    targets = [dict(item) for item in list(plan.get("targetRows") or []) if isinstance(item, dict)]
    supported = extractor == "pdfplumber"
    available = _module_available(extractor) if extractor_available is None else extractor_available
    can_run = approved_to_run and supported and available and not schema_issues and not unsafe_flags
    if not supported:
        blocked_reason = "blocked_unsupported_extractor"
    elif not available:
        blocked_reason = "blocked_extractor_unavailable"
    elif schema_issues or unsafe_flags:
        blocked_reason = "blocked_input_plan"
    else:
        blocked_reason = ""
    loader = extractor_probe_loader or _default_pdfplumber_probe
    rows = [
        _result_row(
            index,
            target,
            extractor=extractor,
            approved_to_run=approved_to_run,
            can_run=can_run,
            blocked_reason=blocked_reason,
            probe_loader=loader,
        )
        for index, target in enumerate(targets, start=1)
    ]
    counts = _counts(rows, schema_issues, unsafe_flags)
    status = (
        "pilot_complete_non_strict"
        if approved_to_run and can_run and counts["probeAttemptedRows"] > 0
        else ("blocked" if approved_to_run else "approval_required")
    )
    return {
        "schema": TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "tableCellIsolatedExtractorPilotPlan": str(plan_path),
            "tableCellIsolatedExtractorPilotPlanSchema": str(plan.get("schema") or ""),
            "extractor": extractor,
            "approvedToRun": approved_to_run,
        },
        "counts": counts,
        "gate": {
            "pilotExecuted": bool(counts["probeAttemptedRows"] > 0),
            "approvalRequiredBeforeInstallOrRun": not approved_to_run,
            "extractorAvailable": available,
            "extractorChoiceMade": False,
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "cellSourceHashLinked": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": (
                "isolated_extractor_pilot_complete_non_strict"
                if status == "pilot_complete_non_strict"
                else ("blocked" if status == "blocked" else "awaiting_explicit_approval")
            ),
            "schemaViolations": schema_issues,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "review_isolated_extractor_pilot_result_before_any_cell_evidence_design",
        },
        "policy": {
            "reportOnly": True,
            "pilotOnly": True,
            "packageInstallAttempted": False,
            "globalPackageInstall": False,
            "isolatedPackageInstall": False,
            "extractorChoiceMade": False,
            "tableCellEvidenceCreated": False,
            "tableCellCitationGradeEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "pilot_result_is_not_table_cell_evidence",
            "cell_bboxes_without_verified_text_pairing_and_source_spans_are_not_citation_grade",
            "no_packages_are_installed_by_this_helper",
            "no_strict_evidence_or_parser_routing_is_created",
        ],
        "resultRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings", "resultRows")
        if key in report
    }


def render_table_cell_isolated_extractor_pilot_result_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    return "\n".join(
        [
            "# TableCell Isolated Extractor Pilot Result",
            "",
            f"- Status: `{report.get('status', '')}`",
            f"- Decision: `{gate.get('decision', '')}`",
            f"- Probe attempted rows: `{counts.get('probeAttemptedRows', 0)}`",
            f"- Approval required rows: `{counts.get('approvalRequiredRows', 0)}`",
            f"- Blocked rows: `{counts.get('blockedRows', 0)}`",
            f"- Cell bbox candidate rows: `{counts.get('cellBboxCandidateRows', 0)}`",
            f"- Strict eligible rows: `{counts.get('strictEligibleRows', 0)}`",
            "",
            "## Boundary",
            "",
            "This result is diagnostic-only. It does not install packages, choose an extractor for production, verify cell/text pairing, create source spans, create table-cell evidence, route parsers, or change answer runtime behavior.",
            "",
            f"- By probe status: `{json.dumps(counts.get('byProbeStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
            "",
        ]
    )


def write_table_cell_isolated_extractor_pilot_result_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "table-cell-isolated-extractor-pilot-result.json"
    summary_path = root / "table-cell-isolated-extractor-pilot-result-summary.json"
    markdown_path = root / "table-cell-isolated-extractor-pilot-result.md"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_cell_isolated_extractor_pilot_result_markdown(report), encoding="utf-8")
    return {"result": str(result_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only TableCell isolated extractor pilot result.")
    parser.add_argument("--table-cell-isolated-extractor-pilot-plan", required=True)
    parser.add_argument("--extractor", default="pdfplumber")
    parser.add_argument("--approved-to-run", action="store_true")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_isolated_extractor_pilot_result(
        table_cell_isolated_extractor_pilot_plan=args.table_cell_isolated_extractor_pilot_plan,
        extractor=args.extractor,
        approved_to_run=args.approved_to_run,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_cell_isolated_extractor_pilot_result_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID",
    "build_table_cell_isolated_extractor_pilot_result",
    "render_table_cell_isolated_extractor_pilot_result_markdown",
    "write_table_cell_isolated_extractor_pilot_result_reports",
]

"""Report-only isolated TableCell extractor pilot plan.

This helper turns the current TableCell next-action gate into a reproducible
operator plan for an isolated dependency pilot.  It does not create a venv,
install packages, run extractors, choose a parser, verify cell/text pairing,
create source spans, create table-cell evidence, route parsers, mutate SQLite,
reindex, reembed, or write canonical parsed artifacts.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-isolated-extractor-pilot-plan.v1"
)
TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID = "knowledge-hub.paper.table-cell-next-action-gate.v1"
TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-pymupdf-pairing-diagnostic.v1"
)


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


def _module_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {"available": bool(spec), "origin": str(getattr(spec, "origin", "") or "") if spec else ""}


def _command_status(command: str, args: list[str], *, timeout_seconds: int = 5) -> dict[str, Any]:
    path = shutil.which(command) or ""
    if not path:
        return {"available": False, "path": "", "returncode": None, "output": ""}
    try:
        result = subprocess.run(
            [path, *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return {"available": False, "path": path, "returncode": None, "output": str(exc)}
    return {
        "available": result.returncode == 0,
        "path": path,
        "returncode": result.returncode,
        "output": "\n".join(str(result.stdout or "").splitlines()[:3]),
    }


def _default_environment() -> dict[str, Any]:
    python_version = subprocess.run(
        ["python", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=5,
    ).stdout.strip()
    return {
        "pythonVersion": python_version,
        "venvAvailable": True,
        "modules": {
            "pdfplumber": _module_status("pdfplumber"),
            "camelot": _module_status("camelot"),
            "tabula": _module_status("tabula"),
            "pandas": _module_status("pandas"),
            "fitz": _module_status("fitz"),
        },
        "systemTools": {
            "java": _command_status("java", ["-version"]),
            "ghostscript": _command_status("gs", ["--version"]),
            "pdftoppm": _command_status("pdftoppm", ["-v"]),
            "tesseract": _command_status("tesseract", ["--version"]),
        },
    }


def _schema_violations(next_action: dict[str, Any], diagnostic: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if next_action.get("schema") != TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID:
        violations.append("table_cell_next_action_gate_schema_mismatch")
    if next_action.get("status") != "next_action_ready":
        violations.append("table_cell_next_action_gate_not_ready")
    if diagnostic.get("schema") != TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID:
        violations.append("table_cell_pymupdf_pairing_diagnostic_schema_mismatch")
    if diagnostic.get("status") != "diagnostic_ready":
        violations.append("table_cell_pymupdf_pairing_diagnostic_not_ready")
    return violations


def _unsafe_violations(report: dict[str, Any], *, prefix: str) -> list[str]:
    violations: list[str] = []
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    policy = dict(report.get("policy") or {})
    for key in (
        "cellBboxTextPairingVerifiedRows",
        "cellSourceSpanCreatedRows",
        "cellSourceHashLinkedRows",
        "tableCellEvidenceCreatedRows",
        "tableCellCitationGradeRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            violations.append(f"{prefix}_{key}_nonzero")
    for key in (
        "cellBboxTextPairingVerified",
        "cellSourceSpansCreated",
        "cellSourceHashLinked",
        "extractorChoiceMade",
        "tableCellEvidenceReady",
        "tableCellCitationGradeReady",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            violations.append(f"{prefix}_{key}_true")
    for key in (
        "extractorChoiceMade",
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
            violations.append(f"{prefix}_{key}_true")
    return list(dict.fromkeys(violations))


def _target_rows(diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(diagnostic.get("diagnosticRows") or []):
        if not isinstance(item, dict):
            continue
        source_pdf = Path(str(item.get("source_pdf_path") or "")).expanduser()
        rows.append(
            {
                "paper_id": str(item.get("paper_id") or ""),
                "table_label": str(item.get("table_label") or ""),
                "page": item.get("page") if isinstance(item.get("page"), int) else None,
                "source_pdf_path": str(source_pdf) if str(source_pdf) != "." else "",
                "source_pdf_exists": bool(str(source_pdf) and source_pdf.exists()),
                "source_table_region_candidate_id": str(item.get("source_table_region_candidate_id") or ""),
                "selected_table_bbox": item.get("selected_table_bbox"),
                "selected_table_cell_bbox_count": _safe_int(item.get("cell_bbox_candidate_count")),
                "selected_table_cell_text_count": _safe_int(item.get("cell_text_candidate_count")),
                "diagnostic_unique_cell_text_matches": _safe_int(item.get("diagnostic_unique_cell_text_matches")),
                "diagnostic_ambiguous_cell_text_matches": _safe_int(item.get("diagnostic_ambiguous_cell_text_matches")),
                "diagnostic_no_match_cell_texts": _safe_int(item.get("diagnostic_no_match_cell_texts")),
                "pilot_target": True,
                "strict_eligible": False,
                "runtime_evidence": False,
            }
        )
    return rows


def _extractor_candidates(environment: dict[str, Any]) -> list[dict[str, Any]]:
    modules = dict(environment.get("modules") or {})
    java = dict(dict(environment.get("systemTools") or {}).get("java") or {})
    return [
        {
            "extractor": "pdfplumber",
            "priority": 1,
            "current_status": "available" if bool(dict(modules.get("pdfplumber") or {}).get("available")) else "missing",
            "isolated_dependency_spec": ["pdfplumber"],
            "why": "pure Python table/text geometry probe with the lowest dependency risk for the isolated pilot",
            "expected_output": "table candidates with row/column text and geometry when extractable",
            "known_limits": [
                "may still miss complex ruled tables",
                "cell bbox/source-span/source-hash linkage still requires later validation",
            ],
            "requires_global_install": False,
            "approval_required_before_install": True,
        },
        {
            "extractor": "camelot",
            "priority": 2,
            "current_status": "available" if bool(dict(modules.get("camelot") or {}).get("available")) else "missing",
            "isolated_dependency_spec": ["camelot-py[cv]"],
            "why": "table-specific detector that may recover ruled table structure better than PyMuPDF",
            "expected_output": "dataframes/table areas and possible cell boundaries depending on flavor",
            "known_limits": [
                "heavier dependencies",
                "may require ghostscript/cv stack",
                "still non-strict until source spans/hash linkage are proven",
            ],
            "requires_global_install": False,
            "approval_required_before_install": True,
        },
        {
            "extractor": "tabula",
            "priority": 3,
            "current_status": "available" if bool(dict(modules.get("tabula") or {}).get("available")) else "missing",
            "isolated_dependency_spec": ["tabula-py"],
            "why": "Java-backed table extraction comparison candidate when the Java runtime is healthy",
            "expected_output": "dataframes/table areas from selected PDF pages",
            "known_limits": [
                "local Java runtime is required",
                "JVM dependency increases operator burden",
                "still non-strict until source spans/hash linkage are proven",
            ],
            "requires_global_install": False,
            "approval_required_before_install": True,
            "blocked_locally": not bool(java.get("available")),
        },
    ]


def _proposed_commands(*, venv_path: str, output_dir: str, target_rows: list[dict[str, Any]]) -> dict[str, str]:
    pages = ",".join(str(row.get("page")) for row in target_rows if row.get("page"))
    paper_ids = sorted({str(row.get("paper_id")) for row in target_rows if row.get("paper_id")})
    paper_id = paper_ids[0] if len(paper_ids) == 1 else "<paper-id>"
    return {
        "create_venv": f"python -m venv {venv_path}",
        "activate": f"source {venv_path}/bin/activate",
        "install_pdfplumber_only": "python -m pip install pdfplumber",
        "probe_target_pages": (
            "python <report-only probe helper> "
            f"--paper-id {paper_id} --pages {pages or '<pages>'} --extractor pdfplumber --output-dir {output_dir}"
        ),
        "deactivate": "deactivate",
    }


def build_table_cell_isolated_extractor_pilot_plan(
    *,
    table_cell_next_action_gate: str | Path,
    table_cell_pymupdf_pairing_diagnostic: str | Path,
    output_dir: str | Path = "~/.khub/reports/layout-parser-pilot/2026-05-17/table-cell-isolated-extractor-pilot-result",
    venv_path: str | Path = "~/.khub/venvs/table-extractor-pilot-20260518",
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report-only isolated table-extractor pilot plan."""

    next_action_path = Path(str(table_cell_next_action_gate)).expanduser()
    diagnostic_path = Path(str(table_cell_pymupdf_pairing_diagnostic)).expanduser()
    next_action = _read_json(next_action_path)
    diagnostic = _read_json(diagnostic_path)
    environment_payload = environment if environment is not None else _default_environment()
    rows = _target_rows(diagnostic)
    modules = dict(environment_payload.get("modules") or {})
    extractor_names = ("pdfplumber", "camelot", "tabula")
    schema_issues = _schema_violations(next_action, diagnostic)
    unsafe_issues = [
        *_unsafe_violations(next_action, prefix="next_action"),
        *_unsafe_violations(diagnostic, prefix="diagnostic"),
    ]
    approval_required = bool(dict(next_action.get("gate") or {}).get("dependencyApprovalRequired"))
    source_ready = bool(rows) and all(bool(row.get("source_pdf_exists")) for row in rows)
    ready = bool(rows) and not schema_issues and not unsafe_issues and source_ready
    counts = {
        "targetTables": len(rows),
        "sourcePdfExistingRows": sum(1 for row in rows if bool(row.get("source_pdf_exists"))),
        "availableExtractorModules": sum(1 for name in extractor_names if bool(dict(modules.get(name) or {}).get("available"))),
        "missingExtractorModules": sum(1 for name in extractor_names if not bool(dict(modules.get(name) or {}).get("available"))),
        "plannedPilotRuns": 0,
        "packagesInstalled": 0,
        "globalPackagesInstalled": 0,
        "tableCellEvidenceCreatedRows": 0,
        "tableCellCitationGradeRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_issues),
        "unsafeUpstreamFlagCount": len(unsafe_issues),
    }
    status = "approval_required" if ready and approval_required else "blocked"
    return {
        "schema": TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "tableCellNextActionGate": str(next_action_path),
            "tableCellPymupdfPairingDiagnostic": str(diagnostic_path),
            "tableCellNextActionGateSchema": str(next_action.get("schema") or ""),
            "tableCellPymupdfPairingDiagnosticSchema": str(diagnostic.get("schema") or ""),
        },
        "environment": environment_payload,
        "counts": counts,
        "gate": {
            "pilotPlanReady": ready,
            "approvalRequiredBeforeInstallOrRun": True,
            "sourcePdfsReady": source_ready,
            "dependencyIsolationRequired": True,
            "globalInstallAllowed": False,
            "pilotExecuted": False,
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
            "decision": "awaiting_explicit_approval_for_isolated_pdfplumber_pilot" if status == "approval_required" else "blocked",
            "schemaViolations": schema_issues,
            "unsafeUpstreamFlags": unsafe_issues,
            "recommendedNextTranche": (
                "run_isolated_pdfplumber_table_extractor_pilot_after_approval"
                if status == "approval_required"
                else "fix_blockers_before_dependency_pilot"
            ),
        },
        "policy": {
            "reportOnly": True,
            "planOnly": True,
            "globalPackageInstall": False,
            "isolatedPackageInstall": False,
            "pilotRunExecuted": False,
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
        "targetRows": rows,
        "extractorCandidates": _extractor_candidates(environment_payload),
        "proposedCommandsAfterApproval": _proposed_commands(
            venv_path=str(Path(str(venv_path)).expanduser()),
            output_dir=str(Path(str(output_dir)).expanduser()),
            target_rows=rows,
        ),
        "successCriteria": [
            "isolated venv only; no global package changes",
            "run only the listed target paper pages",
            "write only local report outputs under the requested report directory",
            "produce row/column/cell text candidates plus any available bbox geometry",
            "keep all outputs non-strict until source span/hash linkage is independently proven",
        ],
        "stopConditions": [
            "dependency install requires global package mutation",
            "source PDF missing or hash/source linkage ambiguous",
            "extractor output cannot expose row/column/cell structure",
            "pilot would need DB/index/reembed/parser routing/canonical parsed artifact writes",
        ],
        "warnings": [
            "approval_required_before_any_dependency_install_or_pilot_run",
            "pdfplumber_is_recommended_first_for_lowest_dependency_risk",
            "planner_output_is_not_table_cell_evidence",
            "no_strict_or_runtime_evidence_created",
        ],
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "inputs",
            "environment",
            "counts",
            "gate",
            "policy",
            "targetRows",
            "extractorCandidates",
            "proposedCommandsAfterApproval",
            "successCriteria",
            "stopConditions",
            "warnings",
        )
        if key in report
    }


def render_table_cell_isolated_extractor_pilot_plan_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    commands = dict(report.get("proposedCommandsAfterApproval") or {})
    lines = [
        "# TableCell Isolated Extractor Pilot Plan",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Target tables: `{counts.get('targetTables', 0)}`",
        f"- Source PDFs ready: `{counts.get('sourcePdfExistingRows', 0)}/{counts.get('targetTables', 0)}`",
        f"- Available extractor modules: `{counts.get('availableExtractorModules', 0)}`",
        f"- Global packages installed: `{counts.get('globalPackagesInstalled', 0)}`",
        f"- Strict eligible rows: `{counts.get('strictEligibleRows', 0)}`",
        "",
        "## Boundary",
        "",
        "This is a plan-only report. It does not install packages, run extractors, mutate DB/index state, write canonical parsed artifacts, route parsers, or create strict/runtime evidence.",
        "",
        "## Proposed Commands After Explicit Approval",
        "",
    ]
    for key, value in commands.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Target Rows", ""])
    for row in list(report.get("targetRows") or []):
        lines.append(
            f"- `{row.get('paper_id')}` `{row.get('table_label')}` page `{row.get('page')}` "
            f"source exists `{bool(row.get('source_pdf_exists'))}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_table_cell_isolated_extractor_pilot_plan_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "table-cell-isolated-extractor-pilot-plan.json"
    summary_path = root / "table-cell-isolated-extractor-pilot-plan-summary.json"
    markdown_path = root / "table-cell-isolated-extractor-pilot-plan.md"
    plan_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_cell_isolated_extractor_pilot_plan_markdown(report), encoding="utf-8")
    return {"plan": str(plan_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only TableCell isolated extractor pilot plan.")
    parser.add_argument("--table-cell-next-action-gate", required=True)
    parser.add_argument("--table-cell-pymupdf-pairing-diagnostic", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--pilot-output-dir", default="~/.khub/reports/layout-parser-pilot/2026-05-17/table-cell-isolated-extractor-pilot-result")
    parser.add_argument("--venv-path", default="~/.khub/venvs/table-extractor-pilot-20260518")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_isolated_extractor_pilot_plan(
        table_cell_next_action_gate=args.table_cell_next_action_gate,
        table_cell_pymupdf_pairing_diagnostic=args.table_cell_pymupdf_pairing_diagnostic,
        output_dir=args.pilot_output_dir,
        venv_path=args.venv_path,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_cell_isolated_extractor_pilot_plan_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_PLAN_SCHEMA_ID",
    "build_table_cell_isolated_extractor_pilot_plan",
    "render_table_cell_isolated_extractor_pilot_plan_markdown",
    "write_table_cell_isolated_extractor_pilot_plan_reports",
]

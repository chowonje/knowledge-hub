#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_design import (
    PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID,
    build_blocked_pymupdf_quality_repair_design,
    build_pymupdf_quality_repair_design,
    write_pymupdf_quality_repair_design_reports,
)


DEFAULT_DEGRADATION_AUDIT_REPORT = (
    REPO_ROOT
    / "eval"
    / "knowledgeos"
    / "runs"
    / "reports"
    / "parsed_artifact_parser_quality_degradation_audit_20260524"
    / "parsed-artifact-parser-quality-degradation-audit.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "eval"
    / "knowledgeos"
    / "runs"
    / "reports"
    / "parsed_artifact_pymupdf_quality_repair_design_20260524"
)


def _read_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _relative_for_print(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return candidate.name


def build_report_from_inputs(args: argparse.Namespace) -> dict:
    try:
        degradation_audit = _read_json(args.degradation_audit_report)
    except Exception:
        return build_blocked_pymupdf_quality_repair_design(reason="degradation_audit_report_unavailable")

    return build_pymupdf_quality_repair_design(degradation_audit=degradation_audit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a report-only PyMuPDF parsed-artifact quality repair design from parser-quality audit JSON."
    )
    parser.add_argument("--degradation-audit-report", default=str(DEFAULT_DEGRADATION_AUDIT_REPORT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print the schema-backed JSON report.")
    parser.add_argument("--no-write", action="store_true", help="Do not write JSON/Markdown report files.")
    args = parser.parse_args(argv)

    report = build_report_from_inputs(args)
    validation = validate_payload(report, PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID, strict=True)
    if not validation.ok:
        report = build_blocked_pymupdf_quality_repair_design(reason="schema_validation_failed")

    if not args.no_write:
        paths = write_pymupdf_quality_repair_design_reports(report, args.output_dir)
        if not args.json:
            print(
                "wrote PyMuPDF quality repair design: "
                f"{_relative_for_print(paths['json'])}, {_relative_for_print(paths['markdown'])}"
            )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"ready", "design_ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

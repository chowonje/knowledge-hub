#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_hub.application.corpus_artifacts import load_corpus_manifest
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_parser_quality_degradation_audit import (
    PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID,
    build_blocked_parser_quality_degradation_audit,
    build_parser_quality_degradation_audit,
    write_parser_quality_degradation_audit_reports,
)


DEFAULT_COVERAGE_REPORT = (
    REPO_ROOT
    / "eval"
    / "knowledgeos"
    / "runs"
    / "reports"
    / "parsed_artifact_coverage_audit_20260524"
    / "parsed-artifact-coverage-audit.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "eval"
    / "knowledgeos"
    / "runs"
    / "reports"
    / "parsed_artifact_parser_quality_degradation_audit_20260524"
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
        coverage_report = _read_json(args.coverage_report)
    except Exception:
        return build_blocked_parser_quality_degradation_audit(reason="coverage_report_unavailable")
    try:
        corpus_manifest = load_corpus_manifest(args.corpus_manifest)
    except Exception:
        return build_blocked_parser_quality_degradation_audit(reason="corpus_manifest_unavailable")

    return build_parser_quality_degradation_audit(
        coverage_report=coverage_report,
        corpus_manifest=corpus_manifest,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a report-only parser-quality degradation audit from parsed artifact coverage JSON."
    )
    parser.add_argument("--coverage-report", default=str(DEFAULT_COVERAGE_REPORT))
    parser.add_argument(
        "--corpus-manifest",
        default=str(REPO_ROOT / "eval" / "knowledgeos" / "fixtures" / "corpus_manifest.json"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print the schema-backed JSON report.")
    parser.add_argument("--no-write", action="store_true", help="Do not write JSON/Markdown report files.")
    args = parser.parse_args(argv)

    report = build_report_from_inputs(args)
    validation = validate_payload(report, PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID, strict=True)
    if not validation.ok:
        report = build_blocked_parser_quality_degradation_audit(reason="schema_validation_failed")

    if not args.no_write:
        paths = write_parser_quality_degradation_audit_reports(report, args.output_dir)
        if not args.json:
            print(
                "wrote parser quality degradation audit: "
                f"{_relative_for_print(paths['json'])}, {_relative_for_print(paths['markdown'])}"
            )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"ready", "degradation_audit_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

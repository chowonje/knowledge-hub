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
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_sidecar_oracle_comparison import (
    PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID,
    build_blocked_pymupdf_quality_repair_sidecar_oracle_comparison,
    build_pymupdf_quality_repair_sidecar_oracle_comparison,
    write_pymupdf_quality_repair_sidecar_oracle_comparison_reports,
)


DEFAULT_EXECUTOR_DRY_RUN_REPORT = (
    REPO_ROOT
    / "eval"
    / "knowledgeos"
    / "runs"
    / "reports"
    / "parsed_artifact_pymupdf_quality_repair_executor_dry_run_20260524"
    / "parsed-artifact-pymupdf-quality-repair-executor-dry-run.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "eval"
    / "knowledgeos"
    / "runs"
    / "reports"
    / "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_comparison_20260525"
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
        executor_dry_run = _read_json(args.executor_dry_run_report)
    except Exception:
        return build_blocked_pymupdf_quality_repair_sidecar_oracle_comparison(
            reason="executor_dry_run_report_unavailable"
        )

    sidecar_oracle_pack = None
    if args.sidecar_oracle_pack:
        try:
            sidecar_oracle_pack = _read_json(args.sidecar_oracle_pack)
        except Exception:
            return build_blocked_pymupdf_quality_repair_sidecar_oracle_comparison(
                reason="sidecar_oracle_pack_unavailable"
            )

    return build_pymupdf_quality_repair_sidecar_oracle_comparison(
        executor_dry_run=executor_dry_run,
        sidecar_oracle_pack=sidecar_oracle_pack,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a report-only sidecar oracle comparison for PyMuPDF parsed-artifact repair plans."
    )
    parser.add_argument("--executor-dry-run-report", default=str(DEFAULT_EXECUTOR_DRY_RUN_REPORT))
    parser.add_argument("--sidecar-oracle-pack", default="", help="Optional already-built local oracle pack JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print the schema-backed JSON report.")
    parser.add_argument("--no-write", action="store_true", help="Do not write JSON/Markdown report files.")
    args = parser.parse_args(argv)

    report = build_report_from_inputs(args)
    validation = validate_payload(report, PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID, strict=True)
    if not validation.ok:
        report = build_blocked_pymupdf_quality_repair_sidecar_oracle_comparison(
            reason="schema_validation_failed"
        )

    if not args.no_write:
        paths = write_pymupdf_quality_repair_sidecar_oracle_comparison_reports(report, args.output_dir)
        if not args.json:
            print(
                "wrote PyMuPDF quality repair sidecar oracle comparison: "
                f"{_relative_for_print(paths['json'])}, {_relative_for_print(paths['markdown'])}"
            )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"ready", "comparison_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

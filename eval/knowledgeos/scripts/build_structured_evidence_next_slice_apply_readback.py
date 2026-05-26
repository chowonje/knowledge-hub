#!/usr/bin/env python3
"""Build/apply Structured Evidence next-slice apply/readback report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.application.corpus_artifacts import DEFAULT_CORPUS_MANIFEST_PATH  # noqa: E402
from knowledge_hub.papers.structured_evidence_next_slice_apply_readback import (  # noqa: E402
    DEFAULT_NEXT_SLICE_CANDIDATE_REPORT_PATH,
    build_structured_evidence_next_slice_apply_readback,
    render_structured_evidence_next_slice_apply_readback_markdown,
)


DEFAULT_REPORT_JSON = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / (
    "structured_evidence_next_slice_apply_readback.v1.json"
)
DEFAULT_REPORT_MD = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / (
    "structured_evidence_next_slice_apply_readback.v1.md"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST_PATH)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_NEXT_SLICE_CANDIDATE_REPORT_PATH)
    parser.add_argument("--papers-dir", type=Path)
    parser.add_argument("--selected-greenfield-rows", type=int, default=7)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write selected SourceSpan/StrictEvidence JSONL records locally, then read them back.",
    )
    args = parser.parse_args()

    payload = build_structured_evidence_next_slice_apply_readback(
        manifest_path=args.manifest,
        candidate_report_path=args.candidate_report,
        papers_dir=args.papers_dir,
        selected_greenfield_rows=args.selected_greenfield_rows,
        apply=args.apply,
    )
    schema_validation = payload.get("schemaValidation") or {}
    if not schema_validation.get("ok"):
        errors = "; ".join(str(item) for item in schema_validation.get("errors") or [])
        raise ValueError(f"next-slice apply/readback schema validation failed: {errors}")

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report_md.write_text(
        render_structured_evidence_next_slice_apply_readback_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps({"status": payload.get("status"), "counts": payload.get("counts")}, indent=2))
    return 0 if payload.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

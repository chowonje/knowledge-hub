#!/usr/bin/env python3
"""Build report-only 300-row corpus freeze and next evidence-slice reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.application.corpus_artifacts import DEFAULT_CORPUS_MANIFEST_PATH  # noqa: E402
from knowledge_hub.papers.priority_corpus_300_freeze import (  # noqa: E402
    DEFAULT_ALLOWLIST_PATH,
    DEFAULT_JOIN_REPORT_PATH,
    build_priority_corpus_300_freeze_report,
    build_structured_evidence_next_slice_candidate_report,
    render_priority_corpus_300_freeze_markdown,
    render_structured_evidence_next_slice_markdown,
)


DEFAULT_FREEZE_JSON = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / (
    "priority_corpus_300_freeze_report.v1.json"
)
DEFAULT_FREEZE_MD = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / (
    "priority_corpus_300_freeze_report.v1.md"
)
DEFAULT_NEXT_SLICE_JSON = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / (
    "structured_evidence_next_slice_candidate_report.v1.json"
)
DEFAULT_NEXT_SLICE_MD = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / (
    "structured_evidence_next_slice_candidate_report.v1.md"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST_PATH)
    parser.add_argument("--join-report", type=Path, default=DEFAULT_JOIN_REPORT_PATH)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST_PATH)
    parser.add_argument("--papers-dir", type=Path)
    parser.add_argument("--greenfield-target-rows", type=int, default=30)
    parser.add_argument("--freeze-json", type=Path, default=DEFAULT_FREEZE_JSON)
    parser.add_argument("--freeze-md", type=Path, default=DEFAULT_FREEZE_MD)
    parser.add_argument("--next-slice-json", type=Path, default=DEFAULT_NEXT_SLICE_JSON)
    parser.add_argument("--next-slice-md", type=Path, default=DEFAULT_NEXT_SLICE_MD)
    args = parser.parse_args()

    freeze = build_priority_corpus_300_freeze_report(
        manifest_path=args.manifest,
        join_report_path=args.join_report,
        allowlist_path=args.allowlist,
        papers_dir=args.papers_dir,
    )
    next_slice = build_structured_evidence_next_slice_candidate_report(
        manifest_path=args.manifest,
        join_report_path=args.join_report,
        papers_dir=args.papers_dir,
        greenfield_target_rows=args.greenfield_target_rows,
    )

    _write_json(args.freeze_json, freeze)
    _write_text(args.freeze_md, render_priority_corpus_300_freeze_markdown(freeze))
    _write_json(args.next_slice_json, next_slice)
    _write_text(args.next_slice_md, render_structured_evidence_next_slice_markdown(next_slice))

    print(
        json.dumps(
            {
                "freeze": {
                    "status": freeze.get("status"),
                    "counts": freeze.get("counts"),
                    "schemaValidation": freeze.get("schemaValidation"),
                },
                "nextSlice": {
                    "status": next_slice.get("status"),
                    "counts": next_slice.get("counts"),
                    "schemaValidation": next_slice.get("schemaValidation"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if freeze.get("status") == "locked" and next_slice.get("status") in {"ready", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

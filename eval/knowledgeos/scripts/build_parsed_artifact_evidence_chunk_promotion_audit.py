#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_hub.papers.parsed_artifact_evidence_chunk_promotion_audit import (  # noqa: E402
    DEFAULT_POSITIVE_COMPLETE_REPORT,
    build_parsed_artifact_evidence_chunk_promotion_audit,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_promotion_report_io import (  # noqa: E402
    write_parsed_artifact_evidence_chunk_promotion_audit,
)


DEFAULT_REPORT_JSON = REPO_ROOT / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_promotion_audit.v1.json"
DEFAULT_REPORT_MD = REPO_ROOT / "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_promotion_audit.v1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the parsed-artifact evidence chunk promotion audit.")
    parser.add_argument("--papers-dir", type=Path, default=None)
    parser.add_argument("--positive-complete-report", type=Path, default=REPO_ROOT / DEFAULT_POSITIVE_COMPLETE_REPORT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = {"positive_complete_report_path": args.positive_complete_report}
    if args.papers_dir is not None:
        kwargs["papers_dir"] = args.papers_dir
    report = build_parsed_artifact_evidence_chunk_promotion_audit(**kwargs)
    paths: dict[str, str] = {}
    if not args.no_write:
        paths = write_parsed_artifact_evidence_chunk_promotion_audit(
            report,
            report_json=args.report_json,
            report_md=args.report_md,
        )
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "decision": report.get("decision"),
                "nextRecommendedTranche": report.get("nextRecommendedTranche"),
                "counts": report.get("counts"),
                "gate": report.get("gate"),
                "blockersByCategory": report.get("blockersByCategory"),
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

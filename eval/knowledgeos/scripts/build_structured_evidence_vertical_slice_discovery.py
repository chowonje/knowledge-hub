#!/usr/bin/env python3
"""Build Structured Evidence vertical slice discovery report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.papers.structured_evidence_vertical_slice_discovery import (
    build_structured_evidence_vertical_slice_discovery,
    render_structured_evidence_vertical_slice_discovery_markdown,
)


class _ConfigStub:
    papers_dir = str(Path.home() / ("." + "khub") / "papers")

    def get_nested(self, *args, default=None):
        if tuple(args) == ("storage", "papers_dir"):
            return self.papers_dir
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "eval/knowledgeos/fixtures/corpus_manifest.json",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "eval/knowledgeos/reports/structured_evidence_vertical_slice_discovery.v1.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "eval/knowledgeos/reports/structured_evidence_vertical_slice_discovery.v1.md",
    )
    args = parser.parse_args()

    config = _ConfigStub()
    payload = build_structured_evidence_vertical_slice_discovery(
        config=config,
        manifest_path=args.manifest,
    )
    schema_validation = payload.get("schemaValidation") or {}
    if not schema_validation.get("ok"):
        errors = "; ".join(str(item) for item in schema_validation.get("errors") or [])
        raise ValueError(f"discovery report schema validation failed: {errors}")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report_md.write_text(
        render_structured_evidence_vertical_slice_discovery_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload.get("counts"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

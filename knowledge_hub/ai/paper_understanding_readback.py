from __future__ import annotations

from pathlib import Path
from typing import Final
import json

from knowledge_hub.ai.paper_understanding_readback_support import (
    BRIEF_PROFILE_SLOTS,
    JsonMap,
    as_maps,
    has_warning,
    row_readback,
)

PAPER_UNDERSTANDING_READBACK_SCHEMA_ID: Final = "knowledge-hub.paper-understanding-readback.v1"


def build_paper_understanding_readback(*, packet_input_report: JsonMap, papers_dir: Path, generated_at: str) -> JsonMap:
    rows = [row_readback(row, papers_dir=papers_dir) for row in as_maps(packet_input_report.get("rows"))]
    paper_readbacks = [paper for row in rows for paper in as_maps(row.get("paperReadbacks"))]
    private_rows = sum(1 for row in rows if has_warning(row, "private_path_marker"))
    forbidden_rows = sum(1 for row in rows if has_warning(row, "forbidden_raw_marker"))
    return {
        "schema": PAPER_UNDERSTANDING_READBACK_SCHEMA_ID,
        "status": "blocked" if private_rows or forbidden_rows else "ready",
        "generatedAt": generated_at,
        "profile": "paper-understanding-readback",
        "policy": {
            "reportOnly": True,
            "localOnlyRawArtifactRefs": True,
            "modelCallsAllowed": False,
            "dbVectorMutation": False,
            "vaultRead": False,
            "publicDefaultPromotionAllowed": False,
        },
        "requirements": {
            "briefProfileSlots": list(BRIEF_PROFILE_SLOTS),
            "requiresCitationLabel": True,
            "requiresSourceHash": True,
            "requiresSnippetHash": True,
        },
        "counts": {
            "rowCount": len(rows),
            "readbackReadyRows": sum(1 for row in rows if row["status"] == "ready"),
            "notApplicableRows": sum(1 for row in rows if row["status"] == "not_applicable"),
            "blockedRows": sum(1 for row in rows if row["status"] == "blocked"),
            "paperReadbackRows": len(paper_readbacks),
            "briefReadyPaperRows": sum(1 for paper in paper_readbacks if bool(paper.get("briefReady"))),
            "briefBlockedPaperRows": sum(1 for paper in paper_readbacks if not bool(paper.get("briefReady"))),
            "privatePathLeakRows": private_rows,
            "forbiddenRawMarkerRows": forbidden_rows,
            "schemaViolationCount": 0,
        },
        "rows": rows,
        "warnings": [] if not private_rows and not forbidden_rows else ["paper understanding readback contains blocked markers"],
    }


def write_paper_understanding_readback(report: JsonMap, *, report_json: Path) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(report_json)}


__all__ = [
    "PAPER_UNDERSTANDING_READBACK_SCHEMA_ID",
    "build_paper_understanding_readback",
    "write_paper_understanding_readback",
]

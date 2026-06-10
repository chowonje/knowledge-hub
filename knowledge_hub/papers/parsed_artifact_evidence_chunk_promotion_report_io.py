from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_parsed_artifact_evidence_chunk_promotion_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    blockers = list(report.get("blockersByCategory") or [])
    lines = [
        "# Parsed Artifact Evidence Chunk Promotion Audit",
        "",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- next: `{report.get('nextRecommendedTranche')}`",
        "",
        "## Counts",
        f"- papersEvaluated: `{counts.get('papersEvaluated')}`",
        f"- papersWithCandidateRows: `{counts.get('papersWithCandidateRows')}`",
        f"- candidateRows: `{counts.get('candidateRows')}`",
        f"- candidateRowsWithValidSourceContentHash: `{counts.get('candidateRowsWithValidSourceContentHash')}`",
        f"- candidateRowsWithValidCharLocators: `{counts.get('candidateRowsWithValidCharLocators')}`",
        f"- answerVisibleEvidenceRows: `{counts.get('answerVisibleEvidenceRows')}`",
        f"- candidateStoreAnswerVisibleRows: `{counts.get('candidateStoreAnswerVisibleRows', 0)}`",
        "",
        "## Blockers",
    ]
    if blockers:
        for row in blockers:
            lines.append(
                f"- `{row.get('category')}`: rows `{row.get('rowCount')}`, papers `{row.get('paperCount')}`"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_parsed_artifact_evidence_chunk_promotion_audit(
    report: dict[str, Any],
    *,
    report_json: str | Path,
    report_md: str | Path,
) -> dict[str, str]:
    json_path = Path(report_json)
    md_path = Path(report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_parsed_artifact_evidence_chunk_promotion_audit_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


__all__ = [
    "render_parsed_artifact_evidence_chunk_promotion_audit_markdown",
    "write_parsed_artifact_evidence_chunk_promotion_audit",
]

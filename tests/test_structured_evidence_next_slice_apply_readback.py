from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.papers.structured_evidence_next_slice_apply_readback import (
    STRUCTURED_EVIDENCE_NEXT_SLICE_APPLY_READBACK_SCHEMA_ID,
    build_structured_evidence_next_slice_apply_readback,
)


class _ConfigStub:
    papers_dir = str(Path.home() / ("." + "khub") / "papers")

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        if tuple(args) == ("storage", "papers_dir"):
            return self.papers_dir
        return default


def test_structured_evidence_next_slice_apply_readback_dry_run_builds_report() -> None:
    root = Path(__file__).resolve().parents[1]
    candidate_report = json.loads(
        (
            root
            / "eval/knowledgeos/reports/structured_evidence_next_slice_candidate_report.v1.json"
        ).read_text(encoding="utf-8")
    )
    expected_baseline = int(candidate_report["counts"]["strictCoveredRows"])
    expected_readback = len(candidate_report["readbackCandidates"])
    expected_greenfield_ids = [
        row["sourceId"] for row in candidate_report["selectedGreenfieldCandidates"][:2]
    ]

    assert int(candidate_report["counts"]["readbackCandidateRows"]) == expected_readback

    payload = build_structured_evidence_next_slice_apply_readback(
        config=_ConfigStub(),
        manifest_path=root / "eval/knowledgeos/fixtures/corpus_manifest.json",
        candidate_report_path=root
        / "eval/knowledgeos/reports/structured_evidence_next_slice_candidate_report.v1.json",
        apply=False,
        selected_greenfield_rows=2,
    )

    assert payload["schema"] == STRUCTURED_EVIDENCE_NEXT_SLICE_APPLY_READBACK_SCHEMA_ID
    assert payload["apply"] is False
    assert payload["status"] == "ready"
    assert payload["schemaValidation"]["ok"] is True
    assert payload["counts"]["baselineStrictCoveredRows"] == expected_baseline
    assert payload["counts"]["readbackPassRows"] == expected_readback
    assert payload["counts"]["greenfieldSelectedRows"] == 2
    assert payload["counts"]["generatedStrictEvidenceRecords"] == 2
    assert payload["counts"]["appliedGreenfieldRows"] == 0
    assert payload["selection"]["greenfieldSourceIds"] == expected_greenfield_ids
    assert payload["policy"]["runtimeAnswerIntegration"] is False
    assert payload["policy"]["citationGradePromotion"] is False
    assert payload["policy"]["tableEquationParser"] is False

    serialized = json.dumps(payload)
    assert "/" + "Users/won" not in serialized
    assert "Mobile" + " Documents" not in serialized
    assert "." + "khub" not in serialized

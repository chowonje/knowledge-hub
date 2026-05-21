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
    assert payload["counts"]["baselineStrictCoveredRows"] == 3
    assert payload["counts"]["readbackPassRows"] == 3
    assert payload["counts"]["greenfieldSelectedRows"] == 2
    assert payload["counts"]["generatedStrictEvidenceRecords"] == 2
    assert payload["counts"]["appliedGreenfieldRows"] == 0
    assert payload["selection"]["greenfieldSourceIds"] == ["2005.11401", "1512.03385"]
    assert payload["policy"]["runtimeAnswerIntegration"] is False
    assert payload["policy"]["citationGradePromotion"] is False
    assert payload["policy"]["tableEquationParser"] is False

    serialized = json.dumps(payload)
    assert "/" + "Users/won" not in serialized
    assert "Mobile" + " Documents" not in serialized
    assert "." + "khub" not in serialized

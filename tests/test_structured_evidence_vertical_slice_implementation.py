"""Tests for structured evidence vertical slice implementation."""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.papers.structured_evidence_vertical_slice_discovery import RECOMMENDED_PAPER_IDS
from knowledge_hub.papers.structured_evidence_vertical_slice_implementation import (
    STRUCTURED_EVIDENCE_VERTICAL_SLICE_IMPLEMENTATION_SCHEMA_ID,
    build_structured_evidence_vertical_slice_implementation,
)


class _ConfigStub:
    papers_dir = str(Path.home() / ("." + "khub") / "papers")

    def get_nested(self, *args, default=None):
        if tuple(args) == ("storage", "papers_dir"):
            return self.papers_dir
        return default


def test_structured_evidence_vertical_slice_implementation_builds_report() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "eval/knowledgeos/fixtures/corpus_manifest.json"
    payload = build_structured_evidence_vertical_slice_implementation(
        config=_ConfigStub(),
        manifest_path=manifest,
        apply=False,
    )

    assert payload["schema"] == STRUCTURED_EVIDENCE_VERTICAL_SLICE_IMPLEMENTATION_SCHEMA_ID
    assert payload["apply"] is False
    assert payload["counts"]["paperRows"] == len(RECOMMENDED_PAPER_IDS)
    assert payload["policy"]["runtimeAnswerIntegration"] is False
    assert payload["policy"]["citationGradePromotion"] is False
    assert payload["policy"]["tableEquationParser"] is False
    assert "table_cell_numeric" in payload["deferredEvidenceTypes"]

    by_id = {row["sourceId"]: row for row in payload["papers"]}
    assert by_id["1706.03762"]["mode"] == "pilot_readback"
    assert by_id["2005.11401"]["mode"] == "greenfield_section"
    assert by_id["1506.02640"]["mode"] == "figure_caption_readback"

    pilot = by_id["1706.03762"]
    assert pilot["status"] == "pass"
    trace = pilot["traceValidation"]
    assert trace["allStrictEvidenceTracePass"] is True
    assert trace["citationGradeFalseMaintained"] is True
    assert trace["runtimeEvidenceFalseMaintained"] is True

    greenfield = by_id["2005.11401"]
    assert greenfield["generatedRecords"]["sourceSpanCount"] == 1
    assert greenfield["generatedRecords"]["strictEvidenceCount"] == 1
    assert greenfield["generatedRecords"]["evidenceType"] == "section_text_offset"
    assert greenfield["traceValidation"]["pass"] is True
    assert greenfield["traceValidation"]["citationGradeFalse"] is True
    assert greenfield["traceValidation"]["runtimeEvidenceFalse"] is True

    serialized = json.dumps(payload)
    assert payload["schemaValidation"]["ok"] is True
    assert payload["schemaValidation"]["errors"] == []
    assert "schema not found" not in serialized
    assert "/" + "Users/won" not in serialized
    assert "Mobile" + " Documents" not in serialized

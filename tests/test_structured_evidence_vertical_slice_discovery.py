"""Tests for structured evidence vertical slice discovery report."""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.papers.structured_evidence_vertical_slice_discovery import (
    RECOMMENDED_PAPER_IDS,
    STRUCTURED_EVIDENCE_VERTICAL_SLICE_DISCOVERY_SCHEMA_ID,
    build_structured_evidence_vertical_slice_discovery,
)


class _ConfigStub:
    papers_dir = str(Path.home() / ("." + "khub") / "papers")

    def get_nested(self, *args, default=None):
        if tuple(args) == ("storage", "papers_dir"):
            return self.papers_dir
        return default


def test_structured_evidence_vertical_slice_discovery_builds_report() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "eval/knowledgeos/fixtures/corpus_manifest.json"
    payload = build_structured_evidence_vertical_slice_discovery(
        config=_ConfigStub(),
        manifest_path=manifest,
    )

    assert payload["schema"] == STRUCTURED_EVIDENCE_VERTICAL_SLICE_DISCOVERY_SCHEMA_ID
    assert payload["counts"]["corpusManifestRows"] == 100
    assert len(payload["recommendedFirstSlice"]) == len(RECOMMENDED_PAPER_IDS)
    assert payload["policy"]["runtimeAnswerIntegration"] is False
    assert payload["policy"]["manifestMutation"] is False
    assert "parsedArtifactLayout" in payload["currentStructure"]
    assert payload["requiredContract"]["traceChain"]
    assert payload["schemaValidation"]["ok"] is True
    assert payload["schemaValidation"]["errors"] == []
    serialized = json.dumps(payload)
    assert "schema not found" not in serialized

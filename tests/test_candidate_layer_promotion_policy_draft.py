from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_promotion_policy_draft import (
    CANDIDATE_LAYER_PROMOTION_POLICY_DRAFT_SCHEMA_ID,
    build_candidate_layer_promotion_policy_draft,
    write_candidate_layer_promotion_policy_draft_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _payloads(*, strict_nonzero: bool = False, wrong_schema: bool = False) -> dict[str, dict]:
    summary_schema = "knowledge-hub.paper.structured-candidate-summary.v1"
    gate_schema = "knowledge-hub.paper.candidate-layer-review-gate.v1"
    backlog_schema = "knowledge-hub.paper.candidate-layer-blocker-backlog.v1"
    source_schema = "knowledge-hub.paper.source-span-offset-authority-audit.v1"
    equation_schema = "knowledge-hub.paper.equation-alignment-feasibility-audit.v1"
    table_schema = "knowledge-hub.paper.table-cell-provenance-feasibility-audit.v1"
    figure_schema = "knowledge-hub.paper.figure-region-link-feasibility-audit.v1"
    if wrong_schema:
        summary_schema = "example.wrong.summary"
        gate_schema = "example.wrong.gate"
    strict_count = 1 if strict_nonzero else 0
    return {
        "structuredSummary": {
            "schema": summary_schema,
            "counts": {
                "totalCandidates": 86,
                "byLayer": {
                    "sectionspan": 61,
                    "figure_caption": 11,
                    "equation_quote": 9,
                    "table_region": 5,
                },
                "strictEligibleCandidates": strict_count,
                "citationGradeCandidates": 0,
                "runtimeEvidenceCandidates": 0,
            },
            "policy": {
                "strictEvidenceCreated": False,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        },
        "candidateLayerReviewGate": {
            "schema": gate_schema,
            "counts": {
                "currentRuntimeAnswerableQuestions": 0,
                "strictEligibleCandidates": 0,
                "citationGradeCandidates": 0,
                "runtimeEvidenceCandidates": 0,
            },
            "gate": {
                "candidateLayerReviewReady": True,
                "strictEvidenceReady": False,
                "parserRoutingReady": False,
                "answerIntegrationReady": False,
            },
            "policy": {
                "strictEvidenceCreated": False,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        },
        "candidateLayerBlockerBacklog": {
            "schema": backlog_schema,
            "counts": {
                "strictEligibleCandidates": 0,
                "citationGradeCandidates": 0,
                "runtimeEvidenceCandidates": 0,
                "currentRuntimeAnswerableQuestions": 0,
            },
            "backlog": [
                {
                    "blocker": "equation_quote_alignment_missing",
                    "affected_layers": ["equation_quote"],
                },
                {
                    "blocker": "table_cell_row_column_bbox_provenance_missing",
                    "affected_layers": ["table_region"],
                },
                {
                    "blocker": "figure_region_link_unverified",
                    "affected_layers": ["figure_caption"],
                },
                {
                    "blocker": "generated_markdown_offsets_are_not_original_pdf_offsets",
                    "affected_layers": ["sectionspan", "figure_caption", "equation_quote", "table_region"],
                },
            ],
            "policy": {
                "strictEvidenceCreated": False,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        },
        "sourceSpanOffsetAuthorityAudit": {
            "schema": source_schema,
            "counts": {
                "byLayer": {
                    "sectionspan": 61,
                    "figure_caption": 11,
                    "equation_quote": 9,
                    "table_region": 5,
                },
                "canonicalParsedTextSpanCandidates": 74,
                "originalPdfOffsetCandidates": 0,
                "strictEligibleCandidates": 0,
                "citationGradeCandidates": 0,
                "runtimeEvidenceCandidates": 0,
            },
            "policy": {
                "strictEvidenceCreated": False,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        },
        "equationAlignmentFeasibilityAudit": {
            "schema": equation_schema,
            "counts": {
                "canonicalSourceSpanCreatedCandidates": 0,
                "strictEligibleCandidates": 0,
                "citationGradeCandidates": 0,
                "runtimeEvidenceCandidates": 0,
            },
            "policy": {
                "strictEvidenceCreated": False,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        },
        "tableCellProvenanceFeasibilityAudit": {
            "schema": table_schema,
            "counts": {
                "rowColumnTextCandidates": 5,
                "tableCellCitationGradeCandidates": 0,
                "strictEligibleCandidates": 0,
                "citationGradeCandidates": 0,
                "runtimeEvidenceCandidates": 0,
            },
            "policy": {
                "strictEvidenceCreated": False,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        },
        "figureRegionLinkFeasibilityAudit": {
            "schema": figure_schema,
            "counts": {
                "captionSourceSpanCandidates": 9,
                "figureRegionLinkVerifiedCandidates": 0,
                "strictEligibleCandidates": 0,
                "citationGradeCandidates": 0,
                "runtimeEvidenceCandidates": 0,
            },
            "policy": {
                "strictEvidenceCreated": False,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        },
    }


def _reports(root: Path, **kwargs: object) -> dict[str, Path]:
    payloads = _payloads(**kwargs)
    return {key: _write(root, f"{key}.json", payload) for key, payload in payloads.items()}


def _build(paths: dict[str, Path]) -> dict:
    return build_candidate_layer_promotion_policy_draft(
        structured_summary_report=paths["structuredSummary"],
        candidate_layer_review_gate_report=paths["candidateLayerReviewGate"],
        candidate_layer_blocker_backlog_report=paths["candidateLayerBlockerBacklog"],
        source_span_offset_authority_audit_report=paths["sourceSpanOffsetAuthorityAudit"],
        equation_alignment_feasibility_audit_report=paths["equationAlignmentFeasibilityAudit"],
        table_cell_provenance_feasibility_audit_report=paths["tableCellProvenanceFeasibilityAudit"],
        figure_region_link_feasibility_audit_report=paths["figureRegionLinkFeasibilityAudit"],
    )


def test_candidate_layer_promotion_policy_draft_reports_tracks_and_validates_schema(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path))

    assert payload["schema"] == CANDIDATE_LAYER_PROMOTION_POLICY_DRAFT_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_PROMOTION_POLICY_DRAFT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "draft_ready"
    assert payload["counts"]["totalCandidates"] == 86
    assert payload["counts"]["promotionTrackCount"] == 4
    assert payload["counts"]["candidateFormalizationReadyTracks"] == 1
    assert payload["counts"]["strictPromotionReadyTracks"] == 0
    assert payload["counts"]["runtimePromotionAllowedTracks"] == 0
    assert payload["gate"]["promotionPolicyDraftReady"] is True


def test_candidate_layer_promotion_policy_draft_keeps_all_tracks_non_strict(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path))

    assert payload["policy"]["draftOnly"] is True
    assert payload["policy"]["runtimePolicyChanged"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["runtimePromotionAllowed"] is False
    assert payload["policy"]["parserRoutingChanged"] is False
    assert payload["policy"]["canonicalParsedArtifactsWritten"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["policy"]["reindexOrReembed"] is False
    assert payload["policy"]["answerIntegrationChanged"] is False
    for track in payload["promotionTracks"]:
        assert track["strict_promotion_ready"] is False
        assert track["parser_routing_ready"] is False
        assert track["answer_integration_ready"] is False
        assert track["strict_eligible"] is False
        assert track["runtime_evidence"] is False
        assert "strict_evidence_promotion" in track["disallowed_actions"]
        assert "runtime_promotion_disabled_for_tranche" in track["strict_blockers"]


def test_candidate_layer_promotion_policy_draft_blocks_unsafe_upstream_counts(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path, strict_nonzero=True))

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert "structuredSummary_strictEligibleCandidates_nonzero" in payload["gate"]["unsafeUpstreamFlags"]


def test_candidate_layer_promotion_policy_draft_blocks_wrong_schema_ids(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path, wrong_schema=True))

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert set(payload["gate"]["schemaViolations"]) == {
        "structuredSummary_schema_mismatch",
        "candidateLayerReviewGate_schema_mismatch",
    }


def test_candidate_layer_promotion_policy_draft_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    payload = _build(_reports(tmp_path / "input"))

    paths = write_candidate_layer_promotion_policy_draft_reports(payload, tmp_path / "reports")

    assert set(paths) == {"draft", "summary", "markdown"}
    draft = json.loads(Path(paths["draft"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(draft, CANDIDATE_LAYER_PROMOTION_POLICY_DRAFT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, CANDIDATE_LAYER_PROMOTION_POLICY_DRAFT_SCHEMA_ID, strict=True).ok
    assert "does not modify runtime evidence policy" in markdown

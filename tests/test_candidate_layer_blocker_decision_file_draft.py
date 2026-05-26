from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_decision_file_draft import (
    CANDIDATE_LAYER_BLOCKER_DECISION_FILE_DRAFT_SCHEMA_ID,
    build_candidate_layer_blocker_decision_file_draft,
    write_candidate_layer_blocker_decision_file_draft_reports,
)
from knowledge_hub.papers.candidate_layer_blocker_decision_file_validation import (
    CANDIDATE_LAYER_BLOCKER_DECISION_FILE_VALIDATION_SCHEMA_ID,
    build_candidate_layer_blocker_decision_file_validation,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _input_row(row_id: str, bucket: str = "manual_decision_required") -> dict:
    allowed = {
        "manual_decision_required": [
            "needs_review",
            "record_manual_approval_in_separate_decision_file",
            "record_manual_rejection_in_separate_decision_file",
            "keep_blocked",
        ],
        "operator_approval_required": [
            "needs_review",
            "approve_diagnostic_operator_action_in_separate_decision_file",
            "decline_diagnostic_operator_action_keep_blocked",
            "keep_blocked",
        ],
    }[bucket]
    return {
        "input_row_id": f"input:{row_id}",
        "source_decision_row_id": row_id,
        "source_review_card_id": f"review:{row_id}",
        "source_backlog_id": f"backlog:{row_id}",
        "blocker": f"blocker:{row_id}",
        "priority": "P0",
        "review_bucket": bucket,
        "affected_layers": ["sectionspan"],
        "allowed_decisions": allowed,
        "decision": "needs_review",
        "reviewer": "",
        "notes": "",
        "decision_scope": "candidate_layer_blocker_decision_input_pack_only_no_runtime_or_strict_promotion",
        "evidence_tier": "candidate_layer_blocker_decision_input_pack_only",
        "report_only": True,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }


def _input_pack(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-input-pack.v1",
        "status": "decision_input_pack_ready",
        "counts": {
            "inputRows": len(rows),
            "pendingSourceDecisionRows": len(rows),
            "defaultNeedsReviewRows": len(rows),
            "acceptedDecisionRows": 0,
            "operatorApprovedRows": 0,
            "strictEligibleRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "decisionInputPackReady": True,
            "containsAcceptedDecisions": False,
            "containsOperatorApprovals": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "decisionInputPackOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "decisionInputs": rows,
    }
    payload.update(overrides)
    return payload


def test_decision_file_draft_emits_only_needs_review_rows(tmp_path: Path) -> None:
    input_pack = _write(
        tmp_path,
        "input-pack.json",
        _input_pack([_input_row("manual"), _input_row("operator", "operator_approval_required")]),
    )

    payload = build_candidate_layer_blocker_decision_file_draft(
        candidate_layer_blocker_decision_input_pack_report=input_pack
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_DECISION_FILE_DRAFT_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_DECISION_FILE_DRAFT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "decision_file_draft_ready"
    assert payload["counts"]["draftRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["nonNeedsReviewRows"] == 0
    assert payload["counts"]["acceptedDecisionRows"] == 0
    assert payload["counts"]["operatorApprovedRows"] == 0
    assert payload["gate"]["containsOnlyNeedsReviewDefaults"] is True
    assert all(row["decision"] == "needs_review" for row in payload["draftRows"])
    assert all(row["decision"] == "needs_review" for row in payload["decisionFileDraft"]["decisions"])


def test_decision_file_draft_output_validates_as_non_approval_decision_file(tmp_path: Path) -> None:
    input_pack = _write(tmp_path, "input-pack.json", _input_pack([_input_row("manual")]))
    payload = build_candidate_layer_blocker_decision_file_draft(
        candidate_layer_blocker_decision_input_pack_report=input_pack
    )
    decisions_file = _write(tmp_path, "candidate-layer-blocker-decisions.draft.json", payload["decisionFileDraft"])

    validation = build_candidate_layer_blocker_decision_file_validation(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
        candidate_layer_blocker_decisions_file=decisions_file,
    )

    assert validate_payload(validation, CANDIDATE_LAYER_BLOCKER_DECISION_FILE_VALIDATION_SCHEMA_ID, strict=True).ok
    assert validation["status"] == "decision_file_validated"
    assert validation["counts"]["validRows"] == 1
    assert validation["counts"]["needsReviewRows"] == 1
    assert validation["counts"]["nonNeedsReviewRows"] == 0
    assert validation["counts"]["acceptedDecisionRows"] == 0
    assert validation["counts"]["operatorApprovedRows"] == 0
    assert validation["policy"]["strictEvidenceCreated"] is False


def test_decision_file_draft_blocks_unsafe_input_pack(tmp_path: Path) -> None:
    input_pack = _write(
        tmp_path,
        "input-pack.json",
        _input_pack(
            [_input_row("manual")],
            schema="example.wrong.input-pack.v1",
            policy={
                "reportOnly": True,
                "decisionInputPackOnly": True,
                "strictEvidenceCreated": True,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        ),
    )

    payload = build_candidate_layer_blocker_decision_file_draft(
        candidate_layer_blocker_decision_input_pack_report=input_pack
    )

    assert payload["status"] == "blocked"
    assert "candidate_layer_blocker_decision_input_pack_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionInputPack_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_decision_file_draft_writer_outputs_schema_valid_files(tmp_path: Path) -> None:
    input_pack = _write(tmp_path / "input", "input-pack.json", _input_pack([_input_row("manual")]))
    payload = build_candidate_layer_blocker_decision_file_draft(
        candidate_layer_blocker_decision_input_pack_report=input_pack
    )

    paths = write_candidate_layer_blocker_decision_file_draft_reports(payload, tmp_path / "reports")

    assert set(paths) == {"draftReport", "decisionFileDraft", "summary", "markdown"}
    report = json.loads(Path(paths["draftReport"]).read_text(encoding="utf-8"))
    decision_file = json.loads(Path(paths["decisionFileDraft"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, CANDIDATE_LAYER_BLOCKER_DECISION_FILE_DRAFT_SCHEMA_ID, strict=True).ok
    assert decision_file["draftOnly"] is True
    assert decision_file["decisions"][0]["decision"] == "needs_review"
    assert summary["counts"]["draftRows"] == 1
    assert "This draft is an editable starting point only" in markdown

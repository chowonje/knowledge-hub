from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.candidate_layer_blocker_manual_decision_command_packet import (
    CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_COMMAND_PACKET_SCHEMA_ID,
    build_candidate_layer_blocker_manual_decision_command_packet,
    write_candidate_layer_blocker_manual_decision_command_packet_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _input_pack_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-input-pack.v1",
        "status": "decision_input_pack_ready",
        "counts": {
            "inputRows": 12,
            "manualDecisionInputRows": 4,
            "technicalDecisionInputRows": 6,
            "policyDecisionInputRows": 2,
            "strictEligibleRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "decisionInputPackReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
        },
        "policy": {
            "reportOnly": True,
            "strictEvidenceCreated": False,
            "parserRoutingChanged": False,
            "databaseMutation": False,
        },
        "inputRows": [],
    }
    payload.update(overrides)
    return payload


def _template_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-template.v1",
        "status": "decision_template_ready",
        "counts": {
            "templateRows": 12,
            "strictEligibleRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "decisionTemplateReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
        },
        "policy": {"reportOnly": True, "strictEvidenceCreated": False},
        "decisionRows": [],
    }
    payload.update(overrides)
    return payload


def _draft_payload(decision_file: str, **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-decision-file-draft.v1",
        "status": "decision_file_draft_ready",
        "counts": {
            "draftRows": 12,
            "needsReviewRows": 12,
            "strictEligibleRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "decisionFileDraftReady": True,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
        },
        "policy": {"reportOnly": True, "decisionFileDraftOnly": True},
        "reportPaths": {"decisionFileDraft": decision_file},
    }
    payload.update(overrides)
    return payload


def _backlog_payload(**overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.candidate-layer-blocker-backlog.v1",
        "status": "ok",
        "counts": {
            "backlogItemCount": 12,
            "strictEligibleCandidates": 0,
            "runtimeEvidenceCandidates": 0,
        },
        "gate": {
            "decision": "blocker_backlog_ready",
            "schemaViolations": [],
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
        },
        "policy": {"backlogOnly": True, "strictEvidenceCreated": False},
        "backlog": [],
    }
    payload.update(overrides)
    return payload


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    decision_file = tmp_path / "candidate-layer-blocker-decisions.draft.json"
    decision_file.write_text('{"decisions": []}\n', encoding="utf-8")
    input_pack = _write(tmp_path, "input-pack.json", _input_pack_payload())
    template = _write(tmp_path, "template.json", _template_payload())
    draft = _write(tmp_path, "draft.json", _draft_payload(str(decision_file)))
    backlog = _write(tmp_path, "backlog.json", _backlog_payload())
    return input_pack, template, draft, backlog, decision_file


def test_manual_decision_command_packet_emits_report_only_commands(tmp_path: Path) -> None:
    input_pack, template, draft, backlog, decision_file = _write_inputs(tmp_path)

    payload = build_candidate_layer_blocker_manual_decision_command_packet(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decision_file_draft_report=draft,
        candidate_layer_blocker_backlog_report=backlog,
        worktree=tmp_path / "worktree",
        output_dir=tmp_path / "packet",
    )

    assert payload["schema"] == CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_COMMAND_PACKET_SCHEMA_ID
    assert validate_payload(payload, CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_COMMAND_PACKET_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "command_packet_ready"
    assert payload["counts"]["commandCount"] == 5
    assert payload["counts"]["manualDecisionInputRows"] == 4
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["counts"]["runtimeEvidenceRows"] == 0
    assert payload["gate"]["commandsExecuted"] is False
    assert payload["gate"]["decisionsRecorded"] is False
    assert payload["policy"]["decisionFileModified"] is False
    assert payload["policy"]["databaseMutation"] is False
    assert payload["commands"][0]["name"] == "create_review_copy"
    assert str(decision_file) in payload["commands"][0]["command"]
    assert payload["commands"][1]["name"] == "manual_edit_review_copy"
    assert "candidate_layer_blocker_decision_file_validation" in payload["commands"][2]["command"]
    assert "candidate_layer_blocker_decision_record" in payload["commands"][3]["command"]
    assert "candidate_layer_blocker_resolution_preview" in payload["commands"][4]["command"]


def test_manual_decision_command_packet_blocks_unsafe_inputs(tmp_path: Path) -> None:
    input_pack, template, draft, backlog, _decision_file = _write_inputs(tmp_path)
    bad_input_pack = _write(tmp_path, "bad-input-pack.json", _input_pack_payload(schema="example.bad.v1"))

    payload = build_candidate_layer_blocker_manual_decision_command_packet(
        candidate_layer_blocker_decision_input_pack_report=bad_input_pack,
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decision_file_draft_report=draft,
        candidate_layer_blocker_backlog_report=backlog,
        worktree=tmp_path / "worktree",
        output_dir=tmp_path / "packet",
    )

    assert payload["status"] == "blocked"
    assert payload["commands"] == []
    assert payload["counts"]["commandCount"] == 0
    assert "candidate_layer_blocker_decision_input_pack_schema_mismatch" in payload["gate"]["schemaViolations"]


def test_manual_decision_command_packet_blocks_missing_draft_decision_file_path(tmp_path: Path) -> None:
    input_pack, template, _draft, backlog, _decision_file = _write_inputs(tmp_path)
    draft_without_file = _write(tmp_path / "missing-draft-dir", "draft-without-file.json", _draft_payload(""))

    payload = build_candidate_layer_blocker_manual_decision_command_packet(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decision_file_draft_report=draft_without_file,
        candidate_layer_blocker_backlog_report=backlog,
        worktree=tmp_path / "worktree",
        output_dir=tmp_path / "packet",
    )

    assert payload["status"] == "blocked"
    assert "candidate_layer_blocker_decision_file_draft_path_missing" in payload["gate"]["schemaViolations"]


def test_manual_decision_command_packet_finds_sibling_draft_file_without_report_paths(tmp_path: Path) -> None:
    input_pack, template, _draft, backlog, decision_file = _write_inputs(tmp_path)
    draft_without_report_paths = _write(
        tmp_path,
        "draft-with-sibling-file.json",
        _draft_payload(str(decision_file), reportPaths={}),
    )

    payload = build_candidate_layer_blocker_manual_decision_command_packet(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decision_file_draft_report=draft_without_report_paths,
        candidate_layer_blocker_backlog_report=backlog,
        worktree=tmp_path / "worktree",
        output_dir=tmp_path / "packet",
    )

    assert payload["status"] == "command_packet_ready"
    assert str(decision_file) in payload["commands"][0]["command"]


def test_manual_decision_command_packet_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    input_pack, template, draft, backlog, _decision_file = _write_inputs(tmp_path)
    payload = build_candidate_layer_blocker_manual_decision_command_packet(
        candidate_layer_blocker_decision_input_pack_report=input_pack,
        candidate_layer_blocker_decision_template_report=template,
        candidate_layer_blocker_decision_file_draft_report=draft,
        candidate_layer_blocker_backlog_report=backlog,
        worktree=tmp_path / "worktree",
        output_dir=tmp_path / "packet",
    )

    paths = write_candidate_layer_blocker_manual_decision_command_packet_reports(payload, tmp_path / "packet")

    assert set(paths) == {"packet", "summary", "markdown"}
    packet = json.loads(Path(paths["packet"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(packet, CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_COMMAND_PACKET_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["commandCount"] == 5
    assert "Candidate Layer Blocker Manual Decision Command Packet" in markdown

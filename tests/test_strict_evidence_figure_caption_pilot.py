from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_policy_gate import (
    POLICY_STATUS_CANDIDATE_ONLY as POLICY_GATE_STATUS_CANDIDATE_ONLY,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_promotion_tranche_plan import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
    PLAN_STATUS_CANDIDATE_ONLY,
    TRANCHE_FIGURE_CAPTION_PILOT,
    TRANCHE_TEXT_SECTION_PILOT,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    build_sample_strict_evidence_record_from_packet_row,
)
from knowledge_hub.papers.strict_evidence_figure_caption_pilot import (
    PILOT_STATUS_BLOCKED_HASH_CONTRACT,
    PILOT_STATUS_CANDIDATE_ONLY,
    PILOT_STATUS_HELD_OUT_SECTION,
    STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID,
    build_strict_evidence_figure_caption_pilot,
    write_strict_evidence_figure_caption_pilot_reports,
)
def _strict_evidence_record(
    *,
    paper_id: str = "paper-1",
    index: int = 1,
    artifact_type: str = "figure",
) -> dict:
    packet_row = {
        "packet_review_row_id": f"packet:{index:04d}",
        "paper_id": paper_id,
        "artifact_type": artifact_type,
        "sourceSpanId": f"source-span:{paper_id}:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:{paper_id}:{artifact_type}:{index}",
        "sourceContentHash": "hash-1",
        "source_file": f"{paper_id}.pdf",
        "text_surface": "Figure 1 caption text",
        "proposed_chars": {
            "start": 0,
            "end": 12,
            "basis": "sourceContentHash",
            "normalization": "nfkc_whitespace_casefold_v1",
            "expectedSubstringSha256": "abc123",
        },
    }
    return build_sample_strict_evidence_record_from_packet_row(packet_row, run_id="run-1")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_tranche_plan_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _synthetic_tranche_plan_row(
    *,
    store_path: Path,
    index: int = 1,
    artifact_type: str = "figure",
    paper_id: str = "paper-1",
) -> dict:
    planned_tranche = (
        TRANCHE_TEXT_SECTION_PILOT
        if artifact_type == "section"
        else TRANCHE_FIGURE_CAPTION_PILOT
    )
    return {
        "plan_row_id": f"parsed-artifact-strict-evidence-promotion-tranche-plan:{index:04d}",
        "policy_gate_row_id": f"parsed-artifact-strict-evidence-policy-gate:{index:04d}",
        "strictEvidenceId": f"strict-evidence:{paper_id}:{artifact_type}:{index}",
        "sourceSpanId": f"source-span:{paper_id}:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:{paper_id}:{artifact_type}:{index}",
        "runId": "run-1",
        "paper_id": paper_id,
        "artifact_type": artifact_type,
        "sourceContentHash": "hash-1",
        "idempotencyKey": f"strict-idem-{index}",
        "strict_evidence_store_path": str(store_path),
        "strict_evidence_store_line": index,
        "policy_gate_status": POLICY_GATE_STATUS_CANDIDATE_ONLY,
        "strictEvidencePolicyCandidateOnly": True,
        "planned_tranche": planned_tranche,
        "planned_tranche_scope": (
            "text strict evidence section pilot"
            if artifact_type == "section"
            else "figure caption text strict evidence pilot (not figure region)"
        ),
        "plan_status": PLAN_STATUS_CANDIDATE_ONLY,
        "plan_blockers": [],
        "promotionTranchePlanCandidateOnly": True,
        "strictEligible": False,
        "strictEvidenceWriteRows": 0,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "canonicalParsedArtifactsWritten": False,
        "sourceSpanUpdatedRows": 0,
        "recommended_action": f"queue_for_{planned_tranche}_dry_run",
    }


def _synthetic_tranche_plan_report(rows: list[dict]) -> dict:
    section_rows = sum(1 for row in rows if row.get("planned_tranche") == TRANCHE_TEXT_SECTION_PILOT)
    figure_rows = sum(1 for row in rows if row.get("planned_tranche") == TRANCHE_FIGURE_CAPTION_PILOT)
    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "policyGateReportPath": "/tmp/policy-gate.json",
            "policyGateSchema": "knowledge-hub.paper.parsed-artifact-strict-evidence-policy-gate.v1",
            "policyGateReportStatus": "ok",
            "requestedPaperIds": [],
        },
        "counts": {
            "inputRows": len(rows),
            "policyCandidateRows": len(rows),
            "plannedPromotionRows": len(rows),
            "sectionPilotRows": section_rows,
            "figureCaptionPilotRows": figure_rows,
            "holdoutRows": 0,
            "blockedInputReportSchemaViolationRows": 0,
            "blockedPolicyGateNotCandidateRows": 0,
            "blockedMissingRecordIdentityRows": 0,
            "blockedUnsupportedArtifactTypeRows": 0,
            "blockedRuntimeOrCitationFlagViolationRows": 0,
            "strictEvidenceWriteRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "schemaViolationCount": 0,
            "byPaperId": {},
            "byArtifactType": {},
            "byPlannedTranche": {},
            "byPlanStatus": {PLAN_STATUS_CANDIDATE_ONLY: len(rows)},
            "byRecommendedAction": {},
        },
        "tranches": [],
        "blockedLater": [],
        "gate": {
            "promotionTranchePlanReady": True,
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_strict_evidence_promotion_tranche_plan_ready",
            "recommendedNextTranche": TRANCHE_FIGURE_CAPTION_PILOT,
            "recommendedTrancheExecutionOrder": [
                TRANCHE_TEXT_SECTION_PILOT,
                TRANCHE_FIGURE_CAPTION_PILOT,
            ],
        },
        "policy": {
            "reportOnly": True,
            "promotionTranchePlanOnly": True,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": [],
        "rows": rows,
    }


def test_figure_caption_pilot_marks_figure_rows_and_holds_out_section(tmp_path: Path) -> None:
    store_path = tmp_path / "strict.jsonl"
    _write_jsonl(
        store_path,
        [
            _strict_evidence_record(index=1, artifact_type="figure"),
            _strict_evidence_record(index=2, artifact_type="section"),
        ],
    )
    tranche_plan_path = tmp_path / "tranche-plan.json"
    _write_tranche_plan_report(
        tranche_plan_path,
        _synthetic_tranche_plan_report(
            [
                _synthetic_tranche_plan_row(store_path=store_path, index=1, artifact_type="figure"),
                _synthetic_tranche_plan_row(store_path=store_path, index=2, artifact_type="section"),
            ]
        ),
    )

    report = build_strict_evidence_figure_caption_pilot(
        tranche_plan_report_path=tranche_plan_path,
    )

    assert report["schema"] == STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["figureCaptionPilotInputRows"] == 1
    assert report["counts"]["figureCaptionPilotCandidateOnlyRows"] == 1
    assert report["counts"]["heldOutSectionRows"] == 1
    assert report["gate"]["figureCaptionPilotReady"] is True
    assert report["gate"]["sectionRowsProcessedAsActivePilot"] is False
    assert report["policy"]["figureCaptionTextOnlyNotFigureRegion"] is True
    assert {row["pilot_status"] for row in report["rows"]} == {
        PILOT_STATUS_CANDIDATE_ONLY,
        PILOT_STATUS_HELD_OUT_SECTION,
    }
    assert validate_payload(
        report,
        STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID,
        strict=True,
    ).ok


def test_figure_caption_pilot_blocks_hash_contract_violation(tmp_path: Path) -> None:
    store_path = tmp_path / "strict.jsonl"
    record = _strict_evidence_record(index=1, artifact_type="figure")
    record["verbatimSubstringSha256"] = "mismatch"
    _write_jsonl(store_path, [record])
    tranche_plan_path = tmp_path / "tranche-plan.json"
    _write_tranche_plan_report(
        tranche_plan_path,
        _synthetic_tranche_plan_report(
            [_synthetic_tranche_plan_row(store_path=store_path, index=1, artifact_type="figure")]
        ),
    )

    report = build_strict_evidence_figure_caption_pilot(
        tranche_plan_report_path=tranche_plan_path,
    )

    assert report["status"] == "blocked"
    assert report["rows"][0]["pilot_status"] == PILOT_STATUS_BLOCKED_HASH_CONTRACT
    assert validate_payload(
        report,
        STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID,
        strict=True,
    ).ok


def test_figure_caption_pilot_blocks_invalid_input_schema(tmp_path: Path) -> None:
    tranche_plan = _synthetic_tranche_plan_report(
        [_synthetic_tranche_plan_row(store_path=tmp_path / "strict.jsonl", index=1, artifact_type="figure")]
    )
    tranche_plan["schema"] = "wrong.schema"
    tranche_plan_path = tmp_path / "tranche-plan.json"
    _write_tranche_plan_report(tranche_plan_path, tranche_plan)

    report = build_strict_evidence_figure_caption_pilot(
        tranche_plan_report_path=tranche_plan_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["gate"]["figureCaptionPilotReady"] is False
    assert validate_payload(
        report,
        STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID,
        strict=True,
    ).ok


def test_figure_caption_pilot_integrated_from_tranche_plan_builder(tmp_path: Path) -> None:
    store_path = tmp_path / "strict.jsonl"
    _write_jsonl(
        store_path,
        [
            _strict_evidence_record(index=1, artifact_type="figure"),
            _strict_evidence_record(index=2, artifact_type="figure"),
        ],
    )
    tranche_plan_path = tmp_path / "tranche-plan.json"
    _write_tranche_plan_report(
        tranche_plan_path,
        _synthetic_tranche_plan_report(
            [
                _synthetic_tranche_plan_row(store_path=store_path, index=1, artifact_type="figure"),
                _synthetic_tranche_plan_row(store_path=store_path, index=2, artifact_type="figure"),
            ]
        ),
    )

    report = build_strict_evidence_figure_caption_pilot(
        tranche_plan_report_path=tranche_plan_path,
    )

    assert report["status"] == "ok"
    assert report["counts"]["figureCaptionPilotCandidateOnlyRows"] == 2
    assert report["counts"]["heldOutSectionRows"] == 0
    assert report["counts"]["strictEvidenceWriteRows"] == 0


def test_figure_caption_pilot_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    store_path = tmp_path / "strict.jsonl"
    _write_jsonl(store_path, [_strict_evidence_record(index=1, artifact_type="figure")])
    tranche_plan_path = tmp_path / "tranche-plan.json"
    _write_tranche_plan_report(
        tranche_plan_path,
        _synthetic_tranche_plan_report(
            [_synthetic_tranche_plan_row(store_path=store_path, index=1, artifact_type="figure")]
        ),
    )
    report = build_strict_evidence_figure_caption_pilot(
        tranche_plan_report_path=tranche_plan_path,
    )
    paths = write_strict_evidence_figure_caption_pilot_reports(report, tmp_path / "reports")
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert validate_payload(
        written,
        STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_SCHEMA_ID,
        strict=True,
    ).ok

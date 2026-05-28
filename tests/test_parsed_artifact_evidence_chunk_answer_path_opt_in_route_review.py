from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_opt_in_route_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review,
    write_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_real_answer_quality_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID,
)


def _quality_report(path: Path, *, status: str = "ready", pass_rows: int = 2) -> Path:
    payload = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID,
        "status": status,
        "counts": {
            "passRows": pass_rows,
            "failRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_route_review_ready_identifies_internal_path_and_public_searcher_gap(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review(
        answer_quality_smoke_report=_quality_report(tmp_path / "quality.json"),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["answerQualitySmokePassRows"] == 2
    assert report["counts"]["routeReviewFailRows"] == 0
    assert report["counts"]["routeReviewGapRows"] == 1
    assert report["gate"]["internalRuntimeQueryPlanIngressReady"] is True
    assert report["gate"]["publicSearcherIngressGap"] is True
    assert report["gate"]["publicCliDefaultUnchanged"] is True
    assert report["counts"]["publicCliFlagRows"] == 0
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_route_review_blocks_when_answer_quality_smoke_not_ready(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review(
        answer_quality_smoke_report=_quality_report(tmp_path / "quality.json", status="blocked", pass_rows=0),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["gate"]["answerQualitySmokeReady"] is False
    assert "answer_quality_smoke_not_ready" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_route_review_blocks_when_answer_quality_smoke_missing(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review(
        answer_quality_smoke_report=tmp_path / "missing.json",
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "answer_quality_smoke_schema_mismatch" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_route_review_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review(
        answer_quality_smoke_report=_quality_report(tmp_path / "quality.json"),
        generated_at="2026-05-29T00:00:00Z",
    )
    paths = write_parsed_artifact_evidence_chunk_answer_path_opt_in_route_review(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path Opt-in Route Review"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import importlib
import json

from knowledge_hub.core.schema_validator import validate_payload


PACKET_REPORT = Path("eval/knowledgeos/reports/evidence_packet_input_completeness.v1.json")
READBACK_REPORT = Path("eval/knowledgeos/reports/paper_understanding_readback.v1.json")
PROFILE_REPORT = Path("eval/knowledgeos/reports/paper_understanding_profile_readiness.v1.json")


def _module():
    return importlib.import_module("knowledge_hub.ai.paper_answer_quality_harness")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_answer_quality_harness_scores_bounded_eight_rows_ready() -> None:
    module = _module()

    report = module.build_paper_answer_quality_harness(
        packet_input_report=_load(PACKET_REPORT),
        readback_report=_load(READBACK_REPORT),
        profile_readiness_report=_load(PROFILE_REPORT),
        generated_at="2026-06-09T00:00:00Z",
    )

    assert report["schema"] == module.PAPER_ANSWER_QUALITY_HARNESS_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["counts"] == {
        "rowCount": 8,
        "answerQualityReadyRows": 6,
        "notApplicableRows": 2,
        "blockedRows": 0,
        "singlePaperReadyRows": 2,
        "synthesisReadyRows": 2,
        "compareReadyRows": 2,
        "abstentionRows": 2,
        "inlineCitationReadyRows": 6,
        "semanticCitationSupportRows": 6,
        "semanticCitationFailRows": 0,
        "twoSidedCompareCitationRows": 2,
        "compareMeaningfulSynthesisRows": 2,
        "compareRestatementOnlyRows": 0,
        "insufficientEvidenceRows": 2,
        "unexpectedAnswerRows": 0,
        "unsupportedInventionRows": 0,
        "sampleDraftRows": 8,
        "humanReadableReviewRows": 8,
        "privatePathLeakRows": 0,
        "forbiddenRawMarkerRows": 0,
        "rawAnswerPersistedRows": 0,
        "externalModelCallRows": 0,
        "modelApiCallRows": 0,
        "dbVectorMutationRows": 0,
        "vaultReadRows": 0,
        "defaultPromotionRows": 0,
        "schemaViolationCount": 0,
    }
    assert validate_payload(report, module.PAPER_ANSWER_QUALITY_HARNESS_SCHEMA_ID, strict=True).ok

    for row in report["rows"]:
        assert "answerText" not in row
        assert "rawPrompt" not in row
        assert "/Users/" not in json.dumps(row, ensure_ascii=False, sort_keys=True)
        assert row["answerHash"].startswith("sha256:")
        assert row["humanReadableReview"]["status"] in {"usable", "insufficient"}


def test_answer_quality_harness_blocks_missing_required_input() -> None:
    module = _module()

    report = module.build_paper_answer_quality_harness(
        packet_input_report={"rows": []},
        readback_report={
            "rows": [
                {
                    "runId": "single_missing__A_hybrid_k5",
                    "caseId": "single_missing",
                    "variantId": "A_hybrid_k5",
                    "status": "ready",
                    "expectedSourceIds": ["9999.99999"],
                    "paperReadbacks": [],
                    "warnings": [],
                }
            ]
        },
        profile_readiness_report={"rows": []},
        generated_at="2026-06-09T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["rowCount"] == 1
    assert report["counts"]["answerQualityReadyRows"] == 0
    assert report["counts"]["blockedRows"] == 1
    assert report["counts"]["unexpectedAnswerRows"] == 0
    assert report["rows"][0]["draftStatus"] == "insufficient_evidence"


def test_answer_quality_harness_rejects_citation_presence_without_claim_support() -> None:
    module = _module()
    readback_report = deepcopy(_load(READBACK_REPORT))
    run_id = "single_paper_attention__A_hybrid_k5"

    report = module.build_paper_answer_quality_harness(
        packet_input_report=_load(PACKET_REPORT),
        readback_report=readback_report,
        profile_readiness_report=_load(PROFILE_REPORT),
        generated_at="2026-06-09T00:00:00Z",
        draft_overrides={
            run_id: {
                "answer": "The Transformer primarily depends on recurrence for sequence modeling [S1].",
                "citationLabels": ["S1"],
                "citationSupport": {"S1": False},
            }
        },
    )

    row = next(item for item in report["rows"] if item["runId"] == run_id)
    assert row["status"] == "blocked"
    assert row["inlineCitationReady"] is True
    assert row["semanticCitationSupported"] is False
    assert "semantic_citation_support_failed" in row["warnings"]
    assert report["counts"]["semanticCitationFailRows"] == 1
    assert report["counts"]["answerQualityReadyRows"] == 5
    assert report["status"] == "blocked"


def test_answer_quality_harness_rejects_compare_restatement_only_output() -> None:
    module = _module()
    run_id = "compare_rag_fid__A_hybrid_k5"

    report = module.build_paper_answer_quality_harness(
        packet_input_report=_load(PACKET_REPORT),
        readback_report=_load(READBACK_REPORT),
        profile_readiness_report=_load(PROFILE_REPORT),
        generated_at="2026-06-09T00:00:00Z",
        draft_overrides={
            run_id: {
                "answer": "RAG uses retrieval [S1]. The survey describes RAG categories [S2].",
                "citationLabels": ["S1", "S2"],
                "citationSupport": {"S1": True, "S2": True},
                "meaningfulSynthesis": False,
            }
        },
    )

    row = next(item for item in report["rows"] if item["runId"] == run_id)
    assert row["status"] == "blocked"
    assert row["twoSidedCompareCitationReady"] is True
    assert row["compareMeaningfulSynthesis"] is False
    assert "compare_restatement_only" in row["warnings"]
    assert report["counts"]["compareRestatementOnlyRows"] == 1
    assert report["counts"]["compareReadyRows"] == 1
    assert report["status"] == "blocked"

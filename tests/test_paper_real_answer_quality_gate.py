from __future__ import annotations

import importlib
import json

from knowledge_hub.core.schema_registry_extensions import SCHEMA_NAME_BY_ID_EXTENSIONS
from knowledge_hub.core.schema_validator import validate_payload


def _module():
    return importlib.import_module("knowledge_hub.ai.paper_real_answer_quality_gate")


def _bounded_real_answer_payload():
    return {
        "runMetadata": {
            "querySet": "bounded-real-answer-quality-v0",
            "collector": "task-9-live-smoke",
            "runDirectory": ".omo/evidence/knowledgeos-answer-quality-convergence-20260609",
            "answerRoute": "codex",
            "allowExternal": True,
        },
        "rows": [
            {
                "runId": "single_transformer__real_smoke",
                "caseId": "single_transformer",
                "variantId": "real_smoke",
                "answerType": "single_paper",
                "answerText": "The Transformer replaces recurrent sequence modeling with self-attention [S1].",
                "expectedSourceIds": ["1706.03762"],
                "citations": [{"label": "S1", "sourceId": "1706.03762"}],
                "claimCitationMap": [{"claimId": "c1", "citationLabels": ["S1"], "supported": True}],
            },
            {
                "runId": "synthesis_rag__real_smoke",
                "caseId": "synthesis_rag",
                "variantId": "real_smoke",
                "answerType": "synthesis",
                "answerText": "The packet frames retrieval as useful when generation needs grounded context [S1].",
                "expectedSourceIds": ["2005.11401"],
                "citations": [{"label": "S1", "sourceId": "2005.11401"}],
                "claimCitationMap": [{"claimId": "c1", "citationLabels": ["S1"], "supported": True}],
            },
            {
                "runId": "compare_rag_fid__real_smoke",
                "caseId": "compare_rag_fid",
                "variantId": "real_smoke",
                "answerType": "compare",
                "answerText": "Both rows use retrieval, but RAG retrieves passages for generation while FiD fuses retrieved evidence in the decoder [S1] [S2].",
                "expectedSourceIds": ["2005.11401", "2007.01282"],
                "citations": [{"label": "S1", "sourceId": "2005.11401"}, {"label": "S2", "sourceId": "2007.01282"}],
                "claimCitationMap": [{"claimId": "c1", "citationLabels": ["S1", "S2"], "supported": True}],
                "compareReview": {"meaningfulSynthesis": True, "restatementOnly": False},
            },
            {
                "runId": "abstain_missing__real_smoke",
                "caseId": "abstain_missing",
                "variantId": "real_smoke",
                "answerType": "abstain",
                "answerText": "insufficient_evidence: no paper evidence was available.",
                "expectedSourceIds": [],
                "expectedInsufficientEvidence": True,
                "citations": [],
                "claimCitationMap": [],
            },
        ],
    }


def test_real_answer_quality_gate_schema_registered() -> None:
    # Given: a new real-answer report contract that downstream gates must validate.
    schema_id = "knowledge-hub.paper-real-answer-quality-gate.v1"

    # When: the schema registry is inspected.
    registered = SCHEMA_NAME_BY_ID_EXTENSIONS

    # Then: the schema is available through the same validator as the deterministic harness.
    assert registered[schema_id] == "paper-real-answer-quality-gate.v1.json"


def test_real_answer_quality_gate_scores_bounded_real_payloads_ready() -> None:
    module = _module()

    # Given: four bounded real-answer payload rows covering single, synthesis, compare, and abstain.
    payload = _bounded_real_answer_payload()

    # When: the report-only real answer gate is built.
    report = module.build_paper_real_answer_quality_gate(
        answer_payload_report=payload,
        generated_at="2026-06-09T00:00:00Z",
    )

    # Then: the gate proves answer quality without persisting raw prompts or raw answers.
    assert report["schema"] == module.PAPER_REAL_ANSWER_QUALITY_GATE_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["allowExternal"] is True
    assert report["counts"] == {
        "rowCount": 4,
        "answerableRows": 3,
        "readyRows": 3,
        "notApplicableRows": 1,
        "blockedRows": 0,
        "singlePaperReadyRows": 1,
        "synthesisReadyRows": 1,
        "compareReadyRows": 1,
        "abstentionRows": 1,
        "inlineCitationReadyRows": 3,
        "semanticCitationSupportRows": 3,
        "semanticCitationFailRows": 0,
        "twoSidedCompareCitationRows": 1,
        "compareMeaningfulSynthesisRows": 1,
        "compareRestatementOnlyRows": 0,
        "insufficientEvidenceRows": 1,
        "unexpectedAnswerRows": 0,
        "unsupportedInventionRows": 0,
        "sampleDraftRows": 4,
        "humanReadableReviewRows": 4,
        "privatePathLeakRows": 0,
        "forbiddenRawMarkerRows": 0,
        "rawPromptPersistedRows": 0,
        "rawAnswerPersistedRows": 0,
        "externalModelCallRows": 1,
        "modelApiCallRows": 1,
        "dbVectorMutationRows": 0,
        "vaultReadRows": 0,
        "defaultPromotionRows": 0,
        "schemaViolationCount": 0,
    }
    assert validate_payload(report, module.PAPER_REAL_ANSWER_QUALITY_GATE_SCHEMA_ID, strict=True).ok
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert "answerText" not in serialized
    assert '"rawPrompt":' not in serialized
    assert "/Users/" not in serialized


def test_real_answer_quality_gate_rejects_citation_presence_without_claim_support() -> None:
    module = _module()
    payload = _bounded_real_answer_payload()
    payload["rows"][0]["claimCitationMap"] = [{"claimId": "c1", "citationLabels": ["S1"], "supported": False}]

    # When: a row has inline citation text but the claim map marks it unsupported.
    report = module.build_paper_real_answer_quality_gate(
        answer_payload_report=payload,
        generated_at="2026-06-09T00:00:00Z",
    )

    # Then: citation presence alone is not enough to pass the gate.
    row = report["rows"][0]
    assert report["status"] == "blocked"
    assert row["inlineCitationReady"] is True
    assert row["semanticCitationSupported"] is False
    assert "semantic_citation_support_failed" in row["warnings"]
    assert report["counts"]["semanticCitationFailRows"] == 1
    assert report["counts"]["readyRows"] == 2


def test_real_answer_quality_gate_rejects_one_sided_compare() -> None:
    module = _module()
    payload = _bounded_real_answer_payload()
    payload["rows"][2]["answerText"] = "RAG uses retrieval while FiD is also a retrieval method [S1]."
    payload["rows"][2]["citations"] = [{"label": "S1", "sourceId": "2005.11401"}]
    payload["rows"][2]["claimCitationMap"] = [{"claimId": "c1", "citationLabels": ["S1"], "supported": True}]

    # When: a compare answer cites only one side of a two-paper comparison.
    report = module.build_paper_real_answer_quality_gate(
        answer_payload_report=payload,
        generated_at="2026-06-09T00:00:00Z",
    )

    # Then: the compare row is blocked even though it has a citation.
    row = report["rows"][2]
    assert report["status"] == "blocked"
    assert row["twoSidedCompareCitationReady"] is False
    assert "compare_two_sided_citation_missing" in row["warnings"]
    assert report["counts"]["twoSidedCompareCitationRows"] == 0
    assert report["counts"]["compareReadyRows"] == 0


def test_real_answer_quality_gate_blocks_unexpected_answer_for_abstain_case() -> None:
    module = _module()
    payload = _bounded_real_answer_payload()
    payload["rows"][3]["answerText"] = "This missing paper definitely proves the requested claim."

    # When: an abstain row receives a confident answer.
    report = module.build_paper_real_answer_quality_gate(
        answer_payload_report=payload,
        generated_at="2026-06-09T00:00:00Z",
    )

    # Then: the gate fails closed instead of accepting a partial answer.
    row = report["rows"][3]
    assert report["status"] == "blocked"
    assert row["status"] == "blocked"
    assert "unexpected_answer_for_abstain" in row["warnings"]
    assert report["counts"]["unexpectedAnswerRows"] == 1


def test_real_answer_quality_gate_accepts_korean_insufficient_evidence_abstain() -> None:
    module = _module()
    payload = _bounded_real_answer_payload()
    payload["rows"][3]["answerText"] = "제공된 근거만으로는 검증 가능한 답변을 생성하기 어렵습니다."

    # When: an abstain row uses the live Korean insufficient-evidence wording.
    report = module.build_paper_real_answer_quality_gate(
        answer_payload_report=payload,
        generated_at="2026-06-09T00:00:00Z",
    )

    # Then: the report treats it as a valid fail-closed abstention.
    row = report["rows"][3]
    assert report["status"] == "ready"
    assert row["status"] == "not_applicable"
    assert row["draftStatus"] == "insufficient_evidence"
    assert report["counts"]["unexpectedAnswerRows"] == 0
    assert report["counts"]["insufficientEvidenceRows"] == 1


def test_real_answer_quality_gate_sanitizes_private_paths_and_raw_prompts() -> None:
    module = _module()
    payload = _bounded_real_answer_payload()
    payload["rows"][0]["answerText"] = "A local file at /Users/won/private/paper.pdf supports the claim [S1]."
    payload["rows"][0]["rawPrompt"] = "Use /Users/won/private/paper.pdf as context."
    payload["runMetadata"]["runDirectory"] = "/Users/won/private/run"

    # When: unsafe local material appears in a real-answer payload.
    report = module.build_paper_real_answer_quality_gate(
        answer_payload_report=payload,
        generated_at="2026-06-09T00:00:00Z",
    )

    # Then: the issue is counted but the report remains sanitized.
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert report["counts"]["forbiddenRawMarkerRows"] == 1
    assert report["counts"]["rawPromptPersistedRows"] == 1
    assert "metadata_private_path_leak" in report["warnings"]
    assert "/Users/" not in serialized
    assert '"rawPrompt":' not in serialized
    assert "Use /Users/won/private/paper.pdf as context." not in serialized

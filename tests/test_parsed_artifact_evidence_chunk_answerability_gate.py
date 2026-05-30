from __future__ import annotations

from knowledge_hub.ai.evidence_assembly import _parsed_artifact_evidence_chunk_answerability_gate


def _gate(**plan):
    return _parsed_artifact_evidence_chunk_answerability_gate(
        query_plan={
            "parsed_artifact_evidence_chunk_adapter": "runtime_v1",
            "parsedArtifactEvidenceChunkAdapter": "runtime_v1",
            **plan,
        },
        evidence_chunk_rows_added=2,
        selected_evidence_count=2,
        selected_result_count=2,
    )


def test_answerability_gate_blocks_expected_no_answer_cases() -> None:
    gate = _gate(
        expectedEvidenceType="section",
        answerabilityExpectation="expected_no_answer",
    )

    assert gate["enabled"] is True
    assert gate["status"] == "blocked"
    assert "answerability_expectation:expected_no_answer" in gate["blockReasons"]
    assert gate["suppressedSelectedEvidenceCount"] == 2


def test_answerability_gate_blocks_structured_evidence_required_cases() -> None:
    gate = _gate(
        expectedEvidenceType="table",
        answerabilityExpectation="answerable",
    )

    assert gate["status"] == "blocked"
    assert "expected_evidence_type_requires_structured_evidence:table" in gate["blockReasons"]


def test_answerability_gate_allows_answerable_section_cases() -> None:
    gate = _gate(
        expectedEvidenceType="section",
        answerabilityExpectation="answerable",
    )

    assert gate["status"] == "allowed"
    assert gate["blockReasons"] == []
    assert gate["suppressedSelectedEvidenceCount"] == 0


def test_answerability_gate_is_disabled_without_runtime_opt_in() -> None:
    gate = _parsed_artifact_evidence_chunk_answerability_gate(
        query_plan={
            "expectedEvidenceType": "table",
            "answerabilityExpectation": "expected_no_answer",
        },
        evidence_chunk_rows_added=2,
        selected_evidence_count=2,
        selected_result_count=2,
    )

    assert gate["enabled"] is False
    assert gate["status"] == "disabled"
    assert gate["blockReasons"] == []

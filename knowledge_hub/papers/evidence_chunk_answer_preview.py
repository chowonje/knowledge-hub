"""Labs-only parsed-artifact evidence chunk answer preview helpers."""

from __future__ import annotations

import inspect
from typing import Any

from knowledge_hub.ai.parsed_artifact_evidence_chunk_runtime_adapter import ADAPTER_OPT_IN_VALUE


PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID = "knowledge-hub.paper.evidence-chunk-answer-preview.result.v1"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_paper_ids(values: Any) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    else:
        try:
            raw_values = list(values or [])
        except TypeError:
            raw_values = [values]
    out: list[str] = []
    for value in raw_values:
        token = _clean_text(value)
        if token and token not in out:
            out.append(token)
    return out


def build_evidence_chunk_query_plan(
    paper_ids: list[str],
    *,
    question_category: str = "",
    expected_evidence_type: str = "",
    answerability_expectation: str = "",
) -> dict[str, Any]:
    resolved_ids = normalize_paper_ids(paper_ids)
    plan = {
        "family": "paper_lookup",
        "source_type": "paper",
        "parsed_artifact_evidence_chunk_adapter": ADAPTER_OPT_IN_VALUE,
        "parsedArtifactEvidenceChunkAdapter": ADAPTER_OPT_IN_VALUE,
        "resolvedPaperIds": resolved_ids,
        "resolved_paper_ids": resolved_ids,
    }
    category = _clean_text(question_category)
    evidence_type = _clean_text(expected_evidence_type)
    expectation = _clean_text(answerability_expectation)
    if category:
        plan["question_category"] = category
        plan["questionCategory"] = category
    if evidence_type:
        plan["expected_evidence_type"] = evidence_type
        plan["expectedEvidenceType"] = evidence_type
    if expectation:
        plan["answerability_expectation"] = expectation
        plan["answerabilityExpectation"] = expectation
    return plan


def _call_generate_answer(searcher: Any, question: str, **kwargs: Any) -> dict[str, Any]:
    generate_answer = getattr(searcher, "generate_answer")
    try:
        signature = inspect.signature(generate_answer)
    except (TypeError, ValueError):
        signature = None
    supported_kwargs = dict(kwargs)
    if signature is not None:
        parameters = signature.parameters
        if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            supported_kwargs = {key: value for key, value in supported_kwargs.items() if key in parameters}
    result = generate_answer(question, **supported_kwargs)
    return result if isinstance(result, dict) else {"answer": str(result), "sources": []}


def _summary_count(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        try:
            return int(payload.get(key) or 0)
        except Exception:
            continue
    return 0


def build_paper_evidence_chunk_answer_preview(
    searcher: Any,
    *,
    question: str,
    paper_ids: list[str],
    question_category: str = "",
    expected_evidence_type: str = "",
    answerability_expectation: str = "",
    top_k: int = 8,
    retrieval_mode: str = "semantic",
    alpha: float = 0.7,
    allow_external: bool = False,
) -> dict[str, Any]:
    cleaned_question = _clean_text(question)
    resolved_ids = normalize_paper_ids(paper_ids)
    if not cleaned_question:
        raise ValueError("question is required")
    if not resolved_ids:
        raise ValueError("at least one paper id is required")
    if bool(allow_external):
        raise ValueError("external model calls are not enabled for this labs preview")

    mode = _clean_text(retrieval_mode).lower() or "semantic"
    if mode not in {"semantic", "keyword", "hybrid"}:
        mode = "semantic"
    try:
        alpha_value = max(0.0, min(1.0, float(alpha)))
    except Exception:
        alpha_value = 0.7
    top_k_value = max(1, int(top_k or 8))
    query_plan = build_evidence_chunk_query_plan(
        resolved_ids,
        question_category=question_category,
        expected_evidence_type=expected_evidence_type,
        answerability_expectation=answerability_expectation,
    )
    metadata_filter = {"arxiv_id": resolved_ids[0]} if len(resolved_ids) == 1 else {}
    answer_payload = _call_generate_answer(
        searcher,
        cleaned_question,
        top_k=top_k_value,
        source_type="paper",
        retrieval_mode=mode,
        alpha=alpha_value,
        allow_external=False,
        ask_v2_mode="claim_first",
        metadata_filter=metadata_filter or None,
        query_plan=query_plan,
    )

    evidence_packet = dict(answer_payload.get("evidencePacket") or {})
    evidence_contract = dict(answer_payload.get("evidencePacketContract") or {})
    adapter_diagnostics = dict(evidence_packet.get("parsedArtifactEvidenceChunkAdapter") or {})
    span_rows = len(list(evidence_contract.get("spans") or []))
    answerable = bool(evidence_packet.get("answerable")) or bool(evidence_contract.get("answerable"))
    selected_evidence_count = _summary_count(evidence_packet, "selectedEvidenceCount", "selected_evidence_count")
    citation_count = _summary_count(evidence_packet, "citationCount", "citation_count")
    preview_status = "ok" if answerable or selected_evidence_count or citation_count else "no_evidence"
    return {
        "schema": PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID,
        "status": preview_status,
        "mode": "labs_opt_in_preview",
        "question": cleaned_question,
        "paperIds": resolved_ids,
        "sourceType": "paper",
        "retrievalMode": mode,
        "alpha": alpha_value,
        "topK": top_k_value,
        "allowExternal": False,
        "queryPlan": {
            "family": query_plan["family"],
            "parsed_artifact_evidence_chunk_adapter": query_plan["parsed_artifact_evidence_chunk_adapter"],
            "parsedArtifactEvidenceChunkAdapter": query_plan["parsedArtifactEvidenceChunkAdapter"],
            "resolvedPaperIds": resolved_ids,
            "questionCategory": _clean_text(query_plan.get("questionCategory")),
            "expectedEvidenceType": _clean_text(query_plan.get("expectedEvidenceType")),
            "answerabilityExpectation": _clean_text(query_plan.get("answerabilityExpectation")),
        },
        "answer": str(answer_payload.get("answer") or ""),
        "answerable": bool(answerable),
        "adapterDiagnostics": adapter_diagnostics,
        "evidencePacketSummary": {
            "answerable": bool(evidence_packet.get("answerable")),
            "selectedEvidenceCount": selected_evidence_count,
            "citationCount": citation_count,
            "adapterStatus": _clean_text(adapter_diagnostics.get("status")),
            "adapterRowsAdded": _summary_count(adapter_diagnostics, "rowsAdded", "rows_added"),
            "adapterCandidateRowsConsidered": _summary_count(
                adapter_diagnostics,
                "candidateRowsConsidered",
                "candidate_rows_considered",
            ),
        },
        "evidencePacketContractSummary": {
            "answerable": bool(evidence_contract.get("answerable")),
            "spanRows": span_rows,
        },
        "citations": list(answer_payload.get("citations") or []),
        "sources": list(answer_payload.get("sources") or []),
        "warnings": list(answer_payload.get("warnings") or []),
        "safety": {
            "labsOnly": True,
            "publicKhubAskChanged": False,
            "defaultMcpAskChanged": False,
            "explicitPaperIdsRequired": True,
            "sourceTypeForced": "paper",
            "externalModelCallsAllowed": False,
        },
    }


__all__ = [
    "PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID",
    "build_evidence_chunk_query_plan",
    "build_paper_evidence_chunk_answer_preview",
    "normalize_paper_ids",
]

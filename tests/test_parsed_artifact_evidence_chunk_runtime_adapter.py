from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from knowledge_hub.ai.answer_contracts import build_evidence_packet_contract
from knowledge_hub.ai.evidence_assembly import EvidenceAssemblyService
from knowledge_hub.ai.parsed_artifact_evidence_chunk_runtime_adapter import ADAPTER_OPT_IN_VALUE
from knowledge_hub.core.models import SearchResult
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
)


class _Collaborator:
    def collect_claim_context(
        self,
        results: list[SearchResult],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return [], [], [], []

    def resolve_parent_context(
        self,
        result: SearchResult,
        doc_cache: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        return {
            "parent_id": str((result.metadata or {}).get("paper_id") or ""),
            "parent_label": str((result.metadata or {}).get("title") or ""),
            "chunk_span": str((result.metadata or {}).get("span_locator") or ""),
            "text": result.document,
        }

    def answer_evidence_item(
        self,
        result: SearchResult,
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        metadata = dict(result.metadata or {})
        return {
            "title": metadata.get("title", ""),
            "source_type": metadata.get("source_type", ""),
            "normalized_source_type": metadata.get("source_type", ""),
            "source_id": metadata.get("source_id", ""),
            "arxiv_id": metadata.get("arxiv_id", ""),
            "citation_target": metadata.get("arxiv_id", ""),
            "source_ref": metadata.get("source_ref", ""),
            "source_content_hash": metadata.get("source_content_hash", ""),
            "span_locator": metadata.get("span_locator", ""),
            "snippet_hash": metadata.get("snippet_hash", ""),
            "excerpt": result.document,
            "score": result.score,
            "semantic_score": result.semantic_score,
            "lexical_score": result.lexical_score,
            "quality_flag": "ok",
            "source_trust_score": 0.95,
        }

    def summarize_answer_signals(
        self,
        evidence: list[dict[str, Any]],
        *,
        contradicting_beliefs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "quality_counts": {"ok": sum(1 for item in evidence if item.get("quality_flag") == "ok")},
            "preferred_sources": sum(1 for item in evidence if item.get("quality_flag") == "ok"),
        }

    def build_answer_context(
        self,
        *,
        filtered: list[SearchResult],
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> str:
        return "\n".join(item.document for item in filtered)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(
    index: int,
    *,
    paper_id: str = "paper-a",
    artifact_type: str = "paragraph",
    source_ref: str | None = None,
    excerpt: str | None = None,
) -> dict[str, Any]:
    text = excerpt or f"{paper_id} method evidence chunk {index} directly supports the question with concrete source text."
    start = index * 100
    end = start + len(text)
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
        "candidateRecordId": f"parsed-artifact-evidence-chunk-candidate:{paper_id}:{artifact_type}:{index}",
        "runId": "test-run",
        "sourceDryRunReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json",
        "sourceCandidateRowId": f"parsed-artifact-evidence-chunk:{paper_id}:{index}",
        "paperId": paper_id,
        "sourceType": "paper",
        "artifactType": artifact_type,
        "sourceRef": source_ref or f"papers_dir/parsed/{paper_id}/document.md",
        "sourceContentHash": "sha256:" + f"{index:064d}"[-64:],
        "locator": {
            "kind": "parsed_document_chars",
            "chars": {"start": start, "end": end, "basis": "parsed_document_text"},
            "page": 1,
        },
        "spanLocator": f"chars:{start}-{end}",
        "excerpt": text,
        "snippetHash": _sha256_text(text),
        "sectionTitle": "Method",
        "sectionPath": ["Method"],
        "evidenceKind": "parsed_artifact_evidence_chunk_candidate",
        "idempotencyKey": f"idempotency:{paper_id}:{index}",
        "candidateOnly": True,
        "answerEvidenceCandidate": True,
        "answerabilityCandidate": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
        "evidenceTier": "parsed_artifact_evidence_chunk_candidate_only",
        "strictBlockers": ["candidate_store_record_not_strict_evidence"],
        "writePolicy": {
            "candidateStoreWrite": True,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "candidateRecordHash": "sha256:" + "a" * 64,
    }


def _write_records(papers_dir: Path, paper_id: str, records: list[dict[str, Any]]) -> None:
    path = papers_dir / "structured_evidence_candidates" / "evidence_chunk" / f"{paper_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _assemble(tmp_path: Path, *, source_type: str = "paper", query_plan: dict[str, Any] | None = None):
    service = EvidenceAssemblyService(_Collaborator(), papers_dir=str(tmp_path))
    return service.assemble(
        query="What method evidence is available?",
        source_type=source_type,
        results=[],
        paper_memory_prefilter={},
        metadata_filter=None,
        query_plan=query_plan,
        query_frame=None,
    )


def test_adapter_opt_in_off_leaves_default_path_unchanged(tmp_path: Path) -> None:
    _write_records(tmp_path, "paper-a", [_record(1), _record(2)])

    packet = _assemble(tmp_path, query_plan={"resolvedPaperIds": ["paper-a"]})

    diagnostics = packet.evidence_packet["parsedArtifactEvidenceChunkAdapter"]
    assert diagnostics["enabled"] is False
    assert diagnostics["status"] == "disabled"
    assert diagnostics["rowsAdded"] == 0
    assert packet.evidence == []
    assert packet.citations == []
    assert packet.evidence_packet["answerable"] is False


def test_adapter_opt_in_adds_bounded_provenance_spans_to_answer_contract(tmp_path: Path) -> None:
    _write_records(tmp_path, "paper-a", [_record(1), _record(2), _record(3)])
    _write_records(tmp_path, "paper-b", [_record(4, paper_id="paper-b", artifact_type="section"), _record(5, paper_id="paper-b")])

    packet = _assemble(
        tmp_path,
        query_plan={
            "parsed_artifact_evidence_chunk_adapter": ADAPTER_OPT_IN_VALUE,
            "resolvedPaperIds": ["paper-a", "paper-b"],
        },
    )

    diagnostics = packet.evidence_packet["parsedArtifactEvidenceChunkAdapter"]
    assert diagnostics["status"] == "applied"
    assert diagnostics["rowsAdded"] == 4
    assert diagnostics["selectedCandidateRecordIds"] == [
        "parsed-artifact-evidence-chunk-candidate:paper-a:paragraph:1",
        "parsed-artifact-evidence-chunk-candidate:paper-a:paragraph:2",
        "parsed-artifact-evidence-chunk-candidate:paper-b:section:4",
        "parsed-artifact-evidence-chunk-candidate:paper-b:paragraph:5",
    ]
    assert [item["source_id"] for item in packet.evidence] == ["paper-a", "paper-a", "paper-b", "paper-b"]
    assert packet.evidence_packet["answerable"] is True
    assert packet.evidence_packet["selectedEvidenceCount"] == 4
    assert packet.evidence_packet["citationCount"] == 4

    contract = build_evidence_packet_contract(
        query="What method evidence is available?",
        retrieval_mode="hybrid",
        pipeline_result=SimpleNamespace(plan=SimpleNamespace(to_dict=lambda: {"queryFrame": {"source_type": "paper"}})),
        evidence_packet=packet,
    )
    assert len(contract["spans"]) == 4
    assert contract["answerable"] is True
    assert contract["coverage"]["excluded_low_provenance"] == 0
    assert contract["spans"][0]["sourceContentHashAvailable"] is True
    assert contract["spans"][0]["spanOffsetAvailable"] is True
    assert validate_payload(contract, "knowledge-hub.evidence-packet.v1", strict=True).ok


def test_adapter_missing_resolved_paper_ids_skips_without_fallback(tmp_path: Path) -> None:
    _write_records(tmp_path, "paper-a", [_record(1)])

    packet = _assemble(
        tmp_path,
        query_plan={"parsedArtifactEvidenceChunkAdapter": ADAPTER_OPT_IN_VALUE},
    )

    diagnostics = packet.evidence_packet["parsedArtifactEvidenceChunkAdapter"]
    assert diagnostics["status"] == "skipped"
    assert diagnostics["skippedReason"] == "resolved_paper_ids_required"
    assert diagnostics["candidateRowsConsidered"] == 0
    assert packet.evidence == []


def test_adapter_non_paper_source_skips(tmp_path: Path) -> None:
    _write_records(tmp_path, "paper-a", [_record(1)])

    packet = _assemble(
        tmp_path,
        source_type="web",
        query_plan={
            "parsed_artifact_evidence_chunk_adapter": ADAPTER_OPT_IN_VALUE,
            "resolvedPaperIds": ["paper-a"],
        },
    )

    diagnostics = packet.evidence_packet["parsedArtifactEvidenceChunkAdapter"]
    assert diagnostics["status"] == "skipped"
    assert diagnostics["skippedReason"] == "source_type_not_paper"
    assert packet.evidence == []


def test_adapter_blocks_invalid_candidate_records_without_adding_evidence(tmp_path: Path) -> None:
    bad_hash = _record(1)
    bad_hash["snippetHash"] = "sha256:" + "0" * 64
    private_path = _record(2, source_ref="/" + "Users" + "/private/document.md")
    unsupported = _record(3, artifact_type="table")
    _write_records(tmp_path, "paper-a", [bad_hash, private_path, unsupported])

    packet = _assemble(
        tmp_path,
        query_plan={
            "parsed_artifact_evidence_chunk_adapter": ADAPTER_OPT_IN_VALUE,
            "resolvedPaperIds": ["paper-a"],
        },
    )

    diagnostics = packet.evidence_packet["parsedArtifactEvidenceChunkAdapter"]
    assert diagnostics["status"] == "blocked"
    assert diagnostics["rowsAdded"] == 0
    assert diagnostics["blockedRows"] == 3
    assert diagnostics["blockedReasons"]["snippet_hash_mismatch"] == 1
    assert diagnostics["blockedReasons"]["private_path_leak"] == 1
    assert diagnostics["blockedReasons"]["unsupported_artifact_type"] == 1
    assert packet.citations == []
    assert packet.evidence_packet["answerable"] is False

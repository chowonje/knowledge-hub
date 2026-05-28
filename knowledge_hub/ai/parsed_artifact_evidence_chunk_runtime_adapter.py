from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.ai.answer_contracts import parse_span_offsets
from knowledge_hub.core.models import SearchResult
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
)


ADAPTER_ID = "parsed_artifact_evidence_chunk_runtime_adapter_v1"
ADAPTER_OPT_IN_VALUE = "runtime_v1"
MAX_ROWS_PER_RESOLVED_PAPER = 2
MAX_ROWS_TOTAL = 4
ALLOWED_ARTIFACT_TYPES = {"section", "paragraph"}
OPT_IN_KEYS = (
    "parsed_artifact_evidence_chunk_adapter",
    "parsedArtifactEvidenceChunkAdapter",
)
PRIVATE_PATH_TOKENS = (
    "/" + "Users" + "/",
    "/" + "Volumes" + "/",
    "Mobile " + "Documents",
    "i" + "Cloud",
)
PRIVATE_PATH_RE = re.compile("|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS), re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceChunkRuntimeAdapterResult:
    results: list[SearchResult]
    evidence: list[dict[str, Any]]
    parent_contexts: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return text.strip("._") or "unknown"


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            warnings.append(f"invalid_jsonl_line:{index}")
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            warnings.append(f"non_object_jsonl_line:{index}")
    return rows, warnings


def _candidate_store_path(papers_dir: str | Path, paper_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "structured_evidence_candidates"
        / "evidence_chunk"
        / f"{_safe_filename(paper_id)}.jsonl"
    )


def _candidate_store_ref(paper_id: str) -> str:
    return f"papers_dir/structured_evidence_candidates/evidence_chunk/{_safe_filename(paper_id)}.jsonl"


def _adapter_enabled(query_plan: dict[str, Any] | None) -> bool:
    plan = dict(query_plan or {})
    return any(_clean_text(plan.get(key)) == ADAPTER_OPT_IN_VALUE for key in OPT_IN_KEYS)


def _resolved_paper_ids(
    *,
    query_plan: dict[str, Any] | None,
    query_frame: dict[str, Any] | None,
    metadata_filter: dict[str, Any] | None,
) -> list[str]:
    plan = dict(query_plan or {})
    frame = dict(query_frame or {})
    scoped = dict(metadata_filter or {})
    values: list[Any] = []
    values.extend([scoped.get("arxiv_id"), scoped.get("paper_id")])
    for key in ("resolved_source_ids", "resolvedSourceIds"):
        values.extend(list(frame.get(key) or []))
    for key in (
        "resolvedPaperIds",
        "resolved_paper_ids",
        "resolvedSourceIds",
        "resolved_source_ids",
        "paperIds",
        "paper_ids",
        "sourceIds",
        "source_ids",
    ):
        values.extend(list(plan.get(key) or []))
    out: list[str] = []
    for value in values:
        token = _clean_text(value)
        if token and token not in out:
            out.append(token)
    return out


def _record_locator(record: dict[str, Any]) -> tuple[int | None, int | None, str]:
    span_locator = _clean_text(record.get("spanLocator") or record.get("span_locator"))
    parsed_start, parsed_end = parse_span_offsets(span_locator)
    locator = dict(record.get("locator") or {})
    chars = dict(locator.get("chars") or {})
    start = chars.get("start")
    end = chars.get("end")
    try:
        start_int = int(start)
    except Exception:
        start_int = parsed_start
    try:
        end_int = int(end)
    except Exception:
        end_int = parsed_end
    if not span_locator and start_int is not None and end_int is not None and end_int > start_int:
        span_locator = f"chars:{start_int}-{end_int}"
    return start_int, end_int, span_locator


def _record_blockers(record: dict[str, Any], *, paper_id: str) -> list[str]:
    blockers: list[str] = []
    if record.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID:
        blockers.append("invalid_candidate_record_schema")
    if _clean_text(record.get("paperId")) != paper_id:
        blockers.append("paper_id_mismatch")
    if _clean_text(record.get("sourceType")) != "paper":
        blockers.append("source_type_not_paper")
    if _clean_text(record.get("artifactType")) not in ALLOWED_ARTIFACT_TYPES:
        blockers.append("unsupported_artifact_type")
    for field_name in ("candidateRecordId", "sourceRef", "sourceContentHash", "snippetHash"):
        if not _clean_text(record.get(field_name)):
            blockers.append(f"{field_name}_missing")
    if _contains_private_path(record):
        blockers.append("private_path_leak")
    source_ref = _clean_text(record.get("sourceRef"))
    if source_ref and not source_ref.startswith("papers_dir/"):
        blockers.append("source_ref_not_sanitized")
    start, end, span_locator = _record_locator(record)
    parsed_start, parsed_end = parse_span_offsets(span_locator)
    if start is None or end is None or parsed_start is None or parsed_end is None or end <= start:
        blockers.append("chars_locator_missing_or_invalid")
    elif start != parsed_start or end != parsed_end:
        blockers.append("locator_chars_mismatch")
    excerpt = _clean_text(record.get("excerpt"))
    if not excerpt:
        blockers.append("excerpt_missing")
    elif _sha256_text(excerpt) != _clean_text(record.get("snippetHash")):
        blockers.append("snippet_hash_mismatch")
    write_policy = dict(record.get("writePolicy") or {})
    required_true = {
        "candidateOnly": record.get("candidateOnly"),
        "answerEvidenceCandidate": record.get("answerEvidenceCandidate"),
        "answerabilityCandidate": record.get("answerabilityCandidate"),
        "candidateStoreWrite": write_policy.get("candidateStoreWrite"),
    }
    required_false = {
        "strictEvidence": record.get("strictEvidence"),
        "citationGrade": record.get("citationGrade"),
        "runtimeEvidence": record.get("runtimeEvidence"),
        "answerVisible": record.get("answerVisible"),
        "sourceSpanCreated": write_policy.get("sourceSpanCreated"),
        "strictEvidenceCreated": write_policy.get("strictEvidenceCreated"),
        "citationGradeEvidenceCreated": write_policy.get("citationGradeEvidenceCreated"),
        "runtimeEvidenceCreated": write_policy.get("runtimeEvidenceCreated"),
        "parserRoutingChanged": write_policy.get("parserRoutingChanged"),
        "answerIntegrationChanged": write_policy.get("answerIntegrationChanged"),
        "databaseMutation": write_policy.get("databaseMutation"),
        "vaultScan": write_policy.get("vaultScan"),
        "reindexOrReembed": write_policy.get("reindexOrReembed"),
        "canonicalParsedArtifactsWritten": write_policy.get("canonicalParsedArtifactsWritten"),
    }
    for field_name, value in required_true.items():
        if value is not True:
            blockers.append(f"{field_name}_not_true")
    for field_name, value in required_false.items():
        if value is not False:
            blockers.append(f"{field_name}_not_false")
    if _clean_text(record.get("evidenceTier")) != "parsed_artifact_evidence_chunk_candidate_only":
        blockers.append("evidence_tier_not_candidate_only")
    return sorted(set(blockers))


def _evidence_item(record: dict[str, Any], *, paper_id: str, record_index: int) -> dict[str, Any]:
    start, end, span_locator = _record_locator(record)
    excerpt = _clean_text(record.get("excerpt"))
    artifact_type = _clean_text(record.get("artifactType"))
    source_ref = _clean_text(record.get("sourceRef"))
    snippet_hash = _clean_text(record.get("snippetHash"))
    section_path = list(record.get("sectionPath") or [])
    section_title = _clean_text(record.get("sectionTitle"))
    return {
        "title": section_title or f"{paper_id} {artifact_type}",
        "source_type": "paper",
        "sourceType": "paper",
        "normalized_source_type": "paper",
        "source_id": paper_id,
        "sourceId": paper_id,
        "arxiv_id": paper_id,
        "paper_id": paper_id,
        "citation_target": paper_id,
        "source_ref": source_ref,
        "sourceRef": source_ref,
        "source_content_hash": _clean_text(record.get("sourceContentHash")),
        "sourceContentHash": _clean_text(record.get("sourceContentHash")),
        "span_locator": span_locator,
        "spanLocator": span_locator,
        "char_start": start,
        "charStart": start,
        "char_end": end,
        "charEnd": end,
        "excerpt": excerpt,
        "text": excerpt,
        "section_path": " > ".join(_clean_text(item) for item in section_path if _clean_text(item)),
        "snippet_hash": snippet_hash,
        "snippetHash": snippet_hash,
        "content_hash": snippet_hash,
        "contentHash": snippet_hash,
        "evidence_kind": "parsed_artifact_evidence_chunk",
        "evidenceKind": "parsed_artifact_evidence_chunk",
        "artifact_type": artifact_type,
        "artifactType": artifact_type,
        "candidateOnly": False,
        "runtimeEvidence": True,
        "answerVisible": True,
        "answerable": True,
        "strictEvidence": False,
        "citationGrade": True,
        "score": 1.0,
        "semantic_score": 0.0,
        "lexical_score": 1.0,
        "retrieval_mode": "parsed_artifact_evidence_chunk_runtime_adapter",
        "quality_flag": "ok",
        "source_trust_score": 0.95,
        "derivative_source": {
            "adapterId": ADAPTER_ID,
            "candidateRecordId": _clean_text(record.get("candidateRecordId")),
            "candidateStoreRef": _candidate_store_ref(paper_id),
            "artifactType": artifact_type,
            "recordIndex": record_index,
        },
        "derivativeSource": {
            "adapterId": ADAPTER_ID,
            "candidateRecordId": _clean_text(record.get("candidateRecordId")),
            "candidateStoreRef": _candidate_store_ref(paper_id),
            "artifactType": artifact_type,
            "recordIndex": record_index,
        },
    }


def _search_result(record: dict[str, Any], *, paper_id: str, record_index: int) -> SearchResult:
    start, end, span_locator = _record_locator(record)
    excerpt = _clean_text(record.get("excerpt"))
    record_id = _clean_text(record.get("candidateRecordId"))
    return SearchResult(
        document=excerpt,
        metadata={
            "title": _clean_text(record.get("sectionTitle")) or f"{paper_id} {_clean_text(record.get('artifactType'))}",
            "source_type": "paper",
            "arxiv_id": paper_id,
            "paper_id": paper_id,
            "source_id": paper_id,
            "source_ref": _clean_text(record.get("sourceRef")),
            "source_content_hash": _clean_text(record.get("sourceContentHash")),
            "snippet_hash": _clean_text(record.get("snippetHash")),
            "span_locator": span_locator,
            "char_start": start,
            "char_end": end,
            "section_path": " > ".join(_clean_text(item) for item in list(record.get("sectionPath") or []) if _clean_text(item)),
            "evidence_kind": "parsed_artifact_evidence_chunk",
            "candidate_record_id": record_id,
        },
        distance=0.0,
        score=1.0,
        document_id=f"{ADAPTER_ID}:{record_id or paper_id}:{record_index}",
        semantic_score=0.0,
        lexical_score=1.0,
        retrieval_mode="parsed_artifact_evidence_chunk_runtime_adapter",
        lexical_extras={
            "normalized_source_type": "paper",
            "source_trust_score": 0.95,
            "ranking_signals": {},
        },
    )


def collect_parsed_artifact_evidence_chunk_runtime_evidence(
    *,
    query_plan: dict[str, Any] | None,
    query_frame: dict[str, Any] | None,
    metadata_filter: dict[str, Any] | None,
    source_type: str | None,
    papers_dir: str | Path | None,
) -> EvidenceChunkRuntimeAdapterResult:
    normalized_source = _clean_text(source_type).lower()
    enabled = _adapter_enabled(query_plan)
    resolved_paper_ids = _resolved_paper_ids(
        query_plan=query_plan,
        query_frame=query_frame,
        metadata_filter=metadata_filter,
    )
    diagnostics: dict[str, Any] = {
        "adapterId": ADAPTER_ID,
        "enabled": enabled,
        "status": "disabled",
        "sourceType": normalized_source or "all",
        "resolvedPaperIds": resolved_paper_ids,
        "candidateRowsConsidered": 0,
        "rowsAdded": 0,
        "blockedRows": 0,
        "blockedReasons": {},
        "skippedReason": "",
        "maxRowsPerResolvedPaper": MAX_ROWS_PER_RESOLVED_PAPER,
        "maxRowsTotal": MAX_ROWS_TOTAL,
        "fallbackToAllPapersAllowed": False,
        "selectedCandidateRecordIds": [],
    }
    if not enabled:
        diagnostics["skippedReason"] = "query_plan_opt_in_not_enabled"
        return EvidenceChunkRuntimeAdapterResult([], [], {}, diagnostics)
    if normalized_source != "paper":
        diagnostics.update({"status": "skipped", "skippedReason": "source_type_not_paper"})
        return EvidenceChunkRuntimeAdapterResult([], [], {}, diagnostics)
    if not resolved_paper_ids:
        diagnostics.update({"status": "skipped", "skippedReason": "resolved_paper_ids_required"})
        return EvidenceChunkRuntimeAdapterResult([], [], {}, diagnostics)
    if not papers_dir:
        diagnostics.update({"status": "skipped", "skippedReason": "papers_dir_not_configured"})
        return EvidenceChunkRuntimeAdapterResult([], [], {}, diagnostics)

    selected_results: list[SearchResult] = []
    selected_evidence: list[dict[str, Any]] = []
    parent_contexts: dict[str, dict[str, Any]] = {}
    per_paper: Counter[str] = Counter()
    blocked_reasons: Counter[str] = Counter()
    total_records = 0
    total_blocked = 0

    for paper_id in resolved_paper_ids:
        path = _candidate_store_path(papers_dir, paper_id)
        records, warnings = _read_jsonl(path)
        for warning in warnings:
            blocked_reasons[warning] += 1
        total_records += len(records)
        for record_index, record in enumerate(records, start=1):
            if len(selected_evidence) >= MAX_ROWS_TOTAL:
                break
            if per_paper[paper_id] >= MAX_ROWS_PER_RESOLVED_PAPER:
                continue
            blockers = _record_blockers(record, paper_id=paper_id)
            if blockers:
                total_blocked += 1
                for blocker in blockers:
                    blocked_reasons[blocker] += 1
                continue
            evidence_item = _evidence_item(record, paper_id=paper_id, record_index=record_index)
            result = _search_result(record, paper_id=paper_id, record_index=record_index)
            selected_results.append(result)
            selected_evidence.append(evidence_item)
            parent_contexts[result.document_id] = {
                "parent_id": paper_id,
                "parent_label": _candidate_store_ref(paper_id),
                "chunk_span": evidence_item["span_locator"],
                "text": evidence_item["excerpt"],
            }
            per_paper[paper_id] += 1
        if len(selected_evidence) >= MAX_ROWS_TOTAL:
            break

    diagnostics.update(
        {
            "candidateRowsConsidered": total_records,
            "rowsAdded": len(selected_evidence),
            "blockedRows": total_blocked
            + sum(
                blocked_reasons[warning]
                for warning in blocked_reasons
                if warning.startswith("invalid_jsonl_line") or warning.startswith("non_object_jsonl_line")
            ),
            "blockedReasons": dict(sorted(blocked_reasons.items())),
            "selectedCandidateRecordIds": [
                _clean_text(item.get("derivative_source", {}).get("candidateRecordId"))
                for item in selected_evidence
            ],
        }
    )
    if selected_evidence:
        diagnostics["status"] = "applied"
    elif total_records > 0 or blocked_reasons:
        diagnostics["status"] = "blocked"
        diagnostics["skippedReason"] = "no_valid_candidate_rows_selected"
    else:
        diagnostics["status"] = "skipped"
        diagnostics["skippedReason"] = "candidate_store_records_missing"
    return EvidenceChunkRuntimeAdapterResult(
        results=selected_results,
        evidence=selected_evidence,
        parent_contexts=parent_contexts,
        diagnostics=diagnostics,
    )


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_OPT_IN_VALUE",
    "ALLOWED_ARTIFACT_TYPES",
    "MAX_ROWS_PER_RESOLVED_PAPER",
    "MAX_ROWS_TOTAL",
    "EvidenceChunkRuntimeAdapterResult",
    "collect_parsed_artifact_evidence_chunk_runtime_evidence",
]

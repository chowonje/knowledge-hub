from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review import (
    KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID,
    READY_DECISION as POSITIVE_COMPLETE_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_searcher_ingress_live_smoke import (
    DEFAULT_PAPERS_DIR,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_PROMOTION_AUDIT_SCHEMA_ID = (
    "knowledge-hub.product.parsed-artifact-evidence-chunk-promotion-audit.v1"
)
CANDIDATE_RECORD_SCHEMA_ID = "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-record.v1"
PROMOTION_BLOCKED = "promotion_blocked"
PROMOTION_CANDIDATE_NARROW_SCOPE = "promotion_candidate_narrow_scope"
PROMOTION_READY_FOR_IMPLEMENTATION = "promotion_ready_for_implementation"
NEXT_BLOCKED = "parsed_artifact_evidence_chunk_candidate_store_repair"
NEXT_NARROW = "parsed_artifact_evidence_chunk_narrow_scope_promotion_design"
NEXT_READY = "parsed_artifact_evidence_chunk_default_promotion_implementation"
DEFAULT_POSITIVE_COMPLETE_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review.v1.json"
)
_PRIVATE_PATH_PATTERNS = ("/" + "Users/", "/" + "Volumes/", "Mobile Documents", "iCloud")
PRIVATE_PATH_RE = re.compile("|".join(re.escape(pattern) for pattern in _PRIVATE_PATH_PATTERNS), re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _valid_sha256(value: Any) -> bool:
    token = _clean_text(value)
    return token.startswith("sha256:") and len(token) == 71


def _valid_chars_locator(record: dict[str, Any]) -> bool:
    locator = _clean_text(record.get("spanLocator") or record.get("span_locator"))
    match = re.fullmatch(r"chars:(\d+)-(\d+)", locator)
    if match is None:
        return False
    start = int(match.group(1))
    end = int(match.group(2))
    chars = dict(dict(record.get("locator") or {}).get("chars") or {})
    try:
        char_start = int(chars.get("start"))
        char_end = int(chars.get("end"))
    except (TypeError, ValueError):
        return False
    return start >= 0 and end > start and char_start == start and char_end == end


def _valid_snippet_hash(record: dict[str, Any]) -> bool:
    excerpt = _clean_text(record.get("excerpt"))
    return _clean_text(record.get("snippetHash")) == "sha256:" + hashlib.sha256(excerpt.encode("utf-8")).hexdigest()


def _contains_private_path(value: Any) -> bool:
    return bool(PRIVATE_PATH_RE.search(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _candidate_blockers(record: dict[str, Any], paper_id: str) -> list[str]:
    blockers: list[str] = []
    if record.get("schema") != CANDIDATE_RECORD_SCHEMA_ID:
        blockers.append("candidate_schema_mismatch")
    if _clean_text(record.get("paperId")) != paper_id:
        blockers.append("paper_id_mismatch")
    if _clean_text(record.get("sourceType")) != "paper":
        blockers.append("source_type_not_paper")
    if _clean_text(record.get("artifactType")) not in {"section", "paragraph"}:
        blockers.append("unsupported_artifact_type")
    if not _valid_sha256(record.get("sourceContentHash")):
        blockers.append("source_content_hash_invalid")
    if not _valid_chars_locator(record):
        blockers.append("chars_locator_invalid")
    if not _valid_snippet_hash(record):
        blockers.append("snippet_hash_mismatch")
    if record.get("answerVisible") is True:
        blockers.append("candidate_store_answer_visible")
    if _contains_private_path(record):
        blockers.append("private_path_leak")
    return blockers


def _candidate_store_audit(papers_dir: str | Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    root = Path(papers_dir).expanduser() / "structured_evidence_candidates" / "evidence_chunk"
    per_paper: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    blocker_papers: dict[str, set[str]] = defaultdict(set)
    for path in sorted(root.glob("*.jsonl")):
        paper_id = path.stem
        counts["papersEvaluated"] += 1
        file_rows = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                counts["candidateRows"] += 1
                blocker_counts["invalid_jsonl"] += 1
                blocker_papers["invalid_jsonl"].add(paper_id)
                continue
            if not isinstance(payload, dict):
                counts["candidateRows"] += 1
                blocker_counts["non_object_jsonl"] += 1
                blocker_papers["non_object_jsonl"].add(paper_id)
                continue
            file_rows += 1
            counts["candidateRows"] += 1
            if _valid_sha256(payload.get("sourceContentHash")):
                counts["candidateRowsWithValidSourceContentHash"] += 1
                per_paper[paper_id].add("source_hash")
            if _valid_chars_locator(payload):
                counts["candidateRowsWithValidCharLocators"] += 1
                per_paper[paper_id].add("chars_locator")
            if payload.get("answerVisible") is True:
                counts["candidateStoreAnswerVisibleRows"] += 1
            for blocker in _candidate_blockers(payload, paper_id):
                blocker_counts[blocker] += 1
                blocker_papers[blocker].add(paper_id)
        if file_rows:
            counts["papersWithCandidateRows"] += 1
    counts["papersWithValidSourceContentHash"] = sum(1 for flags in per_paper.values() if "source_hash" in flags)
    counts["papersWithValidCharLocators"] = sum(1 for flags in per_paper.values() if "chars_locator" in flags)
    counts["candidateBlockerRows"] = sum(blocker_counts.values())
    counts.setdefault("candidateStoreAnswerVisibleRows", 0)
    blockers = [
        {
            "category": category,
            "rowCount": row_count,
            "paperCount": len(blocker_papers[category]),
            "samplePaperIds": sorted(blocker_papers[category])[:5],
        }
        for category, row_count in sorted(blocker_counts.items())
    ]
    return dict(counts), blockers


def _positive_quality_blockers(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    result = validate_payload(
        report,
        KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID,
        strict=True,
    )
    counts = dict(report.get("counts") or {})
    if not result.ok:
        blockers.append("positive_quality_schema_invalid")
    if report.get("status") != "ready":
        blockers.append("positive_quality_not_ready")
    if report.get("decision") != POSITIVE_COMPLETE_READY_DECISION:
        blockers.append("positive_quality_decision_not_ready")
    if int(counts.get("positiveSectionParagraphQualityCompleteRows") or 0) != 1:
        blockers.append("positive_section_paragraph_quality_incomplete")
    if int(counts.get("positiveAnswerPassRows") or 0) <= 0:
        blockers.append("positive_answer_rows_missing")
    if int(counts.get("sourceContentHashRows") or 0) <= 0:
        blockers.append("positive_source_hash_rows_missing")
    if int(counts.get("charsLocatorRows") or 0) <= 0:
        blockers.append("positive_chars_locator_rows_missing")
    return blockers


def _decision(counts: dict[str, int], blockers: list[dict[str, Any]], quality_blockers: list[str]) -> str:
    if quality_blockers or counts.get("candidateRows", 0) <= 0:
        return PROMOTION_BLOCKED
    if counts.get("candidateRowsWithValidSourceContentHash", 0) <= 0 or counts.get("candidateRowsWithValidCharLocators", 0) <= 0:
        return PROMOTION_BLOCKED
    if counts.get("privatePathLeakRows", 0) > 0:
        return PROMOTION_BLOCKED
    if blockers or counts.get("publicDefaultPromotionHeldRows", 0) > 0:
        return PROMOTION_CANDIDATE_NARROW_SCOPE
    return PROMOTION_READY_FOR_IMPLEMENTATION


def build_parsed_artifact_evidence_chunk_promotion_audit(
    *,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    positive_complete_report: dict[str, Any] | None = None,
    positive_complete_report_path: str | Path = DEFAULT_POSITIVE_COMPLETE_REPORT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    positive_report = positive_complete_report or _read_json(positive_complete_report_path)
    counts, blockers = _candidate_store_audit(papers_dir)
    quality_counts = dict(positive_report.get("counts") or {})
    quality_blockers = _positive_quality_blockers(positive_report)
    counts.update(
        {
            "positiveSeedRows": int(quality_counts.get("positiveSeedRows") or 0),
            "positiveAnswerPassRows": int(quality_counts.get("positiveAnswerPassRows") or 0),
            "provenancePassRows": int(quality_counts.get("provenancePassRows") or 0),
            "positiveStrictProvenanceSpanRows": int(quality_counts.get("strictProvenanceSpanRows") or 0),
            "positiveSourceContentHashRows": int(quality_counts.get("sourceContentHashRows") or 0),
            "positiveCharsLocatorRows": int(quality_counts.get("charsLocatorRows") or 0),
            "answerVisibleEvidenceRows": int(quality_counts.get("strictProvenanceSpanRows") or 0),
            "publicDefaultPromotionHeldRows": int(quality_counts.get("publicDefaultPromotionHeldRows") or 0),
            "publicDefaultPromotionReadyRows": int(quality_counts.get("publicDefaultPromotionReadyRows") or 0),
            "schemaViolationCount": int(quality_counts.get("schemaViolationCount") or 0),
        }
    )
    counts["privatePathLeakRows"] = sum(row["rowCount"] for row in blockers if row["category"] == "private_path_leak")
    decision = _decision(counts, blockers, quality_blockers)
    status = "blocked" if decision == PROMOTION_BLOCKED else "ready"
    next_tranche = {PROMOTION_BLOCKED: NEXT_BLOCKED, PROMOTION_CANDIDATE_NARROW_SCOPE: NEXT_NARROW}.get(
        decision,
        NEXT_READY,
    )
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_PROMOTION_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": decision,
        "nextRecommendedTranche": next_tranche,
        "inputs": {
            "papersDirRef": "papers_dir",
            "candidateStoreRef": "papers_dir/structured_evidence_candidates/evidence_chunk/*.jsonl",
            "positiveCompleteReportRef": str(positive_complete_report_path),
        },
        "counts": counts,
        "gate": {
            "reportOnly": True,
            "defaultRuntimeMutationAllowed": False,
            "defaultKhubAskChanged": False,
            "vectorDbMutationAllowed": False,
            "vaultAccessAllowed": False,
            "positiveQualityReady": not quality_blockers,
            "defaultPromotionBlocked": decision != PROMOTION_READY_FOR_IMPLEMENTATION,
            "narrowScopePromotionCandidate": decision == PROMOTION_CANDIDATE_NARROW_SCOPE,
            "semanticViolations": quality_blockers,
        },
        "blockersByCategory": blockers,
        "warnings": [
            "candidate rows remain candidate-store records; answer-visible evidence is proven only through opt-in runtime rows",
            "default promotion remains held unless a later implementation gate explicitly changes the default surface",
        ],
    }


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_PROMOTION_AUDIT_SCHEMA_ID",
    "PROMOTION_BLOCKED",
    "PROMOTION_CANDIDATE_NARROW_SCOPE",
    "PROMOTION_READY_FOR_IMPLEMENTATION",
    "build_parsed_artifact_evidence_chunk_promotion_audit",
]

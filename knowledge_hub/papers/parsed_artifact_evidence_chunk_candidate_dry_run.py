"""Report-only parsed-artifact evidence chunk candidate dry-run.

This helper consumes the parsed-artifact evidence chunk contract review and
scans local parsed artifacts for section/paragraph chunk candidates. It does
not create SourceSpan, StrictEvidence, runtime evidence, or index records.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
    sanitized_report_ref,
)
from knowledge_hub.papers.paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review import (
    PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_CONTRACT_REVIEW_SCHEMA_ID,
    READY_DECISION as CONTRACT_REVIEW_READY_DECISION,
    load_json,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-dry-run.v1"
)
PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-row.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_candidate_dry_run_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_candidate_canary_apply_readback"
NEXT_TRANCHE_HOLD = "parsed_artifact_evidence_chunk_candidate_dry_run_repair"

DEFAULT_PAPERS_DIR = Path.home() / ".khub" / "papers"
DEFAULT_MAX_ROWS_PER_PAPER = 4
DEFAULT_MAX_TOTAL_ROWS = 1200
MIN_EXCERPT_CHARS = 80
MAX_EXCERPT_CHARS = 1400

SECTION_TITLE_RE = re.compile(
    r"(?:^|\s)(?P<title>(?:\d+(?:\.\d+)*|[A-Z])\s+"
    r"(?:Introduction|Background|Related Work|Model Architecture|Approach|Method|Methods|Experiments|"
    r"Results|Evaluation|Analysis|Limitations|Conclusion|Broader Impacts|Discussion)\b[^.\n]{0,120})",
    re.IGNORECASE,
)

ZERO_COUNTER_FIELDS = (
    "runtimeRouteWriteRows",
    "runtimeConfigMutationRows",
    "operationalSearchIndexQueryRows",
    "runtimeVisibleRows",
    "answerVisibleRows",
    "answerGenerationRows",
    "strictEvidenceRows",
    "citationGradeRows",
    "runtimeEvidenceRows",
    "answerableWithoutTextEvidenceRows",
    "candidateStoreWriteRows",
    "sourceSpanCreatedRows",
    "sourceSpanCandidateCreatedRows",
    "parsedArtifactEvidenceChunkCreatedRows",
    "embeddingCallRows",
    "embeddingVectorWriteRows",
    "vectorIndexWriteRows",
    "productionVectorIndexWriteRows",
    "databaseMutationRows",
    "indexMutationRows",
    "reindexOrReembedRows",
    "parserExecutionRows",
    "canonicalParsedArtifactWriteRows",
    "graphDbWriteRows",
    "ontologyWriteRows",
    "memoryCardWriteRows",
    "clusterWriteRows",
    "vaultScanRows",
    "externalDownloadRows",
    "branchDeletionRows",
    "githubPrMutationRows",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _short_hash(value: str, *, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _contract_blockers(contract_review: dict[str, Any]) -> list[str]:
    counts = dict(contract_review.get("counts") or {})
    blockers: list[str] = []
    if contract_review.get("schema") != PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_CONTRACT_REVIEW_SCHEMA_ID:
        blockers.append("invalid_contract_review_schema")
    if contract_review.get("status") != "ready":
        blockers.append("contract_review_not_ready")
    if contract_review.get("decision") != CONTRACT_REVIEW_READY_DECISION:
        blockers.append("contract_review_invalid_decision")
    if dict(contract_review.get("gate") or {}).get("passed") is not True:
        blockers.append("contract_review_gate_not_passed")
    if _int(counts.get("allowedArtifactTypeRows")) != 5:
        blockers.append("contract_review_allowed_artifact_types_not_5")
    if _int(counts.get("disallowedSourceRows")) != 5:
        blockers.append("contract_review_disallowed_sources_not_5")
    for field in ("blockedRows", "privatePathLeakRows", "schemaViolationCount", *ZERO_COUNTER_FIELDS):
        if _int(counts.get(field)) != 0:
            blockers.append(f"contract_review_has_{field}")
    if _contains_private_path(contract_review):
        blockers.append("contract_review_has_private_path_leak")
    return blockers


def _source_hashes(manifest: dict[str, Any], document_json: dict[str, Any]) -> list[str]:
    parser_meta = dict(manifest.get("parser_meta") or {})
    document_meta = dict(document_json.get("parser_meta") or {})
    values = [
        manifest.get("sourceContentHash"),
        manifest.get("source_content_hash"),
        parser_meta.get("sourceContentHash"),
        parser_meta.get("source_content_hash"),
        document_meta.get("sourceContentHash"),
        document_meta.get("source_content_hash"),
    ]
    out: list[str] = []
    for value in values:
        text = normalize_text(value)
        if text and text not in out:
            out.append(text)
    return out


def _find_span(surface: str, excerpt: str, *, start_at: int = 0) -> tuple[int | None, int | None]:
    if not surface or not excerpt:
        return None, None
    start = surface.find(excerpt, max(0, start_at))
    if start < 0:
        start = surface.find(excerpt)
    if start < 0:
        return None, None
    return start, start + len(excerpt)


def _excerpt(value: str) -> str:
    text = normalize_text(value)
    return text[:MAX_EXCERPT_CHARS].strip()


def _row(
    *,
    paper_id: str,
    artifact_type: str,
    excerpt: str,
    source_content_hash: str,
    document_ref: str,
    page: int | None,
    char_start: int,
    char_end: int,
    section_title: str = "",
    section_path: list[str] | None = None,
) -> dict[str, Any]:
    row_basis = "|".join([paper_id, artifact_type, str(char_start), str(char_end), _sha256_text(excerpt)])
    row = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_ROW_SCHEMA_ID,
        "candidateRowId": "parsed-artifact-evidence-chunk:" + _short_hash(row_basis),
        "status": "candidate_ready",
        "paperId": paper_id,
        "sourceType": "paper",
        "artifactType": artifact_type,
        "sourceRef": document_ref,
        "sourceContentHash": source_content_hash,
        "locator": {
            "kind": "parsed_document_chars",
            "chars": {
                "start": char_start,
                "end": char_end,
                "basis": "parsed_document_text",
            },
            "page": page,
        },
        "spanLocator": f"chars:{char_start}-{char_end}",
        "excerpt": excerpt,
        "snippetHash": _sha256_text(excerpt),
        "sectionTitle": section_title,
        "sectionPath": list(section_path or []),
        "evidenceKind": "parsed_artifact_evidence_chunk_candidate",
        "candidateOnly": True,
        "answerEvidenceCandidate": True,
        "answerabilityCandidate": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
        "blockers": [],
    }
    row["candidateRowHash"] = _sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return row


def _candidate_rows_for_paper(
    paper_dir: Path,
    *,
    max_rows_per_paper: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    paper_id = paper_dir.name
    manifest_path = paper_dir / "manifest.json"
    document_json_path = paper_dir / "document.json"
    document_md_path = paper_dir / "document.md"
    blockers: list[str] = []
    if not manifest_path.is_file():
        return [], ["missing_manifest"]
    if not document_json_path.is_file():
        return [], ["missing_document_json"]
    if not document_md_path.is_file():
        return [], ["missing_document_md"]

    manifest = _read_json(manifest_path)
    document_json = _read_json(document_json_path)
    hashes = _source_hashes(manifest, document_json)
    if not hashes:
        return [], ["missing_source_content_hash"]
    if len(hashes) > 1:
        return [], ["source_content_hash_mismatch"]
    source_content_hash = hashes[0]

    document_text = document_json.get("markdown_text")
    if not isinstance(document_text, str) or not document_text.strip():
        try:
            document_text = document_md_path.read_text(encoding="utf-8")
        except Exception:
            document_text = ""
    if not document_text.strip():
        return [], ["empty_document_text"]

    elements = document_json.get("elements")
    if not isinstance(elements, list):
        return [], ["missing_document_elements"]

    rows: list[dict[str, Any]] = []
    last_search_start = 0
    document_ref = f"papers_dir/parsed/{paper_id}/document.md"
    for element in elements:
        if len(rows) >= max_rows_per_paper:
            break
        if not isinstance(element, dict):
            continue
        text = _excerpt(_safe_text(element.get("text")))
        if len(text) < MIN_EXCERPT_CHARS:
            continue
        page = _int(element.get("page")) or None
        char_start, char_end = _find_span(document_text, text, start_at=last_search_start)
        if char_start is None or char_end is None:
            blockers.append("paragraph_locator_not_found")
            continue
        last_search_start = char_end
        heading_path = [str(item) for item in list(element.get("heading_path") or []) if str(item).strip()]
        rows.append(
            _row(
                paper_id=paper_id,
                artifact_type="paragraph",
                excerpt=text,
                source_content_hash=source_content_hash,
                document_ref=document_ref,
                page=page,
                char_start=char_start,
                char_end=char_end,
                section_path=heading_path,
            )
        )
        if len(rows) >= max_rows_per_paper:
            break
        match = SECTION_TITLE_RE.search(text)
        if not match:
            continue
        section_title = normalize_text(match.group("title"))
        section_excerpt = text[match.start("title") :].strip()[:MAX_EXCERPT_CHARS]
        if len(section_excerpt) < MIN_EXCERPT_CHARS:
            continue
        sec_start, sec_end = _find_span(document_text, section_excerpt, start_at=char_start)
        if sec_start is None or sec_end is None:
            blockers.append("section_locator_not_found")
            continue
        rows.append(
            _row(
                paper_id=paper_id,
                artifact_type="section",
                excerpt=section_excerpt,
                source_content_hash=source_content_hash,
                document_ref=document_ref,
                page=page,
                char_start=sec_start,
                char_end=sec_end,
                section_title=section_title,
                section_path=[*heading_path, section_title],
            )
        )
    return rows, blockers


def _scan_papers(
    papers_dir: Path,
    *,
    max_rows_per_paper: int,
    max_total_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    parsed_root = papers_dir.expanduser() / "parsed"
    if not parsed_root.is_dir():
        return [], {"parsedArtifactRows": 0, "missingParsedRootRows": 1}, ["missing_parsed_root"]
    rows: list[dict[str, Any]] = []
    counts = {
        "parsedArtifactRows": 0,
        "paperRowsWithCandidates": 0,
        "heldCandidateRows": 0,
        "blockedMissingManifestRows": 0,
        "blockedMissingDocumentJsonRows": 0,
        "blockedMissingDocumentMdRows": 0,
        "blockedMissingSourceHashRows": 0,
        "blockedSourceHashMismatchRows": 0,
        "blockedMissingLocatorRows": 0,
    }
    warnings: list[str] = []
    for paper_dir in sorted(path for path in parsed_root.iterdir() if path.is_dir()):
        counts["parsedArtifactRows"] += 1
        paper_rows, paper_blockers = _candidate_rows_for_paper(
            paper_dir,
            max_rows_per_paper=max_rows_per_paper,
        )
        for blocker in paper_blockers:
            if blocker == "missing_manifest":
                counts["blockedMissingManifestRows"] += 1
            elif blocker == "missing_document_json":
                counts["blockedMissingDocumentJsonRows"] += 1
            elif blocker == "missing_document_md":
                counts["blockedMissingDocumentMdRows"] += 1
            elif blocker == "missing_source_content_hash":
                counts["blockedMissingSourceHashRows"] += 1
            elif blocker == "source_content_hash_mismatch":
                counts["blockedSourceHashMismatchRows"] += 1
            elif "locator_not_found" in blocker:
                counts["blockedMissingLocatorRows"] += 1
            else:
                warnings.append(f"{paper_dir.name}:{blocker}")
        if paper_rows:
            counts["paperRowsWithCandidates"] += 1
        available = max(0, max_total_rows - len(rows))
        if available <= 0:
            counts["heldCandidateRows"] += len(paper_rows)
            continue
        rows.extend(paper_rows[:available])
        counts["heldCandidateRows"] += max(0, len(paper_rows) - available)
    return rows, counts, warnings[:50]


def _counts(
    *,
    rows: list[dict[str, Any]],
    scan_counts: dict[str, int],
    blockers: list[str],
) -> dict[str, int]:
    counts = {
        "inputContractRows": 1,
        "parsedArtifactRows": _int(scan_counts.get("parsedArtifactRows")),
        "missingParsedRootRows": _int(scan_counts.get("missingParsedRootRows")),
        "paperRowsWithCandidates": _int(scan_counts.get("paperRowsWithCandidates")),
        "candidateRows": len(rows) + _int(scan_counts.get("heldCandidateRows")),
        "selectedCandidateRows": len(rows),
        "heldCandidateRows": _int(scan_counts.get("heldCandidateRows")),
        "paragraphCandidateRows": sum(1 for row in rows if row.get("artifactType") == "paragraph"),
        "sectionCandidateRows": sum(1 for row in rows if row.get("artifactType") == "section"),
        "answerEvidenceCandidateRows": sum(1 for row in rows if row.get("answerEvidenceCandidate") is True),
        "answerabilityCandidateRows": sum(1 for row in rows if row.get("answerabilityCandidate") is True),
        "candidateOnlyRows": sum(1 for row in rows if row.get("candidateOnly") is True),
        "blockedMissingManifestRows": _int(scan_counts.get("blockedMissingManifestRows")),
        "blockedMissingDocumentJsonRows": _int(scan_counts.get("blockedMissingDocumentJsonRows")),
        "blockedMissingDocumentMdRows": _int(scan_counts.get("blockedMissingDocumentMdRows")),
        "blockedMissingSourceHashRows": _int(scan_counts.get("blockedMissingSourceHashRows")),
        "blockedSourceHashMismatchRows": _int(scan_counts.get("blockedSourceHashMismatchRows")),
        "blockedMissingLocatorRows": _int(scan_counts.get("blockedMissingLocatorRows")),
        "blockedRows": len(blockers),
        "privatePathLeakRows": sum(1 for blocker in blockers if "private_path" in blocker),
        "schemaViolationCount": 0,
    }
    counts.update({field: 0 for field in ZERO_COUNTER_FIELDS})
    return counts


def _gate(counts: dict[str, int], blockers: list[str]) -> dict[str, Any]:
    checks = {
        "contractReviewReady": not any(blocker.startswith("contract_review") or blocker.startswith("invalid_contract") for blocker in blockers),
        "parsedRootPresent": counts.get("missingParsedRootRows", 0) == 0,
        "hasCandidateRows": counts.get("selectedCandidateRows", 0) > 0,
        "allSelectedRowsCandidateOnly": counts.get("selectedCandidateRows", 0) == counts.get("candidateOnlyRows", 0),
        "hasAnswerEvidenceCandidates": counts.get("answerEvidenceCandidateRows", 0) > 0,
        "hasAnswerabilityCandidates": counts.get("answerabilityCandidateRows", 0) > 0,
        "noBlockedRows": counts.get("blockedRows", 0) == 0,
        "noPrivatePathLeaks": counts.get("privatePathLeakRows", 0) == 0,
        "noMutationOrRuntimeExposure": all(counts.get(field, 0) == 0 for field in ZERO_COUNTER_FIELDS),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "parsedArtifactRows": counts.get("parsedArtifactRows", 0),
            "selectedCandidateRows": counts.get("selectedCandidateRows", 0),
            "paragraphCandidateRows": counts.get("paragraphCandidateRows", 0),
            "sectionCandidateRows": counts.get("sectionCandidateRows", 0),
            "blockedRows": counts.get("blockedRows", 0),
        },
    }


def build_parsed_artifact_evidence_chunk_candidate_dry_run(
    *,
    contract_review: dict[str, Any],
    source_contract_review_ref: str,
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    max_rows_per_paper: int = DEFAULT_MAX_ROWS_PER_PAPER,
    max_total_rows: int = DEFAULT_MAX_TOTAL_ROWS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    blockers = sorted(set(_contract_blockers(contract_review)))
    rows: list[dict[str, Any]] = []
    scan_counts: dict[str, int] = {}
    warnings: list[str] = []
    if not blockers:
        rows, scan_counts, warnings = _scan_papers(
            Path(str(papers_dir)),
            max_rows_per_paper=max(1, int(max_rows_per_paper)),
            max_total_rows=max(1, int(max_total_rows)),
        )
        if _contains_private_path(rows):
            blockers.append("candidate_rows_have_private_path_leak")
        if _contains_private_path(warnings):
            blockers.append("candidate_warnings_have_private_path_leak")
    counts = _counts(rows=rows, scan_counts=scan_counts, blockers=blockers)
    gate = _gate(counts, blockers)
    status = "ready" if gate.get("passed") else "blocked"
    report = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_HOLD,
        "input": {
            "sourceContractReviewRef": normalize_text(source_contract_review_ref),
            "papersDirRef": "papers_dir",
            "maxRowsPerPaper": max(1, int(max_rows_per_paper)),
            "maxTotalRows": max(1, int(max_total_rows)),
        },
        "policy": {
            "dryRunOnly": True,
            "candidateOnly": True,
            "runtimeRouteWrite": False,
            "runtimeConfigMutation": False,
            "operationalSearchIndexQuery": False,
            "answerVisibleExposure": False,
            "answerGeneration": False,
            "strictEvidenceCreation": False,
            "citationGradePromotion": False,
            "runtimeEvidenceCreation": False,
            "sourceSpanCreation": False,
            "sourceSpanCandidateCreation": False,
            "parsedArtifactEvidenceChunkCreation": False,
            "databaseMutation": False,
            "indexMutation": False,
            "reindexOrReembed": False,
            "parserExecution": False,
            "canonicalParsedArtifactWrite": False,
            "vaultScan": False,
            "externalDownload": False,
            "branchDeletion": False,
            "githubMutation": False,
        },
        "sourceContractReview": {
            "schema": normalize_text(contract_review.get("schema")),
            "status": normalize_text(contract_review.get("status")),
            "decision": normalize_text(contract_review.get("decision")),
            "reportRef": normalize_text(source_contract_review_ref),
        },
        "method": {
            "name": "parsed_artifact_evidence_chunk_candidate_dry_run_v1",
            "completionBoundary": "candidate_dry_run_report_only",
            "nextEvidenceWork": NEXT_TRANCHE_READY,
        },
        "counts": counts,
        "gate": gate,
        "candidateRows": rows,
        "technicalBlockers": blockers,
        "warnings": warnings,
    }
    if _contains_private_path(report):
        report["technicalBlockers"] = sorted(set([*blockers, "report_has_private_path_leak"]))
        report["counts"]["privatePathLeakRows"] = 1
        report["counts"]["blockedRows"] = len(report["technicalBlockers"])
        report["gate"] = _gate(report["counts"], report["technicalBlockers"])
        report["status"] = "blocked"
        report["decision"] = BLOCKED_DECISION
        report["nextRecommendedTranche"] = NEXT_TRANCHE_HOLD
    return report


def render_parsed_artifact_evidence_chunk_candidate_dry_run_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Candidate Dry Run",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- parsedArtifactRows: `{counts.get('parsedArtifactRows')}`",
        f"- paperRowsWithCandidates: `{counts.get('paperRowsWithCandidates')}`",
        f"- selectedCandidateRows: `{counts.get('selectedCandidateRows')}`",
        f"- heldCandidateRows: `{counts.get('heldCandidateRows')}`",
        f"- paragraphCandidateRows: `{counts.get('paragraphCandidateRows')}`",
        f"- sectionCandidateRows: `{counts.get('sectionCandidateRows')}`",
        f"- answerEvidenceCandidateRows: `{counts.get('answerEvidenceCandidateRows')}`",
        f"- answerabilityCandidateRows: `{counts.get('answerabilityCandidateRows')}`",
        f"- blockedMissingSourceHashRows: `{counts.get('blockedMissingSourceHashRows')}`",
        f"- blockedSourceHashMismatchRows: `{counts.get('blockedSourceHashMismatchRows')}`",
        f"- blockedMissingLocatorRows: `{counts.get('blockedMissingLocatorRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        f"- answerVisibleRows: `{counts.get('answerVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        "",
        "## Gate",
        "",
        f"- passed: `{gate.get('passed')}`",
        f"- hasCandidateRows: `{dict(gate.get('checks') or {}).get('hasCandidateRows')}`",
        f"- allSelectedRowsCandidateOnly: `{dict(gate.get('checks') or {}).get('allSelectedRowsCandidateOnly')}`",
        f"- noMutationOrRuntimeExposure: `{dict(gate.get('checks') or {}).get('noMutationOrRuntimeExposure')}`",
        "",
        "## Non-Scope",
        "",
        "- No SourceSpan creation.",
        "- No StrictEvidence, citation-grade, runtime evidence, or answer-visible exposure.",
        "- No parser execution, reindex, DB/index mutation, vault scan, or external download.",
    ]
    blockers = list(report.get("technicalBlockers") or [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def write_parsed_artifact_evidence_chunk_candidate_dry_run(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_parsed_artifact_evidence_chunk_candidate_dry_run_markdown(report), encoding="utf-8")
    return {"json": str(report_json), "md": str(report_md)}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_candidate_dry_run",
    "load_json",
    "render_parsed_artifact_evidence_chunk_candidate_dry_run_markdown",
    "sanitized_report_ref",
    "write_parsed_artifact_evidence_chunk_candidate_dry_run",
]

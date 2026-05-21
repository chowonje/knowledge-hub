"""Structured Evidence vertical slice implementation helper.

Validates pilot source_span / strict_evidence readback on selected papers,
generates minimal greenfield section_text_offset records when explicitly
applied, and emits trace validation reports. Does not integrate runtime
answers, run complex QA eval, expand corpus manifest, or implement table or
equation parsers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.application.corpus_artifacts import (
    corpus_entry_ref,
    find_corpus_entry_for_source,
    load_corpus_manifest,
)
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_store_contract import (
    PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_STORE,
)
from knowledge_hub.papers.parsed_artifact_source_span_original_source_offset_authority_design import (
    CHARS_BASIS,
    CHARS_NORMALIZATION,
    _canonical_text_from_pages,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_record_contract import (
    AUTHORITY_TYPE_TEXT_OFFSET,
    PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
    PROMOTION_GATE_ID as LEGACY_PROMOTION_GATE_ID,
    validate_strict_evidence_record_semantics,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_executor_dry_run import (
    compute_contract_substring_sha256,
)
from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import (
    _exact_matches,
    _extract_pdf_pages,
    _normalized_matches,
    _with_offsets,
)
from knowledge_hub.papers.structured_evidence_vertical_slice_discovery import (
    DEFAULT_MANIFEST_PATH,
    PROJECT_ROOT,
    RECOMMENDED_PAPER_IDS,
)


STRUCTURED_EVIDENCE_VERTICAL_SLICE_IMPLEMENTATION_SCHEMA_ID = (
    "knowledge-hub.paper.structured-evidence-vertical-slice-implementation.v1"
)

RUN_ID = "structured-evidence-vertical-slice-20260521"
VERTICAL_SLICE_PROMOTION_GATE_ID = "structured_evidence_vertical_slice_implementation"

PAPER_MODES: dict[str, str] = {
    "1706.03762": "pilot_readback",
    "2005.11401": "greenfield_section",
    "1512.03385": "greenfield_section",
    "1506.02640": "figure_caption_readback",
    "2005.14165": "pilot_readback",
}

DEFERRED_EVIDENCE_TYPES = (
    "table_cell_numeric",
    "equation_citation",
    "appendix_table_lookup",
)

_MINIMAL_EVIDENCE_TYPES = (
    "section_text_offset",
    "figure_caption_text",
)

_PRIVATE_PATH_RE = re.compile(
    rf"({'/' + 'Users/'}|{'Mobile' + ' Documents'}|{re.escape('.' + 'khub')}(?:/|$))",
    re.IGNORECASE,
)

_NO_RUNTIME_WRITE_POLICY = {
    "executorRequired": True,
    "databaseMutation": False,
    "parserRoutingChanged": False,
    "answerIntegrationChanged": False,
    "reindexOrReembed": False,
    "canonicalParsedArtifactsWritten": False,
}

_STRICT_EVIDENCE_WRITE_POLICY = {
    **_NO_RUNTIME_WRITE_POLICY,
    "sourceSpanStoreWrite": False,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_hash(value: Any) -> str:
    text = _clean(value).lower()
    if text and not text.startswith("sha256:"):
        text = f"sha256:{text}"
    return text


def _hash_body(value: Any) -> str:
    return _normalize_hash(value).removeprefix("sha256:")


def _configured_papers_dir(config: Any) -> Path:
    raw = ""
    if hasattr(config, "get_nested"):
        raw = _clean(config.get_nested("storage", "papers_dir", default=""))
    if not raw:
        raw = _clean(getattr(config, "papers_dir", ""))
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ("." + "khub") / "papers"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _read_jsonl_excluding_current_run(path: Path) -> list[dict[str, Any]]:
    return [row for row in _read_jsonl(path) if _clean(row.get("runId")) != RUN_ID]


def _write_jsonl_idempotent(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming_by_key = {
        str(record.get("idempotencyKey") or record.get("sourceSpanId") or record.get("strictEvidenceId")): record
        for record in records
    }
    retained: list[dict[str, Any]] = []
    for existing in _read_jsonl(path):
        key = str(
            existing.get("idempotencyKey")
            or existing.get("sourceSpanId")
            or existing.get("strictEvidenceId")
        )
        if key and key in incoming_by_key:
            continue
        retained.append(existing)
    output = retained + list(incoming_by_key.values())
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output),
        encoding="utf-8",
    )
    return len(incoming_by_key)


def _resolve_corpus_source_pdf(
    *,
    entry: dict[str, Any],
    papers_dir: Path,
) -> tuple[Path | None, str, list[str]]:
    blockers: list[str] = []
    filename = _clean(entry.get("expectedFilename"))
    expected_hash = _normalize_hash(entry.get("expectedSourceContentHash"))
    if not filename:
        blockers.append("expectedFilename_missing")
        return None, expected_hash, blockers
    if not expected_hash:
        blockers.append("expectedSourceContentHash_missing")
        return None, expected_hash, blockers
    pdf_path = papers_dir / filename
    if not pdf_path.is_file():
        blockers.append("corpus_source_pdf_missing")
        return None, expected_hash, blockers
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = f"sha256:{digest.hexdigest()}"
    if observed != expected_hash:
        blockers.append("sourceContentHash_mismatch")
        return pdf_path, expected_hash, blockers
    return pdf_path, expected_hash, blockers


def _load_canonical_context(*, pdf_path: Path) -> tuple[list[dict[str, Any]], str]:
    pages = _with_offsets(_extract_pdf_pages(pdf_path))
    return pages, _canonical_text_from_pages(pages)


def _stable_suffix(*parts: str) -> str:
    joined = "|".join(_clean(part) for part in parts if _clean(part))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _pick_section_match(
    *,
    document: dict[str, Any],
    pages: list[dict[str, Any]],
    canonical_text: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    blockers: list[str] = []
    elements = [item for item in list(document.get("elements") or []) if isinstance(item, dict)]
    candidates: list[dict[str, Any]] = []
    for element in elements:
        if _clean(element.get("type")) != "paragraph":
            continue
        page = int(element.get("page") or 0)
        if page < 2:
            continue
        text = str(element.get("text") or "").strip()
        if len(text) < 150:
            continue
        candidates.append(element)
    if not candidates:
        blockers.append("no_section_paragraph_candidate")
        return None, None, blockers

    for element in sorted(candidates, key=lambda item: (-len(str(item.get("text") or "")), int(item.get("page") or 0))):
        text = str(element.get("text") or "").strip()
        for frac in (0.25, 0.5, 0.75):
            start = int(len(text) * frac)
            needle = text[start : start + 100].strip()
            if len(needle) < 30:
                continue
            for matcher, label in ((_exact_matches, "exact"), (_normalized_matches, "normalized")):
                matches = matcher(pages, needle)
                if len(matches) != 1:
                    continue
                match = matches[0]
                chars_start = int(match["chars_start"])
                chars_end = int(match["chars_end"])
                expected_hash = compute_contract_substring_sha256(
                    canonical_text,
                    chars_start,
                    chars_end,
                    normalization=CHARS_NORMALIZATION,
                )
                return (
                    element,
                    {
                        "matchMethod": label,
                        "charsStart": chars_start,
                        "charsEnd": chars_end,
                        "expectedSubstringSha256": expected_hash,
                        "claimSurface": needle[:120],
                        "verbatimText": canonical_text[chars_start:chars_end],
                    },
                    [],
                )
    blockers.append("section_text_non_unique_or_unmatched")
    return None, None, blockers


def _source_span_record(
    *,
    paper_id: str,
    artifact_type: str,
    source_candidate_id: str,
    source_content_hash: str,
    source_file: str,
    locator: dict[str, Any],
    suffix: str,
    claim_hint: str,
) -> dict[str, Any]:
    source_span_id = f"source-span:{paper_id}:{artifact_type}:{suffix}"
    candidate_record_id = f"source-span-candidate:{paper_id}:{artifact_type}:{suffix}"
    idempotency_key = hashlib.sha256(
        f"{PARSED_ARTIFACT_SOURCE_SPAN_STORE}|{paper_id}|{artifact_type}|{source_candidate_id}|{suffix}".encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
        "sourceSpanId": source_span_id,
        "candidateRecordId": candidate_record_id,
        "runId": RUN_ID,
        "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_STORE,
        "paperId": paper_id,
        "artifactType": artifact_type,
        "sourceCandidateId": source_candidate_id,
        "sourceContentHash": _hash_body(source_content_hash),
        "sourceFile": source_file,
        "locator": locator,
        "idempotencyKey": idempotency_key,
        "evidenceTier": "parsed_artifact_source_span",
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "strictBlockers": [
            "source_span_store_record_not_strict_evidence",
            "runtime_integration_not_allowed",
        ],
        "writePolicy": dict(_NO_RUNTIME_WRITE_POLICY),
        "claimSurfaceHint": claim_hint,
    }


def _strict_evidence_record(
    *,
    paper_id: str,
    artifact_type: str,
    source_span_record: dict[str, Any],
    match: dict[str, Any],
    source_content_hash: str,
    source_file: str,
    parsed_locator_ref: str,
) -> dict[str, Any]:
    source_span_id = _clean(source_span_record.get("sourceSpanId"))
    candidate_record_id = _clean(source_span_record.get("candidateRecordId"))
    suffix = source_span_id.split(":")[-1]
    start = int(match["charsStart"])
    end = int(match["charsEnd"])
    expected_hash = _clean(match["expectedSubstringSha256"])
    strict_evidence_id = f"strict-evidence:{paper_id}:{artifact_type}:{suffix}"
    idempotency_key = (
        f"strict:{source_span_id}:{CHARS_NORMALIZATION}:{start}:{end}:{expected_hash}"
    )
    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID,
        "strictEvidenceId": strict_evidence_id,
        "runId": RUN_ID,
        "plannedWriteTarget": PARSED_ARTIFACT_STRICT_EVIDENCE_STORE,
        "paperId": paper_id,
        "artifactType": artifact_type,
        "claimSurface": _clean(match.get("claimSurface")),
        "sourceSpanIds": [source_span_id],
        "candidateRecordIds": [candidate_record_id],
        "sourceContentHash": _hash_body(source_content_hash),
        "sourceFile": source_file,
        "verbatimText": _clean(match.get("verbatimText")),
        "verbatimSubstringSha256": expected_hash,
        "authority": {
            "type": AUTHORITY_TYPE_TEXT_OFFSET,
            "chars": {
                "start": start,
                "end": end,
                "basis": CHARS_BASIS,
                "normalization": CHARS_NORMALIZATION,
                "expectedSubstringSha256": expected_hash,
            },
        },
        "provenanceTrace": {
            "promotionGateId": VERTICAL_SLICE_PROMOTION_GATE_ID,
            "parsedArtifactLocatorRef": parsed_locator_ref,
            "legacyPromotionGateId": LEGACY_PROMOTION_GATE_ID,
        },
        "designPacketReviewRowId": f"{VERTICAL_SLICE_PROMOTION_GATE_ID}:{paper_id}:{artifact_type}",
        "promotionGateId": VERTICAL_SLICE_PROMOTION_GATE_ID,
        "idempotencyKey": idempotency_key,
        "evidenceTier": "parsed_artifact_strict_evidence",
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "writePolicy": dict(_STRICT_EVIDENCE_WRITE_POLICY),
    }


def _validate_record_flags(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if bool(record.get("runtimeEvidence")):
        errors.append("runtimeEvidence_must_be_false")
    if bool(record.get("citationGrade")):
        errors.append("citationGrade_must_be_false")
    return errors


def _validate_strict_record_trace(
    *,
    record: dict[str, Any],
    expected_hash_body: str,
    canonical_text: str,
    source_span_ids: set[str],
) -> dict[str, Any]:
    errors: list[str] = []
    errors.extend(_validate_record_flags(record))
    if _hash_body(record.get("sourceContentHash")) != expected_hash_body:
        errors.append("sourceContentHash_mismatch")
    authority = record.get("authority") if isinstance(record.get("authority"), dict) else {}
    if _clean(authority.get("type")) != AUTHORITY_TYPE_TEXT_OFFSET:
        errors.append("authority_type_must_be_text_offset")
    errors.extend(validate_strict_evidence_record_semantics(record))
    chars = authority.get("chars") if isinstance(authority.get("chars"), dict) else {}
    try:
        start = int(chars.get("start"))
        end = int(chars.get("end"))
        recomputed = compute_contract_substring_sha256(
            canonical_text,
            start,
            end,
            normalization=CHARS_NORMALIZATION,
        )
        expected = _clean(chars.get("expectedSubstringSha256"))
        if recomputed != expected:
            errors.append("hash_recompute_mismatch")
        if _clean(record.get("verbatimSubstringSha256")) != expected:
            errors.append("verbatim_hash_mismatch")
    except Exception:
        errors.append("authority_chars_invalid")
    linked = False
    for source_span_id in list(record.get("sourceSpanIds") or []):
        if _clean(source_span_id) in source_span_ids:
            linked = True
            break
    if not linked:
        errors.append("sourceSpanIds_not_resolved")
    schema = validate_payload(record, PARSED_ARTIFACT_STRICT_EVIDENCE_RECORD_SCHEMA_ID, strict=True)
    if not schema.ok:
        errors.append("strict_evidence_schema_invalid")
    return {
        "sourceContentHashMatch": "sourceContentHash_mismatch" not in errors,
        "authorityTextOffset": "authority_type_must_be_text_offset" not in errors,
        "runtimeEvidenceFalse": "runtimeEvidence_must_be_false" not in errors,
        "citationGradeFalse": "citationGrade_must_be_false" not in errors,
        "hashRecomputePass": "hash_recompute_mismatch" not in errors and "verbatim_hash_mismatch" not in errors,
        "sourceSpanLinkage": "sourceSpanIds_not_resolved" not in errors,
        "schemaValid": "strict_evidence_schema_invalid" not in errors,
        "pass": not errors,
        "errors": errors,
    }


def _validate_source_span_record_trace(
    *,
    record: dict[str, Any],
    expected_hash_body: str,
) -> dict[str, Any]:
    errors: list[str] = []
    errors.extend(_validate_record_flags(record))
    if _hash_body(record.get("sourceContentHash")) != expected_hash_body:
        errors.append("sourceContentHash_mismatch")
    schema = validate_payload(record, PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID, strict=True)
    if not schema.ok:
        errors.append("source_span_schema_invalid")
    return {
        "sourceContentHashMatch": "sourceContentHash_mismatch" not in errors,
        "runtimeEvidenceFalse": "runtimeEvidence_must_be_false" not in errors,
        "citationGradeFalse": "citationGrade_must_be_false" not in errors,
        "schemaValid": "source_span_schema_invalid" not in errors,
        "pass": not errors,
        "errors": errors,
    }


def _pilot_readback(
    *,
    source_id: str,
    entry: dict[str, Any],
    papers_dir: Path,
    pdf_path: Path,
    expected_hash: str,
) -> dict[str, Any]:
    expected_hash_body = _hash_body(expected_hash)
    pages, canonical_text = _load_canonical_context(pdf_path=pdf_path)
    source_span_path = papers_dir / "structured_evidence" / "source_span" / f"{source_id}.jsonl"
    strict_path = papers_dir / "structured_evidence" / "strict_evidence" / f"{source_id}.jsonl"
    source_spans = _read_jsonl(source_span_path)
    strict_records = _read_jsonl(strict_path)
    source_span_ids = {_clean(item.get("sourceSpanId")) for item in source_spans if _clean(item.get("sourceSpanId"))}

    source_span_checks = [
        _validate_source_span_record_trace(record=item, expected_hash_body=expected_hash_body)
        for item in source_spans
    ]
    strict_checks = [
        _validate_strict_record_trace(
            record=item,
            expected_hash_body=expected_hash_body,
            canonical_text=canonical_text,
            source_span_ids=source_span_ids,
        )
        for item in strict_records
    ]

    section_strict = [item for item in strict_records if _clean(item.get("artifactType")) == "section"]
    figure_strict = [item for item in strict_records if _clean(item.get("artifactType")) == "figure"]
    document = _read_json(papers_dir / "parsed" / source_id / "document.json")
    figure_artifacts = list(document.get("figure_artifacts") or [])

    blockers: list[str] = []
    if not source_spans:
        blockers.append("source_span_store_empty")
    if not strict_records:
        blockers.append("strict_evidence_store_empty")
    if source_span_checks and not all(item["pass"] for item in source_span_checks):
        blockers.append("source_span_trace_failures")
    if strict_checks and not all(item["pass"] for item in strict_checks):
        blockers.append("strict_evidence_trace_failures")

    status = "pass" if not blockers else "blocked"
    return {
        "sourceId": source_id,
        "artifactId": corpus_entry_ref(entry),
        "mode": PAPER_MODES[source_id],
        "status": status,
        "expectedSourceContentHash": expected_hash,
        "parsedArtifactLocator": f"papers_dir/parsed/{source_id}/document.json",
        "existingRecords": {
            "sourceSpanCount": len(source_spans),
            "strictEvidenceCount": len(strict_records),
            "sectionStrictEvidenceCount": len(section_strict),
            "figureStrictEvidenceCount": len(figure_strict),
            "figureArtifactCount": len(figure_artifacts),
        },
        "generatedRecords": {
            "sourceSpanCount": 0,
            "strictEvidenceCount": 0,
        },
        "traceValidation": {
            "sourceSpanRecordsChecked": len(source_span_checks),
            "sourceSpanRecordsPass": sum(1 for item in source_span_checks if item["pass"]),
            "strictEvidenceRecordsChecked": len(strict_checks),
            "strictEvidenceRecordsPass": sum(1 for item in strict_checks if item["pass"]),
            "allSourceSpanTracePass": bool(source_span_checks) and all(item["pass"] for item in source_span_checks),
            "allStrictEvidenceTracePass": bool(strict_checks) and all(item["pass"] for item in strict_checks),
            "citationGradeFalseMaintained": all(
                not bool(item.get("citationGrade")) for item in source_spans + strict_records
            ),
            "runtimeEvidenceFalseMaintained": all(
                not bool(item.get("runtimeEvidence")) for item in source_spans + strict_records
            ),
        },
        "blockers": blockers,
    }


def _figure_caption_readback(
    *,
    source_id: str,
    entry: dict[str, Any],
    papers_dir: Path,
    pdf_path: Path,
    expected_hash: str,
) -> dict[str, Any]:
    pilot = _pilot_readback(
        source_id=source_id,
        entry=entry,
        papers_dir=papers_dir,
        pdf_path=pdf_path,
        expected_hash=expected_hash,
    )
    pilot["mode"] = PAPER_MODES[source_id]
    document = _read_json(papers_dir / "parsed" / source_id / "document.json")
    figure_artifacts = [item for item in list(document.get("figure_artifacts") or []) if isinstance(item, dict)]
    strict_path = papers_dir / "structured_evidence" / "strict_evidence" / f"{source_id}.jsonl"
    figure_strict = [
        item
        for item in _read_jsonl(strict_path)
        if _clean(item.get("artifactType")) == "figure"
    ]
    if figure_artifacts and not figure_strict:
        pilot["blockers"].append("figure_artifacts_without_strict_figure_evidence")
        pilot["status"] = "blocked"
    pilot["figureCaptionReadback"] = {
        "figureArtifactCount": len(figure_artifacts),
        "figureStrictEvidenceCount": len(figure_strict),
        "sampleCaptionPresent": bool(
            figure_artifacts and _clean(figure_artifacts[0].get("caption"))
        ),
    }
    return pilot


def _greenfield_section(
    *,
    source_id: str,
    entry: dict[str, Any],
    papers_dir: Path,
    pdf_path: Path,
    expected_hash: str,
    apply: bool,
) -> dict[str, Any]:
    expected_hash_body = _hash_body(expected_hash)
    source_file = _clean(entry.get("expectedFilename"))
    parsed_locator = f"papers_dir/parsed/{source_id}/document.json"
    document_path = papers_dir / "parsed" / source_id / "document.json"
    document = _read_json(document_path)
    blockers: list[str] = []
    if not document_path.is_file():
        blockers.append("parsed_document_json_missing")
    pages, canonical_text = _load_canonical_context(pdf_path=pdf_path)
    if not canonical_text:
        blockers.append("canonical_text_unavailable")

    element, match, pick_blockers = _pick_section_match(
        document=document,
        pages=pages,
        canonical_text=canonical_text,
    )
    blockers.extend(pick_blockers)

    generated_source_span: dict[str, Any] | None = None
    generated_strict: dict[str, Any] | None = None
    trace_validation: dict[str, Any] = {}

    if element and match and not blockers:
        page = int(element.get("page") or 0)
        reading_order = int(element.get("reading_order") or 0)
        source_candidate_id = f"vertical-slice:{source_id}:paragraph:p{page}:ro{reading_order}"
        suffix = _stable_suffix(source_candidate_id, str(match["charsStart"]), str(match["charsEnd"]))
        generated_source_span = _source_span_record(
            paper_id=source_id,
            artifact_type="section",
            source_candidate_id=source_candidate_id,
            source_content_hash=expected_hash,
            source_file=source_file,
            locator={
                "page": page,
                "bbox": [],
                "blockIndexes": [],
                "chars": {"start": None, "end": None},
            },
            suffix=suffix,
            claim_hint=_clean(match.get("claimSurface")),
        )
        generated_strict = _strict_evidence_record(
            paper_id=source_id,
            artifact_type="section",
            source_span_record=generated_source_span,
            match=match,
            source_content_hash=expected_hash,
            source_file=source_file,
            parsed_locator_ref=parsed_locator,
        )
        source_span_ids = {_clean(generated_source_span.get("sourceSpanId"))}
        trace_validation = _validate_strict_record_trace(
            record=generated_strict,
            expected_hash_body=expected_hash_body,
            canonical_text=canonical_text,
            source_span_ids=source_span_ids,
        )
        source_span_trace = _validate_source_span_record_trace(
            record=generated_source_span,
            expected_hash_body=expected_hash_body,
        )
        trace_validation["sourceSpanTracePass"] = source_span_trace["pass"]
        trace_validation["pass"] = trace_validation["pass"] and source_span_trace["pass"]
        if not trace_validation["pass"]:
            blockers.append("generated_trace_validation_failed")

        if apply and trace_validation["pass"]:
            _write_jsonl_idempotent(
                papers_dir / "structured_evidence" / "source_span" / f"{source_id}.jsonl",
                [generated_source_span],
            )
            _write_jsonl_idempotent(
                papers_dir / "structured_evidence" / "strict_evidence" / f"{source_id}.jsonl",
                [generated_strict],
            )

    status = "generated" if generated_strict and trace_validation.get("pass") else "blocked"
    if blockers:
        status = "blocked"
    return {
        "sourceId": source_id,
        "artifactId": corpus_entry_ref(entry),
        "mode": PAPER_MODES[source_id],
        "status": status,
        "expectedSourceContentHash": expected_hash,
        "parsedArtifactLocator": parsed_locator,
        "existingRecords": {
            "sourceSpanCount": len(
                _read_jsonl_excluding_current_run(
                    papers_dir / "structured_evidence" / "source_span" / f"{source_id}.jsonl"
                )
            ),
            "strictEvidenceCount": len(
                _read_jsonl_excluding_current_run(
                    papers_dir / "structured_evidence" / "strict_evidence" / f"{source_id}.jsonl"
                )
            ),
            "operatorLocalRunIdExcluded": RUN_ID,
        },
        "generatedRecords": {
            "sourceSpanCount": 1 if generated_source_span and trace_validation.get("pass") else 0,
            "strictEvidenceCount": 1 if generated_strict and trace_validation.get("pass") else 0,
            "evidenceType": "section_text_offset" if generated_strict else None,
            "sourceSpanId": _clean((generated_source_span or {}).get("sourceSpanId")) or None,
            "strictEvidenceId": _clean((generated_strict or {}).get("strictEvidenceId")) or None,
            "applied": bool(apply and generated_strict and trace_validation.get("pass")),
        },
        "traceValidation": trace_validation,
        "blockers": blockers,
    }


def _sanitize_for_public(value: Any) -> Any:
    if isinstance(value, str):
        if _PRIVATE_PATH_RE.search(value):
            return "<private-path-redacted>"
        return value
    if isinstance(value, list):
        return [_sanitize_for_public(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_for_public(item) for key, item in value.items()}
    return value


def build_structured_evidence_vertical_slice_implementation(
    *,
    config: Any,
    manifest_path: str | Path | None = None,
    papers_dir: str | Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    manifest = load_corpus_manifest(manifest_path or DEFAULT_MANIFEST_PATH)
    resolved_papers_dir = Path(str(papers_dir)).expanduser() if papers_dir else _configured_papers_dir(config)

    paper_results: list[dict[str, Any]] = []
    for source_id in RECOMMENDED_PAPER_IDS:
        entry = find_corpus_entry_for_source(source_id, manifest)
        if not entry:
            paper_results.append(
                {
                    "sourceId": source_id,
                    "mode": PAPER_MODES.get(source_id, "unknown"),
                    "status": "blocked",
                    "blockers": ["corpus_manifest_entry_missing"],
                }
            )
            continue

        pdf_path, expected_hash, resolve_blockers = _resolve_corpus_source_pdf(
            entry=entry,
            papers_dir=resolved_papers_dir,
        )
        if resolve_blockers or pdf_path is None:
            paper_results.append(
                {
                    "sourceId": source_id,
                    "artifactId": corpus_entry_ref(entry),
                    "mode": PAPER_MODES.get(source_id, "unknown"),
                    "status": "blocked",
                    "expectedSourceContentHash": expected_hash or None,
                    "blockers": resolve_blockers,
                }
            )
            continue

        mode = PAPER_MODES[source_id]
        if mode == "pilot_readback":
            result = _pilot_readback(
                source_id=source_id,
                entry=entry,
                papers_dir=resolved_papers_dir,
                pdf_path=pdf_path,
                expected_hash=expected_hash,
            )
        elif mode == "figure_caption_readback":
            result = _figure_caption_readback(
                source_id=source_id,
                entry=entry,
                papers_dir=resolved_papers_dir,
                pdf_path=pdf_path,
                expected_hash=expected_hash,
            )
        elif mode == "greenfield_section":
            result = _greenfield_section(
                source_id=source_id,
                entry=entry,
                papers_dir=resolved_papers_dir,
                pdf_path=pdf_path,
                expected_hash=expected_hash,
                apply=apply,
            )
        else:
            result = {
                "sourceId": source_id,
                "status": "blocked",
                "blockers": [f"unsupported_mode:{mode}"],
            }
        paper_results.append(_sanitize_for_public(result))

    counts = Counter()
    for result in paper_results:
        counts["paperRows"] += 1
        status = _clean(result.get("status"))
        counts[f"status_{status}"] += 1
        generated = result.get("generatedRecords") if isinstance(result.get("generatedRecords"), dict) else {}
        counts["generatedSourceSpanRecords"] += int(generated.get("sourceSpanCount") or 0)
        counts["generatedStrictEvidenceRecords"] += int(generated.get("strictEvidenceCount") or 0)
        if status == "pass":
            counts["pilotReadbackPassRows"] += 1
        if status == "generated":
            counts["greenfieldGeneratedRows"] += 1

    manifest_ref = Path(str(manifest_path or DEFAULT_MANIFEST_PATH))
    try:
        manifest_input = manifest_ref.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        manifest_input = manifest_ref.name

    payload: dict[str, Any] = {
        "schema": STRUCTURED_EVIDENCE_VERTICAL_SLICE_IMPLEMENTATION_SCHEMA_ID,
        "generatedAt": _now_iso(),
        "runId": RUN_ID,
        "apply": bool(apply),
        "scopeNote": (
            "Structured Evidence vertical slice implementation for five selected "
            "papers: pilot readback, greenfield section_text_offset generation, and "
            "figure caption readback. Runtime answer integration, complex QA eval, "
            "table/equation parsers, manifest expansion, vault access, and external "
            "downloads remain out of scope."
        ),
        "inputs": {
            "corpusManifest": manifest_input,
            "papersDirRef": "papers_dir",
            "targetPaperIds": list(RECOMMENDED_PAPER_IDS),
        },
        "policy": {
            "runtimeAnswerIntegration": False,
            "complexQaEval": False,
            "manifestMutation": False,
            "tableEquationParser": False,
            "citationGradePromotion": False,
            "runtimeEvidencePromotion": False,
            "sourceContentHashAuthority": "corpus_manifest.expectedSourceContentHash",
        },
        "minimalEvidenceTypes": list(_MINIMAL_EVIDENCE_TYPES),
        "deferredEvidenceTypes": list(DEFERRED_EVIDENCE_TYPES),
        "papers": paper_results,
        "counts": dict(counts),
    }
    validation = validate_payload(payload, STRUCTURED_EVIDENCE_VERTICAL_SLICE_IMPLEMENTATION_SCHEMA_ID, strict=False)
    payload["schemaValidation"] = {
        "ok": validation.ok and validation.schema_found and not validation.errors,
        "errors": list(validation.errors),
    }
    return payload


def render_structured_evidence_vertical_slice_implementation_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Structured Evidence Vertical Slice Implementation",
        "",
        f"- generatedAt: {report.get('generatedAt', '')}",
        f"- runId: {report.get('runId', '')}",
        f"- apply: {json.dumps(report.get('apply'))}",
        f"- paperRows: {int(counts.get('paperRows') or 0)}",
        f"- pilotReadbackPassRows: {int(counts.get('pilotReadbackPassRows') or 0)}",
        f"- greenfieldGeneratedRows: {int(counts.get('greenfieldGeneratedRows') or 0)}",
        f"- generatedSourceSpanRecords: {int(counts.get('generatedSourceSpanRecords') or 0)}",
        f"- generatedStrictEvidenceRecords: {int(counts.get('generatedStrictEvidenceRecords') or 0)}",
        "",
        "## Policy",
        "",
        f"- runtime answer integration: {json.dumps((report.get('policy') or {}).get('runtimeAnswerIntegration'))}",
        f"- table/equation parser: {json.dumps((report.get('policy') or {}).get('tableEquationParser'))}",
        f"- sourceContentHash authority: {(report.get('policy') or {}).get('sourceContentHashAuthority')}",
        "",
        "## Deferred Evidence Types",
        "",
    ]
    for item in list(report.get("deferredEvidenceTypes") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## Papers", ""])
    for paper in list(report.get("papers") or []):
        lines.extend(
            [
                f"### {paper.get('sourceId', '')}",
                "",
                f"- mode: {paper.get('mode', '')}",
                f"- status: {paper.get('status', '')}",
                f"- artifactId: {paper.get('artifactId', '')}",
                f"- expectedSourceContentHash: {paper.get('expectedSourceContentHash', '')}",
                f"- parsedArtifactLocator: {paper.get('parsedArtifactLocator', '')}",
            ]
        )
        existing = paper.get("existingRecords") if isinstance(paper.get("existingRecords"), dict) else {}
        generated = paper.get("generatedRecords") if isinstance(paper.get("generatedRecords"), dict) else {}
        trace = paper.get("traceValidation") if isinstance(paper.get("traceValidation"), dict) else {}
        if existing:
            lines.append(
                f"- existing: sourceSpan={existing.get('sourceSpanCount')} strictEvidence={existing.get('strictEvidenceCount')}"
            )
        if generated:
            lines.append(
                f"- generated: sourceSpan={generated.get('sourceSpanCount')} strictEvidence={generated.get('strictEvidenceCount')} applied={generated.get('applied')}"
            )
        if trace:
            lines.append(f"- trace pass: {trace.get('pass', trace.get('allStrictEvidenceTracePass'))}")
        blockers = list(paper.get("blockers") or [])
        if blockers:
            lines.append(f"- blockers: {', '.join(blockers)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

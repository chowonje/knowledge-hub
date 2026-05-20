"""Report-only equation authority hash/identity design.

This helper consumes the Equation Authority readiness audit report and derives
deterministic design-only identity/hash candidates for a future equation
authority tranche. It does not create EquationArtifact, StrictEvidence,
SourceSpan rows, DB/index state, vault content, parser routing, answer
integration, or a new authority policy.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_authority_readiness_audit import (
    EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID,
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH as READINESS_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_MISSING_PDF_REGION as READINESS_STATUS_BLOCKED_MISSING_PDF_REGION,
    STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH as READINESS_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION as READINESS_STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_CANDIDATE_ONLY as READINESS_STATUS_CANDIDATE_ONLY,
)


EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.equation-authority-hash-identity-design.v1"
)

DEFAULT_EQUATION_AUTHORITY_READINESS_AUDIT_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-readiness-audit"
    / "equation-authority-readiness-audit-report.json"
)

DEFAULT_EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-20"
    / "equation-authority-hash-identity-design"
)

TEXT_NORMALIZATION_ID = "equation_text_nfkc_whitespace_v1"
IDENTITY_DIGEST_ID = "equation_authority_identity_design_digest_v1"

STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY = "hash_identity_design_candidate_only"
STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL = "blocked_missing_equation_identity_signal"
STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT = "blocked_missing_tex_or_mathml_text"
STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH = "blocked_ambiguous_equation_match"
STATUS_BLOCKED_RASTER_ONLY_EQUATION = "blocked_raster_only_equation"
STATUS_HELD_OUT_NON_HASH_IDENTITY_BLOCKER = "held_out_non_hash_identity_blocker"
STATUS_BLOCKED_INPUT_REPORT_MISSING = "blocked_input_report_missing"
STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION = "blocked_input_schema_violation"

_DESIGN_COMPATIBLE_READINESS_STATUSES = {
    READINESS_STATUS_CANDIDATE_ONLY,
    READINESS_STATUS_BLOCKED_MISSING_PDF_REGION,
    READINESS_STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_text(value: Any) -> str:
    return " ".join(_safe_text(value).split())


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_safe_text(value)] if _safe_text(value) else []
    if not isinstance(value, (list, tuple)):
        return []
    return [_safe_text(item) for item in value if _safe_text(item)]


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""


def _json_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(encoded)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return slug.strip("-") or "unknown-paper"


def _normalize_equation_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _safe_text(value))
    return " ".join(text.split())


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, "unreadable"
    if not isinstance(payload, dict):
        return {}, "unreadable"
    return payload, ""


def _input_blockers(report_path: Path, report: dict[str, Any], load_error: str) -> list[dict[str, str]]:
    if load_error == "missing":
        return [
            {
                "inputName": "equationAuthorityReadinessAudit",
                "inputLabel": "Equation Authority readiness audit report",
                "path": str(report_path),
                "design_status": STATUS_BLOCKED_INPUT_REPORT_MISSING,
                "detail": "expected input report is missing",
            }
        ]
    if load_error:
        return [
            {
                "inputName": "equationAuthorityReadinessAudit",
                "inputLabel": "Equation Authority readiness audit report",
                "path": str(report_path),
                "design_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
                "detail": "expected input report is unreadable or not a JSON object",
            }
        ]
    schema = _safe_text(report.get("schema"))
    blockers: list[dict[str, str]] = []
    if schema != EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID:
        blockers.append(
            {
                "inputName": "equationAuthorityReadinessAudit",
                "inputLabel": "Equation Authority readiness audit report",
                "path": str(report_path),
                "design_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
                "detail": (
                    "schema mismatch: expected "
                    f"{EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID}, got {schema or 'missing'}"
                ),
            }
        )
        return blockers
    validation = validate_payload(report, EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID, strict=True)
    if not validation.ok:
        blockers.extend(
            {
                "inputName": "equationAuthorityReadinessAudit",
                "inputLabel": "Equation Authority readiness audit report",
                "path": str(report_path),
                "design_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
                "detail": f"schema validation failed: {error}",
            }
            for error in validation.errors
        )
    return blockers


def _identity_signal(row: dict[str, Any]) -> dict[str, Any]:
    identity = dict(row.get("equation_identity") or {})
    latex_labels = _string_list(identity.get("latexLabels"))
    equation_numbers = _string_list(identity.get("equationNumbers"))
    equation_id = _safe_text(identity.get("equationId"))
    source_tex_row_id = _safe_text(identity.get("sourceTexRowId"))
    source_candidate_id = _safe_text(identity.get("sourceCandidateId") or row.get("source_candidate_id"))
    basis = _safe_text(identity.get("identityBasis"))
    available = _safe_bool(identity.get("available")) and bool(
        equation_id or source_tex_row_id or source_candidate_id or latex_labels or equation_numbers
    )
    if not basis:
        if equation_id:
            basis = "equationId"
        elif source_tex_row_id:
            basis = "source_tex_row_id"
        elif source_candidate_id:
            basis = "source_candidate_id"
        elif latex_labels or equation_numbers:
            basis = "label_or_number_hint"
        else:
            basis = "missing"
    return {
        "equationId": equation_id,
        "sourceCandidateId": source_candidate_id,
        "sourceTexRowId": source_tex_row_id,
        "latexLabels": latex_labels,
        "equationNumbers": equation_numbers,
        "identityBasis": basis,
        "available": available,
    }


def _tex_mathml_signal(row: dict[str, Any]) -> dict[str, Any]:
    tex = dict(row.get("tex_mathml_hash_availability") or {})
    candidate_text = _clean_text(tex.get("candidateText"))
    normalized = _normalize_equation_text(candidate_text)
    normalized_hash = _sha256_text(normalized)
    equation_tex_hash = _safe_text(tex.get("equationTeXHash"))
    mathml_hash = _safe_text(tex.get("mathmlHash"))
    mathml_available = _safe_bool(tex.get("mathmlAvailable"))
    hash_source = "none"
    selected_hash = ""
    if normalized_hash:
        hash_source = "normalized_candidate_text_sha256"
        selected_hash = normalized_hash
    elif equation_tex_hash:
        hash_source = "input_equation_tex_hash"
        selected_hash = equation_tex_hash
    elif mathml_hash:
        hash_source = "input_mathml_hash"
        selected_hash = mathml_hash
    return {
        "candidateText": candidate_text,
        "normalizedCandidateText": normalized,
        "normalizedCandidateTextSha256": normalized_hash,
        "equationTeXHash": equation_tex_hash,
        "mathmlHash": mathml_hash,
        "texAvailable": bool(candidate_text),
        "mathmlAvailable": mathml_available,
        "hashAvailable": bool(equation_tex_hash or mathml_hash),
        "designHashSource": hash_source,
        "selectedDesignHash": selected_hash,
        "designHashAvailable": bool(selected_hash),
    }


def _pdf_region_signal(row: dict[str, Any]) -> dict[str, Any]:
    region = dict(row.get("pdf_region") or {})
    page = _safe_int(region.get("page"))
    bbox = _float_list(region.get("bbox"))
    return {
        "available": _safe_bool(region.get("available")) and page is not None and len(bbox) >= 4,
        "page": page,
        "bbox": bbox,
    }


def _classify(row: dict[str, Any], identity: dict[str, Any], tex: dict[str, Any]) -> tuple[str, list[str], str]:
    source_hash = _safe_text(row.get("sourceContentHash"))
    readiness_status = _safe_text(row.get("readiness_status"))
    ambiguity = dict(row.get("ambiguity") or {})
    if not _safe_bool(identity.get("available")):
        return (
            STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL,
            ["equation_identity_signal_missing"],
            "recover_equation_identity_signal_before_hash_identity_design",
        )
    if not source_hash:
        return (
            STATUS_BLOCKED_MISSING_SOURCE_HASH,
            ["sourceContentHash_missing"],
            "recover_source_content_hash_before_hash_identity_design",
        )
    if _safe_bool(row.get("raster_image_only_blocker")) or readiness_status == READINESS_STATUS_BLOCKED_RASTER_ONLY_EQUATION:
        return (
            STATUS_BLOCKED_RASTER_ONLY_EQUATION,
            ["raster_or_image_only_equation_signal"],
            "route_to_image_or_ocr_equation_extractor_before_hash_identity_design",
        )
    if _safe_bool(ambiguity.get("nonUniqueMatch")) or readiness_status == READINESS_STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH:
        return (
            STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
            _string_list(ambiguity.get("reasons")) or ["equation_match_non_unique"],
            "resolve_non_unique_equation_match_before_hash_identity_design",
        )
    if not _safe_bool(tex.get("designHashAvailable")):
        return (
            STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT,
            ["candidate_text_equationTeXHash_or_mathmlHash_missing"],
            "recover_tex_or_mathml_text_before_hash_identity_design",
        )
    if readiness_status not in _DESIGN_COMPATIBLE_READINESS_STATUSES:
        return (
            STATUS_HELD_OUT_NON_HASH_IDENTITY_BLOCKER,
            [f"readiness_status={readiness_status or 'missing'}"],
            "resolve_non_hash_identity_readiness_blocker_before_design_review",
        )
    return (
        STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY,
        ["hash_identity_design_only"],
        "queue_for_later_explicit_equation_authority_contract_design",
    )


def _proposed_designs(
    *,
    paper_id: str,
    source_hash: str,
    identity: dict[str, Any],
    tex: dict[str, Any],
    readiness_audit_row_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity_components = {
        "paperId": paper_id,
        "sourceContentHash": source_hash,
        "equationId": _safe_text(identity.get("equationId")),
        "sourceCandidateId": _safe_text(identity.get("sourceCandidateId")),
        "sourceTexRowId": _safe_text(identity.get("sourceTexRowId")),
        "latexLabels": _string_list(identity.get("latexLabels")),
        "equationNumbers": _string_list(identity.get("equationNumbers")),
        "identityBasis": _safe_text(identity.get("identityBasis")),
        "selectedDesignHash": _safe_text(tex.get("selectedDesignHash")),
        "readinessAuditRowId": readiness_audit_row_id,
    }
    design_digest = _json_digest(identity_components)
    design_id = f"eqauth-design:{_slug(paper_id)}:{design_digest[:20]}"
    proposed_identity = {
        "designOnly": True,
        "identityDigestAlgorithm": IDENTITY_DIGEST_ID,
        "identityComponents": identity_components,
        "identityDigestSha256": design_digest,
        "proposedEquationAuthorityDesignId": design_id,
        "equationIdentityPromoted": False,
        "promotionAllowed": False,
    }
    proposed_hash = {
        "designOnly": True,
        "textNormalization": TEXT_NORMALIZATION_ID,
        "hashSource": _safe_text(tex.get("designHashSource")),
        "normalizedCandidateTextSha256": _safe_text(tex.get("normalizedCandidateTextSha256")),
        "inputEquationTeXHash": _safe_text(tex.get("equationTeXHash")),
        "inputMathMLHash": _safe_text(tex.get("mathmlHash")),
        "selectedDesignHash": _safe_text(tex.get("selectedDesignHash")),
        "equationHashPromoted": False,
        "promotionAllowed": False,
    }
    return proposed_identity, proposed_hash


def _row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    readiness_audit_row_id = _safe_text(row.get("audit_row_id"))
    paper_id = _safe_text(row.get("paper_id"))
    source_hash = _safe_text(row.get("sourceContentHash"))
    identity = _identity_signal(row)
    tex = _tex_mathml_signal(row)
    pdf_region = _pdf_region_signal(row)
    design_status, blockers, recommended_action = _classify(row, identity, tex)
    proposed_identity, proposed_hash = _proposed_designs(
        paper_id=paper_id,
        source_hash=source_hash,
        identity=identity,
        tex=tex,
        readiness_audit_row_id=readiness_audit_row_id,
    )
    readiness_status = _safe_text(row.get("readiness_status"))
    readiness_marker = [f"input_readiness_status={readiness_status}"] if readiness_status != READINESS_STATUS_CANDIDATE_ONLY else []
    downstream_blockers = _dedupe(
        [
            *readiness_marker,
            *_string_list(row.get("blockers")),
            "no_equation_artifact_created",
            "no_strict_evidence_created",
            "no_source_span_mutation",
            "no_parser_routing_change",
            "no_answer_integration_change",
        ]
    )
    return {
        "design_row_id": f"equation-authority-hash-identity-design:{paper_id}:{index:04d}",
        "candidate_type": "equation_authority_hash_identity_design",
        "readiness_audit_row_id": readiness_audit_row_id,
        "paper_id": paper_id,
        "source_candidate_id": _safe_text(row.get("source_candidate_id")),
        "source_file": _safe_text(row.get("source_file")),
        "equation_environment": _safe_text(row.get("equation_environment")),
        "readiness_status": readiness_status,
        "equation_identity": identity,
        "tex_mathml_hash_signal": tex,
        "pdf_region_signal": pdf_region,
        "canonical_text_alignment": dict(row.get("canonical_text_alignment") or {}),
        "sourceContentHash": source_hash,
        "source_hash_available": bool(source_hash),
        "surrounding_context": dict(row.get("surrounding_context") or {}),
        "ambiguity": dict(row.get("ambiguity") or {}),
        "raster_image_only_blocker": _safe_bool(row.get("raster_image_only_blocker")),
        "proposed_identity_design": proposed_identity,
        "proposed_hash_design": proposed_hash,
        "downstream_strict_evidence_blockers": downstream_blockers,
        "design_status": design_status,
        "blockers": _dedupe([*blockers, "equation_authority_hash_identity_design_only"]),
        "recommended_action": recommended_action,
        "equationArtifactCreated": False,
        "strictEvidenceCreated": False,
        "sourceSpanMutated": False,
        "databaseMutation": False,
        "indexMutation": False,
        "reindexOrReembed": False,
        "vaultScan": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "authorityPolicyCreated": False,
    }


def _count_status(rows: list[dict[str, Any]], status: str) -> int:
    return sum(1 for row in rows if _safe_text(row.get("design_status")) == status)


def _counts(rows: list[dict[str, Any]], input_blockers: list[dict[str, str]], input_rows: int) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("design_status")) for row in rows)
    input_statuses = Counter(_safe_text(item.get("design_status")) for item in input_blockers)
    for status, count in input_statuses.items():
        by_status[status] += count
    return {
        "inputRows": input_rows,
        "targetRows": len(rows),
        "hashIdentityDesignCandidateOnlyRows": _count_status(rows, STATUS_HASH_IDENTITY_DESIGN_CANDIDATE_ONLY),
        "blockedMissingEquationIdentitySignalRows": _count_status(
            rows, STATUS_BLOCKED_MISSING_EQUATION_IDENTITY_SIGNAL
        ),
        "blockedMissingTexOrMathmlTextRows": _count_status(rows, STATUS_BLOCKED_MISSING_TEX_OR_MATHML_TEXT),
        "blockedMissingSourceHashRows": _count_status(rows, STATUS_BLOCKED_MISSING_SOURCE_HASH),
        "blockedAmbiguousEquationMatchRows": _count_status(rows, STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH),
        "blockedRasterOnlyEquationRows": _count_status(rows, STATUS_BLOCKED_RASTER_ONLY_EQUATION),
        "heldOutNonHashIdentityBlockerRows": _count_status(rows, STATUS_HELD_OUT_NON_HASH_IDENTITY_BLOCKER),
        "blockedInputReportMissingRows": int(input_statuses.get(STATUS_BLOCKED_INPUT_REPORT_MISSING, 0)),
        "blockedInputSchemaViolationRows": int(input_statuses.get(STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION, 0)),
        "equationArtifactCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "sourceSpanMutatedRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "authorityPolicyCreatedRows": 0,
        "inputBlockerCount": len(input_blockers),
        "byDesignStatus": dict(by_status),
        "byReadinessStatus": dict(Counter(_safe_text(row.get("readiness_status")) for row in rows)),
        "byPaper": dict(Counter(_safe_text(row.get("paper_id")) for row in rows)),
        "byEnvironment": dict(Counter(_safe_text(row.get("equation_environment")) for row in rows)),
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "gate", "policy", "warnings", "inputBlockers")
        if key in report
    }


def render_equation_authority_hash_identity_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Equation Authority Hash/Identity Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Target rows: `{int(counts.get('targetRows') or 0)}`",
        f"- Design candidate-only rows: `{int(counts.get('hashIdentityDesignCandidateOnlyRows') or 0)}`",
        f"- Blocked missing equation identity rows: `{int(counts.get('blockedMissingEquationIdentitySignalRows') or 0)}`",
        f"- Blocked missing TeX/MathML text rows: `{int(counts.get('blockedMissingTexOrMathmlTextRows') or 0)}`",
        f"- Blocked missing source hash rows: `{int(counts.get('blockedMissingSourceHashRows') or 0)}`",
        f"- Blocked ambiguous equation match rows: `{int(counts.get('blockedAmbiguousEquationMatchRows') or 0)}`",
        f"- Blocked raster/image-only rows: `{int(counts.get('blockedRasterOnlyEquationRows') or 0)}`",
        f"- Held out non-hash/identity blockers: `{int(counts.get('heldOutNonHashIdentityBlockerRows') or 0)}`",
        f"- Input blockers: `{int(counts.get('inputBlockerCount') or 0)}`",
        f"- EquationArtifact created rows: `{int(counts.get('equationArtifactCreatedRows') or 0)}`",
        f"- StrictEvidence created rows: `{int(counts.get('strictEvidenceCreatedRows') or 0)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        design = dict(row.get("proposed_identity_design") or {})
        lines.append(
            f"- `{row.get('paper_id')}` `{row.get('design_status')}` "
            f"`{row.get('readiness_status')}` `{design.get('proposedEquationAuthorityDesignId', '')}`"
        )
    if report.get("inputBlockers"):
        lines.extend(["", "## Input blockers", ""])
        for blocker in list(report.get("inputBlockers") or []):
            lines.append(f"- `{blocker.get('design_status')}` `{blocker.get('inputName')}` {blocker.get('detail')}")
    return "\n".join(lines)


def write_equation_authority_hash_identity_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "equation-authority-hash-identity-design-report.json"
    summary_path = root / "equation-authority-hash-identity-design-summary.json"
    markdown_path = root / "equation-authority-hash-identity-design.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_authority_hash_identity_design_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def build_equation_authority_hash_identity_design(
    equation_authority_readiness_audit_report: str | Path = DEFAULT_EQUATION_AUTHORITY_READINESS_AUDIT_REPORT,
    *,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    report_path = Path(str(equation_authority_readiness_audit_report)).expanduser()
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    report_exists = report_path.is_file()
    readiness_report, load_error = _load_json(report_path)
    input_blockers = _input_blockers(report_path, readiness_report, load_error)
    input_rows = len([row for row in list(readiness_report.get("rows") or []) if isinstance(row, dict)])
    rows: list[dict[str, Any]] = []
    if not input_blockers:
        source_rows = [
            dict(row)
            for row in list(readiness_report.get("rows") or [])
            if isinstance(row, dict) and (not allowed or _safe_text(row.get("paper_id")) in allowed)
        ]
        rows = [_row(index, row) for index, row in enumerate(source_rows, start=1)]
    counts = _counts(rows, input_blockers, input_rows)
    status = "ok" if rows and not input_blockers else "blocked"
    decision = (
        "equation_authority_hash_identity_design_ready"
        if status == "ok"
        else STATUS_BLOCKED_INPUT_REPORT_MISSING
        if any(item.get("design_status") == STATUS_BLOCKED_INPUT_REPORT_MISSING for item in input_blockers)
        else STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION
        if input_blockers
        else "blocked_no_equation_authority_readiness_rows"
    )
    warnings = _dedupe(
        [
            "equation authority hash/identity design is report-only and design-only",
            "proposed identity/hash values are not EquationArtifact or StrictEvidence authority",
            "PDF-region, ambiguity, and raster blockers are carried forward from the readiness audit",
            "no SourceSpan, StrictEvidence, DB/index, vault, parser routing, or answer integration mutation occurs",
            *[item.get("detail", "") for item in input_blockers],
        ]
    )
    return {
        "schema": EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "input": {
            "paperIds": requested,
            "equationAuthorityReadinessAuditReportPath": str(report_path),
            "equationAuthorityReadinessAuditReportSchema": _safe_text(readiness_report.get("schema")),
            "equationAuthorityReadinessAuditReportExists": report_exists,
            "equationAuthorityReadinessAuditReportStrictSchemaValid": not input_blockers,
        },
        "counts": counts,
        "gate": {
            "hashIdentityDesignRows": bool(rows),
            "equationAuthorityPromotionReady": False,
            "equationArtifactCreationReady": False,
            "strictEvidenceReady": False,
            "sourceSpanMutationReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "recommendedNextTranche": "equation_authority_contract_design",
            "inputBlockers": input_blockers,
        },
        "policy": {
            "reportOnly": True,
            "designOnly": True,
            "classificationOnly": True,
            "authorityPolicyCreated": False,
            "equationIdentityPromoted": False,
            "equationHashPromoted": False,
            "equationArtifactCreated": False,
            "strictEvidenceCreated": False,
            "sourceSpanMutation": False,
            "databaseMutation": False,
            "indexMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "equationInterpretationAllowed": False,
        },
        "warnings": warnings,
        "inputBlockers": input_blockers,
        "rows": rows,
    }


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equation-authority-readiness-audit-report",
        default=str(DEFAULT_EQUATION_AUTHORITY_READINESS_AUDIT_REPORT),
        help="Path to an existing equation authority readiness audit report.",
    )
    parser.add_argument(
        "--paper-id",
        dest="paper_ids",
        action="append",
        default=[],
        help="Optional paper id filter. May be supplied multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EQUATION_AUTHORITY_HASH_IDENTITY_DESIGN_OUTPUT_DIR),
        help="Directory for report, summary, and Markdown outputs.",
    )
    parser.add_argument("--json", action="store_true", help="Print output paths as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_equation_authority_hash_identity_design(
        equation_authority_readiness_audit_report=args.equation_authority_readiness_audit_report,
        paper_ids=args.paper_ids,
    )
    paths = write_equation_authority_hash_identity_design_reports(report, args.output_dir)
    if args.json:
        print(json.dumps({"status": report["status"], "paths": paths}, ensure_ascii=False, indent=2))
    else:
        print(paths["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

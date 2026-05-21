"""Report-only discovery helper for Structured Evidence vertical slice planning.

Scans the verified corpus manifest, local parsed artifacts, and structured
evidence stores to recommend a minimal 3-5 paper vertical slice. Does not
mutate manifests, create runtime evidence, integrate answers, scan vault
content, or download external sources.
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
    DEFAULT_CORPUS_MANIFEST_PATH,
    corpus_entry_ref,
    load_corpus_manifest,
)
from knowledge_hub.core.schema_validator import validate_payload


STRUCTURED_EVIDENCE_VERTICAL_SLICE_DISCOVERY_SCHEMA_ID = (
    "knowledge-hub.paper.structured-evidence-vertical-slice-discovery.v1"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "knowledgeos" / "fixtures" / "corpus_manifest.json"

EVAL_CRITICAL_SOURCE_IDS = frozenset(
    {
        "2005.11401",
        "2007.01282",
        "2404.16130",
        "2410.05779",
        "alexnet-2012",
        "1706.03762",
        "2312.00752",
        "1810.04805",
        "2005.14165",
        "1512.03385",
        "2310.11511",
        "1502.03167",
        "2201.11903",
    }
)

RECOMMENDED_PAPER_IDS = (
    "1706.03762",
    "2005.11401",
    "1512.03385",
    "1506.02640",
    "2005.14165",
)

MINIMAL_EVIDENCE_TYPES = (
    "section_text_offset",
    "figure_caption_text",
)

STRUCTURED_EVIDENCE_STORES = (
    "source_span",
    "strict_evidence",
    "strict_evidence_eligibility",
    "strict_evidence_citation_grade",
    "strict_evidence_runtime_binding",
)

OPERATOR_LOCAL_RUN_IDS_EXCLUDED = frozenset({"structured-evidence-vertical-slice-20260521"})

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_hash(value: Any) -> str:
    text = _clean(value).lower()
    if text and not text.startswith("sha256:"):
        text = f"sha256:{text}"
    return text


def _hash_body(value: Any) -> str:
    text = _normalize_hash(value)
    return text.removeprefix("sha256:")


def _configured_papers_dir(config: Any) -> Path:
    raw = ""
    if hasattr(config, "get_nested"):
        raw = _clean(config.get_nested("storage", "papers_dir", default=""))
    if not raw:
        raw = _clean(getattr(config, "papers_dir", ""))
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ("." + "khub") / "papers"


def _jsonl_count(path: Path, *, exclude_operator_local_runs: bool = False) -> int:
    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if exclude_operator_local_runs:
            try:
                item = json.loads(line)
            except Exception:
                item = {}
            if isinstance(item, dict) and _clean(item.get("runId")) in OPERATOR_LOCAL_RUN_IDS_EXCLUDED:
                continue
        count += 1
    return count


def _parsed_inspection(*, papers_dir: Path, source_id: str) -> dict[str, Any]:
    parsed_dir = papers_dir / "parsed" / source_id
    manifest_path = parsed_dir / "manifest.json"
    document_path = parsed_dir / "document.json"
    out: dict[str, Any] = {
        "parsedDirRef": f"papers_dir/parsed/{source_id}",
        "manifestExists": manifest_path.is_file(),
        "documentJsonExists": document_path.is_file(),
        "documentMarkdownExists": (parsed_dir / "document.md").is_file(),
        "parsedManifestSourceContentHash": None,
        "parsedManifestSourceContentHashMatchesCorpusManifest": None,
        "parser": None,
        "elementCount": 0,
        "figureArtifactCount": 0,
        "elementTypes": {},
        "hasSectionLikeParagraphs": False,
        "hasTableStructure": False,
        "hasEquationStructure": False,
    }
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        parser_meta = dict(manifest.get("parser_meta") or {})
        parsed_hash = _normalize_hash(
            manifest.get("sourceContentHash")
            or manifest.get("source_content_hash")
            or parser_meta.get("sourceContentHash")
            or parser_meta.get("source_content_hash")
        )
        out["parsedManifestSourceContentHash"] = parsed_hash or None
        out["parser"] = _clean(manifest.get("parser") or parser_meta.get("parser")) or None
    if document_path.is_file():
        try:
            document = json.loads(document_path.read_text(encoding="utf-8"))
        except Exception:
            document = {}
        elements = list(document.get("elements") or [])
        out["elementCount"] = len(elements)
        out["figureArtifactCount"] = len(list(document.get("figure_artifacts") or []))
        type_counts = Counter(_clean(item.get("type")) or "unknown" for item in elements if isinstance(item, dict))
        out["elementTypes"] = dict(type_counts)
        out["hasSectionLikeParagraphs"] = bool(elements)
        out["hasTableStructure"] = bool(type_counts.get("table"))
        out["hasEquationStructure"] = bool(type_counts.get("equation"))
    return out


def _structured_evidence_inspection(*, papers_dir: Path, source_id: str) -> dict[str, Any]:
    root = papers_dir / "structured_evidence"
    stores: dict[str, Any] = {}
    for store_name in STRUCTURED_EVIDENCE_STORES:
        path = root / store_name / f"{source_id}.jsonl"
        stores[store_name] = {
            "storeRef": f"papers_dir/structured_evidence/{store_name}/{source_id}.jsonl",
            "exists": path.is_file(),
            "recordCount": _jsonl_count(path, exclude_operator_local_runs=True),
            "operatorLocalRunIdsExcluded": sorted(OPERATOR_LOCAL_RUN_IDS_EXCLUDED),
        }
    return stores


def _score_paper(row: dict[str, Any]) -> int:
    score = 0
    if row.get("corpusManifestSourceAvailable"):
        score += 5
    parsed = row.get("parsed") or {}
    if parsed.get("manifestExists") and parsed.get("documentJsonExists"):
        score += 4
    if parsed.get("elementCount", 0) >= 10:
        score += 1
    if parsed.get("figureArtifactCount", 0) >= 1:
        score += 2
    if row.get("evalCritical"):
        score += 3
    stores = row.get("structuredEvidenceStores") or {}
    source_span = stores.get("source_span") or {}
    strict = stores.get("strict_evidence") or {}
    if source_span.get("recordCount"):
        score += 2
    if strict.get("recordCount"):
        score += 2
    if not source_span.get("recordCount"):
        score += 1
    return score


def build_structured_evidence_vertical_slice_discovery(
    *,
    config: Any,
    manifest_path: str | Path | None = None,
    papers_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_corpus_manifest(manifest_path or DEFAULT_MANIFEST_PATH)
    resolved_papers_dir = Path(str(papers_dir)).expanduser() if papers_dir else _configured_papers_dir(config)

    paper_rows: list[dict[str, Any]] = []
    for entry in manifest.get("artifacts") or []:
        source_ids = [str(item) for item in list(entry.get("sourceIds") or []) if _clean(item)]
        source_id = source_ids[0] if source_ids else ""
        expected_hash = _normalize_hash(entry.get("expectedSourceContentHash"))
        filename = _clean(entry.get("expectedFilename"))
        local_pdf = resolved_papers_dir / filename if filename else None
        observed_hash = ""
        byte_length = None
        source_available = False
        if local_pdf and local_pdf.is_file() and expected_hash:
            digest = hashlib.sha256()
            size = 0
            with local_pdf.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
            observed_hash = f"sha256:{digest.hexdigest()}"
            byte_length = size
            source_available = observed_hash == expected_hash

        parsed = _parsed_inspection(papers_dir=resolved_papers_dir, source_id=source_id)
        if expected_hash and parsed.get("parsedManifestSourceContentHash"):
            parsed["parsedManifestSourceContentHashMatchesCorpusManifest"] = (
                _hash_body(parsed["parsedManifestSourceContentHash"]) == _hash_body(expected_hash)
            )
        elif expected_hash and parsed.get("manifestExists"):
            parsed["parsedManifestSourceContentHashMatchesCorpusManifest"] = False
        stores = _structured_evidence_inspection(papers_dir=resolved_papers_dir, source_id=source_id)

        row = {
            "sourceId": source_id,
            "artifactId": corpus_entry_ref(entry),
            "titleHint": filename.replace(".pdf", "") if filename else None,
            "evalCritical": source_id in EVAL_CRITICAL_SOURCE_IDS,
            "corpusManifestExpectedSourceContentHash": expected_hash or None,
            "corpusManifestByteLength": entry.get("byteLength"),
            "corpusManifestSourceAvailable": source_available,
            "observedSourceContentHash": observed_hash or None,
            "observedByteLength": byte_length,
            "expectedFilename": filename or None,
            "parsed": parsed,
            "structuredEvidenceStores": stores,
            "traceReadiness": {
                "manifestToSourceArtifact": bool(source_available and expected_hash),
                "sourceArtifactToParsedManifest": bool(parsed.get("manifestExists")),
                "parsedManifestToDocumentJson": bool(parsed.get("documentJsonExists")),
                "parsedToSourceSpanStore": bool((stores.get("source_span") or {}).get("recordCount")),
                "sourceSpanToStrictEvidenceStore": bool((stores.get("strict_evidence") or {}).get("recordCount")),
                "strictEvidenceToRuntimeEvidencePacket": False,
            },
            "verticalSliceCandidateScore": 0,
            "recommendedForFirstSlice": source_id in RECOMMENDED_PAPER_IDS,
        }
        row["verticalSliceCandidateScore"] = _score_paper(row)
        paper_rows.append(row)

    recommended = [row for row in paper_rows if row["recommendedForFirstSlice"]]
    recommended.sort(key=lambda row: (-row["verticalSliceCandidateScore"], row["sourceId"]))

    counts = Counter()
    for row in paper_rows:
        parsed = row.get("parsed") or {}
        if parsed.get("manifestExists"):
            counts["parsedManifestRows"] += 1
        if parsed.get("documentJsonExists"):
            counts["parsedDocumentJsonRows"] += 1
        if (row.get("structuredEvidenceStores") or {}).get("source_span", {}).get("recordCount"):
            counts["sourceSpanStoreRows"] += 1
        if (row.get("structuredEvidenceStores") or {}).get("strict_evidence", {}).get("recordCount"):
            counts["strictEvidenceStoreRows"] += 1
        if row.get("corpusManifestSourceAvailable"):
            counts["corpusSourceAvailableRows"] += 1

    manifest_ref = Path(str(manifest_path or DEFAULT_CORPUS_MANIFEST_PATH))
    try:
        manifest_input = manifest_ref.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        manifest_input = manifest_ref.name

    payload: dict[str, Any] = {
        "schema": STRUCTURED_EVIDENCE_VERTICAL_SLICE_DISCOVERY_SCHEMA_ID,
        "generatedAt": _now_iso(),
        "scopeNote": (
            "Discovery-only report for Structured Evidence vertical slice planning on the "
            "verified corpus manifest. Does not mutate manifests, create runtime "
            "evidence, integrate answers, scan vault content, or download external sources."
        ),
        "inputs": {
            "corpusManifest": manifest_input,
            "papersDirRef": "papers_dir",
        },
        "currentStructure": {
            "parsedArtifactLayout": {
                "rootTemplate": "papers_dir/parsed/{source_id}/",
                "files": ["manifest.json", "document.json", "document.md", "figures/ (optional)"],
                "documentJsonKeys": ["markdown_text", "elements", "parser_meta", "figure_artifacts"],
                "primaryElementType": "paragraph",
                "nativeTableOrEquationSupportInCurrentParser": False,
            },
            "structuredEvidenceStores": [
                {
                    "store": name,
                    "pathTemplate": f"papers_dir/structured_evidence/{name}/{{source_id}}.jsonl",
                }
                for name in STRUCTURED_EVIDENCE_STORES
            ],
            "evidencePacketSchema": "knowledge-hub.evidence-packet.v1",
            "sourceSpanRecordSchema": "knowledge-hub.paper.parsed-artifact-source-span-record.v1",
            "strictEvidenceRecordSchema": "knowledge-hub.paper.parsed-artifact-strict-evidence-record.v1",
        },
        "counts": {
            "corpusManifestRows": len(paper_rows),
            **dict(counts),
            "recommendedFirstSliceRows": len(recommended),
        },
        "minimalEvidenceTypes": list(MINIMAL_EVIDENCE_TYPES),
        "deferredEvidenceTypes": [
            "table_cell_numeric",
            "equation_citation",
            "appendix_table_lookup",
        ],
        "recommendedFirstSlice": [
            {
                "sourceId": row["sourceId"],
                "artifactId": row["artifactId"],
                "titleHint": row.get("titleHint"),
                "evalCritical": row.get("evalCritical"),
                "verticalSliceCandidateScore": row.get("verticalSliceCandidateScore"),
                "reason": _recommendation_reason(row),
                "traceReadiness": row.get("traceReadiness"),
            }
            for row in recommended
        ],
        "papers": paper_rows,
        "requiredContract": {
            "traceChain": [
                "corpus_manifest.sourceIds[]",
                "corpus_manifest.artifactId",
                "corpus_manifest.expectedSourceContentHash",
                "local source PDF bytes",
                "parsed/{source_id}/manifest.json",
                "parsed/{source_id}/document.json",
                "structured_evidence/source_span/{source_id}.jsonl",
                "structured_evidence/strict_evidence/{source_id}.jsonl",
            ],
            "hashRule": "corpus manifest expectedSourceContentHash must match observed local source bytes; strict/source_span records must preserve sourceContentHash body",
            "publicOutputRule": "no absolute local paths in public reports or manifests",
            "outOfScopeNow": [
                "runtime answer integration",
                "complex QA full eval execution",
                "citation_grade promotion",
                "runtime_binding promotion",
                "vault access",
                "corpus manifest expansion",
            ],
        },
        "nextImplementationTranche": {
            "durationTarget": "1-2 days",
            "deliverables": [
                "report-only trace readback validator for 5 selected papers",
                "section_text_offset promotion/readback on one greenfield paper (2005.11401)",
                "figure_caption_text readback on one pilot paper (1706.03762 or 1506.02640)",
                "schema-valid discovery/readback JSON report only",
            ],
            "explicitlyNotInTranche": [
                "table/equation structured evidence",
                "runtime evidence packet emission",
                "answer path wiring",
                "complex QA seed execution",
            ],
        },
        "policy": {
            "reportOnly": True,
            "manifestMutation": False,
            "runtimeAnswerIntegration": False,
            "vaultScan": False,
            "externalDownload": False,
        },
    }

    validation = validate_payload(payload, STRUCTURED_EVIDENCE_VERTICAL_SLICE_DISCOVERY_SCHEMA_ID, strict=False)
    payload["schemaValidation"] = {
        "ok": validation.ok and validation.schema_found and not validation.errors,
        "errors": list(validation.errors),
    }
    return payload


def _recommendation_reason(row: dict[str, Any]) -> str:
    source_id = row.get("sourceId")
    parsed = row.get("parsed") or {}
    stores = row.get("structuredEvidenceStores") or {}
    parts: list[str] = []
    if row.get("evalCritical"):
        parts.append("eval-critical manifest paper")
    if stores.get("source_span", {}).get("recordCount"):
        parts.append("existing source_span pilot records for readback")
    elif source_id == "2005.11401":
        parts.append("greenfield RAG paper with verified source and parsed artifact")
    if parsed.get("figureArtifactCount", 0):
        parts.append("figure_artifacts present for caption slice")
    if parsed.get("elementCount", 0) >= 10:
        parts.append("section-like paragraph coverage")
    return "; ".join(parts) or "verified corpus row with parsed artifact"


def render_structured_evidence_vertical_slice_discovery_markdown(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    lines = [
        "# Structured Evidence Vertical Slice Discovery",
        "",
        f"Generated: `{payload.get('generatedAt')}`",
        "",
        payload.get("scopeNote", ""),
        "",
        "## Corpus / Parsed Coverage",
        "",
        f"- Corpus manifest rows: **{counts.get('corpusManifestRows', 0)}**",
        f"- Corpus source available rows: **{counts.get('corpusSourceAvailableRows', 0)}**",
        f"- Parsed manifest rows: **{counts.get('parsedManifestRows', 0)}**",
        f"- Parsed document.json rows: **{counts.get('parsedDocumentJsonRows', 0)}**",
        f"- SourceSpan store rows: **{counts.get('sourceSpanStoreRows', 0)}**",
        f"- StrictEvidence store rows: **{counts.get('strictEvidenceStoreRows', 0)}**",
        "",
        "## Recommended First Slice (5 papers)",
        "",
    ]
    for row in payload.get("recommendedFirstSlice") or []:
        lines.append(
            f"- `{row.get('sourceId')}` ({row.get('artifactId')}) — {row.get('reason')}"
        )
    lines.extend(
        [
            "",
            "## Minimal Evidence Types",
            "",
        ]
    )
    for item in payload.get("minimalEvidenceTypes") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Deferred Evidence Types", ""])
    for item in payload.get("deferredEvidenceTypes") or []:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "STRUCTURED_EVIDENCE_VERTICAL_SLICE_DISCOVERY_SCHEMA_ID",
    "RECOMMENDED_PAPER_IDS",
    "build_structured_evidence_vertical_slice_discovery",
    "render_structured_evidence_vertical_slice_discovery_markdown",
]

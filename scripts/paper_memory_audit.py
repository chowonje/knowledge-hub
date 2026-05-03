#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from knowledge_hub.application.context import AppContextFactory
from knowledge_hub.papers.text_sanitizer import normalize_paper_texts

_LATEX_MARKERS = (
    "\\documentclass",
    "\\usepackage",
    "\\begin{document}",
    "\\maketitle",
    "\\thanks{",
    "\\title{",
    "\\author{",
    "\\hypersetup",
    "\\includepdf",
    "\\section{",
    "\\subsection{",
    "\\paragraph{",
    "\\begin{abstract}",
)
_GENERIC_LIMITATION_PATTERNS = (
    "limited information provided",
    "limitations are not explicitly stated",
    "the provided excerpts do not describe limitations",
    "insufficient information to determine limitations",
    "the available context does not mention limitations",
    "the paper does not explicitly discuss limitations",
    "no explicit limitation",
)
_TITLE_STOPWORDS = {
    "agent",
    "agents",
    "ai",
    "analysis",
    "approach",
    "benchmark",
    "bench",
    "evaluation",
    "evaluating",
    "framework",
    "large",
    "language",
    "llm",
    "llms",
    "model",
    "models",
    "paper",
    "reasoning",
    "retrieval",
    "study",
    "survey",
    "system",
    "systems",
    "using",
    "with",
    "towards",
    "via",
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _read_text(path_value: Any) -> str:
    token = str(path_value or "").strip()
    if not token:
        return ""
    path = Path(token)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _text_head(text: Any, *, limit: int = 1600) -> str:
    return str(text or "").lstrip()[:limit]


def _contains_latex_markup(text: Any) -> bool:
    body = str(text or "")
    lowered = body.casefold()
    return any(marker in lowered for marker in _LATEX_MARKERS)


def _text_starts_latex(text: Any) -> bool:
    return _contains_latex_markup(_text_head(text))


def _is_generic_limitation(text: Any) -> bool:
    lowered = _clean_text(text).casefold()
    if not lowered:
        return False
    return any(pattern in lowered for pattern in _GENERIC_LIMITATION_PATTERNS)


def _tokens(text: Any) -> set[str]:
    return {
        token
        for token in re.split(r"[^0-9A-Za-z가-힣]+", str(text or "").casefold())
        if len(token) >= 4
    }


def _significant_title_tokens(title: Any) -> set[str]:
    return {token for token in _tokens(title) if token not in _TITLE_STOPWORDS}


def _likely_semantic_mismatch(title: Any, *fields: Any) -> bool:
    title_tokens = _significant_title_tokens(title)
    if len(title_tokens) < 2:
        return False
    content = _clean_text(" ".join(str(field or "") for field in fields))
    if len(content) < 80 or _contains_latex_markup(content):
        return False
    content_tokens = _tokens(content)
    if not content_tokens:
        return False
    overlap = title_tokens & content_tokens
    return len(overlap) == 0


def _review_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if bool(row.get("latexCore")):
        reasons.append("latex_core")
    if bool(row.get("latexProblemContext")):
        reasons.append("latex_problem_context")
    if bool(row.get("textStartsLatex")):
        reasons.append("text_starts_latex")
    if bool(row.get("genericLimitation")):
        reasons.append("generic_limitation")
    if bool(row.get("emptyProblemContext")):
        reasons.append("empty_problem_context")
    if bool(row.get("emptyMethod")):
        reasons.append("empty_method")
    if bool(row.get("emptyEvidence")):
        reasons.append("empty_evidence")
    if bool(row.get("fallbackUsed")):
        reasons.append("fallback_used")
    if bool(row.get("likelySemanticMismatch")):
        reasons.append("likely_semantic_mismatch")
    return reasons


def _problem_score(row: dict[str, Any]) -> int:
    weights = {
        "latexCore": 5,
        "latexProblemContext": 4,
        "textStartsLatex": 3,
        "genericLimitation": 2,
        "emptyProblemContext": 2,
        "emptyMethod": 2,
        "emptyEvidence": 2,
        "fallbackUsed": 4,
        "likelySemanticMismatch": 5,
    }
    return sum(weight for key, weight in weights.items() if bool(row.get(key)))


def build_paper_memory_audit_rows(
    records: list[dict[str, Any]],
    *,
    diagnostics_by_paper_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    diagnostics_by_paper_id = diagnostics_by_paper_id or {}
    rows: list[dict[str, Any]] = []
    for record in records:
        paper_id = str(record.get("paper_id") or record.get("paperId") or "").strip()
        if not paper_id:
            continue
        diagnostics = dict(diagnostics_by_paper_id.get(paper_id) or {})
        translated_text = _read_text(record.get("translated_path"))
        raw_text = _read_text(record.get("text_path"))
        normalization = normalize_paper_texts(translated_text=translated_text, raw_text=raw_text)
        source_text = translated_text or raw_text
        source_text_kind = "translated" if translated_text else ("raw" if raw_text else "missing")
        row = {
            "paperId": paper_id,
            "title": str(record.get("title") or ""),
            "paperCore": str(record.get("paper_core") or ""),
            "problemContext": str(record.get("problem_context") or ""),
            "methodCore": str(record.get("method_core") or ""),
            "evidenceCore": str(record.get("evidence_core") or ""),
            "limitations": str(record.get("limitations") or ""),
            "qualityFlag": str(record.get("quality_flag") or ""),
            "sourceTextKind": source_text_kind,
            "oldPdfPath": str(record.get("pdf_path") or ""),
            "oldTextPath": str(record.get("text_path") or ""),
            "sourceTextPath": str(record.get("translated_path") or record.get("text_path") or ""),
            "latexCore": _contains_latex_markup(record.get("paper_core")),
            "latexProblemContext": _contains_latex_markup(record.get("problem_context")),
            "textStartsLatex": _text_starts_latex(source_text),
            "genericLimitation": _is_generic_limitation(record.get("limitations")),
            "emptyProblemContext": not _clean_text(record.get("problem_context")),
            "emptyMethod": not _clean_text(record.get("method_core")),
            "emptyEvidence": not _clean_text(record.get("evidence_core")),
            "fallbackUsed": bool(diagnostics.get("fallbackUsed")),
            "likelySemanticMismatch": _likely_semantic_mismatch(
                record.get("title"),
                record.get("paper_core"),
                record.get("problem_context"),
                record.get("method_core"),
                record.get("evidence_core"),
            ),
            "extractorModel": str(diagnostics.get("extractorModel") or ""),
            "warningCount": len(list(diagnostics.get("warnings") or [])),
            "preferredSource": str(normalization.preferred_source),
            "selectedStartAnchor": str((normalization.translated if normalization.preferred_source == "translated" else normalization.raw).selected_start_anchor),
            "weakSanitizedContent": bool(normalization.weak_content),
            "removedReferencesTail": bool(normalization.translated.removed_references_tail or normalization.raw.removed_references_tail),
            "droppedLatexLineCount": int(normalization.translated.dropped_latex_line_count + normalization.raw.dropped_latex_line_count),
        }
        row["emptyMethodAndEvidence"] = bool(row["emptyMethod"] and row["emptyEvidence"])
        row["recommendedParser"] = (
            "mineru"
            if bool(row["textStartsLatex"]) and bool(row["weakSanitizedContent"])
            else ("opendataloader" if bool(row["likelySemanticMismatch"]) else "raw")
        )
        row["parserEscalationCandidate"] = bool(
            row["textStartsLatex"] or row["likelySemanticMismatch"] or row["fallbackUsed"] or row["weakSanitizedContent"]
        )
        row["cleanupReason"] = ",".join(
            token
            for token, enabled in (
                ("semantic_mismatch", bool(row["likelySemanticMismatch"])),
                ("fallback", bool(row["fallbackUsed"])),
                ("text_starts_latex", bool(row["textStartsLatex"])),
            )
            if enabled
        )
        row["reviewReasons"] = _review_reasons(row)
        row["needsReview"] = bool(row["reviewReasons"])
        row["problemScore"] = _problem_score(row)
        rows.append(row)
    rows.sort(key=lambda item: (-int(item.get("problemScore") or 0), str(item.get("paperId") or "")))
    return rows


def summarize_paper_memory_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in list(row.get("reviewReasons") or []):
            token = str(reason or "").strip()
            if token:
                reason_counts[token] = reason_counts.get(token, 0) + 1
    needs_review = sum(1 for row in rows if bool(row.get("needsReview")))
    summary = {
        "schema": "knowledge-hub.paper-memory-audit.v1",
        "total": total,
        "needsReviewCount": needs_review,
        "weakCardRate": round((needs_review / total), 4) if total else 0.0,
        "latexCoreCount": sum(1 for row in rows if bool(row.get("latexCore"))),
        "latexProblemContextCount": sum(1 for row in rows if bool(row.get("latexProblemContext"))),
        "textStartsLatexCount": sum(1 for row in rows if bool(row.get("textStartsLatex"))),
        "genericLimitationCount": sum(1 for row in rows if bool(row.get("genericLimitation"))),
        "emptyProblemContextCount": sum(1 for row in rows if bool(row.get("emptyProblemContext"))),
        "emptyMethodCount": sum(1 for row in rows if bool(row.get("emptyMethod"))),
        "emptyEvidenceCount": sum(1 for row in rows if bool(row.get("emptyEvidence"))),
        "emptyMethodAndEvidenceCount": sum(1 for row in rows if bool(row.get("emptyMethodAndEvidence"))),
        "fallbackCount": sum(1 for row in rows if bool(row.get("fallbackUsed"))),
        "semanticMismatchCount": sum(1 for row in rows if bool(row.get("likelySemanticMismatch"))),
        "weakSanitizedContentCount": sum(1 for row in rows if bool(row.get("weakSanitizedContent"))),
        "removedReferencesTailCount": sum(1 for row in rows if bool(row.get("removedReferencesTail"))),
        "parserEscalationCount": sum(1 for row in rows if bool(row.get("parserEscalationCandidate"))),
        "reviewReasonCounts": reason_counts,
        "topProblemPaperIds": [str(row.get("paperId") or "") for row in rows[:20]],
        "passBPaperIds": [
            str(row.get("paperId") or "")
            for row in rows
            if bool(row.get("textStartsLatex"))
            or bool(row.get("genericLimitation"))
            or bool(row.get("fallbackUsed"))
            or bool(row.get("likelySemanticMismatch"))
        ],
    }
    return summary


def compare_paper_memory_audit_summaries(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    metric_keys = (
        "total",
        "needsReviewCount",
        "latexCoreCount",
        "textStartsLatexCount",
        "genericLimitationCount",
        "emptyMethodCount",
        "emptyEvidenceCount",
        "emptyMethodAndEvidenceCount",
        "fallbackCount",
        "semanticMismatchCount",
    )
    delta = {
        key: int(current.get(key) or 0) - int(previous.get(key) or 0)
        for key in metric_keys
    }
    return {
        "schema": "knowledge-hub.paper-memory-audit-comparison.v1",
        "current": {key: current.get(key) for key in metric_keys},
        "previous": {key: previous.get(key) for key in metric_keys},
        "delta": delta,
        "rates": {
            "currentWeakCardRate": float(current.get("weakCardRate") or 0.0),
            "previousWeakCardRate": float(previous.get("weakCardRate") or 0.0),
            "weakCardRateDelta": round(
                float(current.get("weakCardRate") or 0.0) - float(previous.get("weakCardRate") or 0.0), 4
            ),
        },
    }


def _load_previous_summary(path_value: str | None) -> dict[str, Any]:
    token = str(path_value or "").strip()
    if not token:
        return {}
    try:
        return dict(json.loads(Path(token).expanduser().read_text(encoding="utf-8")))
    except Exception:
        return {}


def write_paper_memory_audit_artifacts(
    *,
    artifact_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    comparison: dict[str, Any] | None = None,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows_path = artifact_dir / "paper_memory_quality_rows.jsonl"
    summary_path = artifact_dir / "paper_memory_quality_summary.json"
    csv_path = artifact_dir / "paper_memory_problem_cards.csv"
    top20_path = artifact_dir / "paper_memory_top20_problem_cards.json"
    pass_b_ids_path = artifact_dir / "paper_memory_pass_b_ids.txt"
    cleanup_queue_path = artifact_dir / "paper_memory_source_cleanup_queue.csv"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paperId",
                "title",
                "problemScore",
                "needsReview",
                "reviewReasons",
                "latexCore",
                "textStartsLatex",
                "genericLimitation",
                "emptyMethod",
                "emptyEvidence",
                "fallbackUsed",
                "likelySemanticMismatch",
                "preferredSource",
                "selectedStartAnchor",
                "weakSanitizedContent",
                "removedReferencesTail",
                "recommendedParser",
                "parserEscalationCandidate",
                "sourceTextKind",
                "sourceTextPath",
            ],
        )
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["reviewReasons"] = ",".join(list(row.get("reviewReasons") or []))
            writer.writerow({key: payload.get(key, "") for key in writer.fieldnames or []})
    with cleanup_queue_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paperId",
                "title",
                "oldPdfPath",
                "oldTextPath",
                "newPdfPath",
                "newTextPath",
                "reason",
                "recommendedParser",
            ],
        )
        writer.writeheader()
        for row in rows:
            if not (bool(row.get("likelySemanticMismatch")) or bool(row.get("fallbackUsed"))):
                continue
            writer.writerow(
                {
                    "paperId": str(row.get("paperId") or ""),
                    "title": str(row.get("title") or ""),
                    "oldPdfPath": str(row.get("oldPdfPath") or ""),
                    "oldTextPath": str(row.get("oldTextPath") or ""),
                    "newPdfPath": "",
                    "newTextPath": "",
                    "reason": str(row.get("cleanupReason") or ",".join(list(row.get("reviewReasons") or []))),
                    "recommendedParser": str(row.get("recommendedParser") or "raw"),
                }
            )
    top20_path.write_text(json.dumps(rows[:20], ensure_ascii=False, indent=2), encoding="utf-8")
    pass_b_ids_path.write_text("\n".join(str(item) for item in list(summary.get("passBPaperIds") or []) if str(item).strip()) + "\n", encoding="utf-8")
    result = {
        "rows": str(rows_path),
        "summary": str(summary_path),
        "problemCardsCsv": str(csv_path),
        "top20": str(top20_path),
        "passBIds": str(pass_b_ids_path),
        "sourceCleanupQueue": str(cleanup_queue_path),
    }
    if comparison:
        comparison_path = artifact_dir / "paper_memory_quality_comparison.json"
        comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        result["comparison"] = str(comparison_path)
    return result


def load_paper_memory_records(
    sqlite_db,
    *,
    paper_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    query = (
        "SELECT c.*, p.pdf_path AS pdf_path, p.text_path AS text_path, p.translated_path AS translated_path "
        "FROM paper_memory_cards c LEFT JOIN papers p ON p.arxiv_id = c.paper_id"
    )
    if paper_ids:
        placeholders = ", ".join(["?"] * len(paper_ids))
        query += f" WHERE c.paper_id IN ({placeholders})"
        params.extend(sorted(paper_ids))
    query += " ORDER BY c.updated_at DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(max(1, int(limit)))
    return [dict(row) for row in sqlite_db.conn.execute(query, tuple(params)).fetchall()]


def load_diagnostics_rows(path_value: str | None) -> dict[str, dict[str, Any]]:
    token = str(path_value or "").strip()
    if not token:
        return {}
    path = Path(token).expanduser()
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        paper_id = str(payload.get("paperId") or "").strip()
        if paper_id:
            rows[paper_id] = payload
    return rows


def _load_selected_paper_ids(path_value: str | None) -> set[str]:
    token = str(path_value or "").strip()
    if not token:
        return set()
    path = Path(token).expanduser()
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return {row for row in rows if row and not row.startswith("#")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit paper-memory cards for weak-card heuristics.")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument("--paper-id-file", default=None, help="Optional newline-delimited paper id file")
    parser.add_argument("--limit", type=int, default=None, help="Optional card limit")
    parser.add_argument("--diagnostics-rows", default=None, help="Optional paper_memory_extraction_rows.jsonl path")
    parser.add_argument("--artifact-dir", default=None, help="Directory to write audit artifacts")
    parser.add_argument("--compare-to-summary", default=None, help="Optional previous audit summary JSON path")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON summary")
    args = parser.parse_args()

    factory = AppContextFactory(config_path=args.config)
    sqlite_db = factory.get_sqlite_db()
    records = load_paper_memory_records(
        sqlite_db,
        paper_ids=_load_selected_paper_ids(args.paper_id_file),
        limit=args.limit,
    )
    diagnostics = load_diagnostics_rows(args.diagnostics_rows)
    rows = build_paper_memory_audit_rows(records, diagnostics_by_paper_id=diagnostics)
    summary = summarize_paper_memory_audit(rows)
    previous = _load_previous_summary(args.compare_to_summary)
    comparison = compare_paper_memory_audit_summaries(summary, previous) if previous else None

    payload = {
        "status": "ok",
        "paperMemoryAudit": summary,
    }
    if comparison:
        payload["comparison"] = comparison
    if args.artifact_dir:
        payload["artifactPaths"] = write_paper_memory_audit_artifacts(
            artifact_dir=Path(str(args.artifact_dir)).expanduser(),
            rows=rows,
            summary=summary,
            comparison=comparison,
        )
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("paper-memory audit summary")
    print(f"- total={summary['total']} needs_review={summary['needsReviewCount']} weak_card_rate={summary['weakCardRate']}")
    print(
        "- counts "
        f"latex_core={summary['latexCoreCount']} "
        f"text_starts_latex={summary['textStartsLatexCount']} "
        f"generic_limitation={summary['genericLimitationCount']} "
        f"empty_method={summary['emptyMethodCount']} "
        f"empty_evidence={summary['emptyEvidenceCount']} "
        f"fallback={summary['fallbackCount']} "
        f"semantic_mismatch={summary['semanticMismatchCount']}"
    )
    if comparison:
        print(f"- comparison {comparison['delta']}")
    if args.artifact_dir:
        print(f"- artifacts {payload['artifactPaths']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

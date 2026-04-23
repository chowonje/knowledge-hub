from __future__ import annotations

from typing import Any

from knowledge_hub.application.ops_alerts import evaluate_rag_report_alerts


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if parsed < 0:
            return 0.0
        if parsed > 1:
            return 1.0
        return parsed
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _truncate_text(text: str, limit: int = 280) -> str:
    body = " ".join(str(text or "").strip().split())
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 3)].rstrip() + "..."


def build_rag_ops_report(sqlite_db, *, limit: int = 100, days: int = 7) -> dict[str, Any]:
    lister = getattr(sqlite_db, "list_rag_answer_logs", None)
    normalized_limit = max(1, int(limit))
    normalized_days = max(0, int(days))
    if not callable(lister):
        return {
            "schema": "knowledge-hub.rag.report.result.v1",
            "status": "failed",
            "days": normalized_days,
            "limit": normalized_limit,
            "counts": {
                "total": 0,
                "needsCaution": 0,
                "rewriteAttempted": 0,
                "rewriteApplied": 0,
                "conservativeFallback": 0,
                "unsupportedClaimLogs": 0,
            },
            "verification": {
                "verified": 0,
                "caution": 0,
                "failed": 0,
                "skipped": 0,
                "unknown": 0,
            },
            "rates": {
                "needsCautionRate": 0.0,
                "rewriteAttemptedRate": 0.0,
                "rewriteAppliedRate": 0.0,
                "conservativeFallbackRate": 0.0,
                "unsupportedClaimRate": 0.0,
            },
            "topWarningPatterns": [],
            "samples": [],
            "alerts": [],
            "recommendedActions": [],
            "warnings": ["rag answer log store unavailable"],
        }

    rows = list(lister(limit=normalized_limit, days=normalized_days))
    verification_counts = {"verified": 0, "caution": 0, "failed": 0, "skipped": 0, "unknown": 0}
    warning_patterns: dict[str, int] = {}
    counts = {
        "total": len(rows),
        "needsCaution": 0,
        "rewriteAttempted": 0,
        "rewriteApplied": 0,
        "conservativeFallback": 0,
        "unsupportedClaimLogs": 0,
    }
    samples: list[dict[str, Any]] = []
    for row in rows:
        verification_status = str(row.get("verification_status") or "").strip().lower()
        if verification_status not in verification_counts:
            verification_status = "unknown"
        verification_counts[verification_status] += 1
        if bool(row.get("needs_caution")):
            counts["needsCaution"] += 1
        if bool(row.get("rewrite_attempted")):
            counts["rewriteAttempted"] += 1
        if bool(row.get("rewrite_applied")):
            counts["rewriteApplied"] += 1
        if str(row.get("final_answer_source") or "") == "conservative_fallback":
            counts["conservativeFallback"] += 1
        if int(row.get("unsupported_claim_count") or 0) > 0:
            counts["unsupportedClaimLogs"] += 1
        for warning in list(row.get("warnings_json") or [])[:3]:
            token = _truncate_text(str(warning or ""), 120)
            if token:
                warning_patterns[token] = warning_patterns.get(token, 0) + 1
        if len(samples) < 10:
            samples.append(
                {
                    "id": int(row.get("id") or 0),
                    "createdAt": str(row.get("created_at") or ""),
                    "queryHash": str(row.get("query_hash") or ""),
                    "queryDigest": str(row.get("query_digest") or ""),
                    "resultStatus": str(row.get("result_status") or ""),
                    "verificationStatus": str(row.get("verification_status") or ""),
                    "needsCaution": bool(row.get("needs_caution")),
                    "unsupportedClaimCount": int(row.get("unsupported_claim_count") or 0),
                    "rewriteApplied": bool(row.get("rewrite_applied")),
                    "finalAnswerSource": str(row.get("final_answer_source") or ""),
                    "warningCount": int(row.get("warning_count") or 0),
                }
            )
    total = max(1, counts["total"])
    rates = {
        "needsCautionRate": round(counts["needsCaution"] / total, 4),
        "rewriteAttemptedRate": round(counts["rewriteAttempted"] / total, 4),
        "rewriteAppliedRate": round(counts["rewriteApplied"] / total, 4),
        "conservativeFallbackRate": round(counts["conservativeFallback"] / total, 4),
        "unsupportedClaimRate": round(counts["unsupportedClaimLogs"] / total, 4),
    }
    top_warning_patterns = [
        {"warning": warning, "count": count}
        for warning, count in sorted(warning_patterns.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    alerts, recommended_actions = evaluate_rag_report_alerts(
        days=normalized_days,
        limit=normalized_limit,
        counts=counts,
        verification=verification_counts,
        rates=rates,
    )
    return {
        "schema": "knowledge-hub.rag.report.result.v1",
        "status": "ok",
        "days": normalized_days,
        "limit": normalized_limit,
        "counts": counts,
        "verification": verification_counts,
        "rates": rates,
        "topWarningPatterns": top_warning_patterns,
        "samples": samples,
        "alerts": alerts,
        "recommendedActions": recommended_actions,
        "warnings": [],
    }

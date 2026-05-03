#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

from knowledge_hub.application.context import AppContextFactory

USER_ANSWER_EVAL_FIELDNAMES = [
    "query",
    "source",
    "query_type",
    "expected_primary_source",
    "expected_answer_style",
    "difficulty",
    "review_bucket",
    "answer_status",
    "answer_text",
    "answer_preview",
    "no_result",
    "needs_caution",
    "verification_status",
    "verification_summary",
    "unsupported_claim_count",
    "source_count",
    "source_titles",
    "source_refs",
    "runtime_used",
    "answer_route",
    "router_provider",
    "router_model",
    "latency_ms",
    "top_k",
    "retrieval_mode",
    "answer_backend",
    "answer_backend_model",
    "packet_ref",
    "pred_label",
    "pred_groundedness",
    "pred_usefulness",
    "pred_readability",
    "pred_source_accuracy",
    "pred_should_abstain",
    "pred_confidence",
    "pred_reason",
    "judge_provider",
    "judge_model",
    "final_label",
    "final_groundedness",
    "final_usefulness",
    "final_readability",
    "final_source_accuracy",
    "final_should_abstain",
    "final_notes",
]


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _preview_text(value: Any, *, limit: int = 240) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}…"


def _read_queries(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    items: list[dict[str, str]] = []
    for row in rows:
        query = _clean_text(row.get("query"))
        if not query:
            continue
        items.append({str(key): _clean_text(value) for key, value in row.items()})
    return items


def _auto_load_dotenv() -> None:
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"]:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
        break


def _serialize_sources(result: dict[str, Any]) -> tuple[str, str, int]:
    titles: list[str] = []
    refs: list[str] = []
    for item in list(result.get("sources") or []):
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title") or item.get("name"))
        ref = _clean_text(
            item.get("source_ref")
            or item.get("sourceRef")
            or item.get("url")
            or item.get("file_path")
            or item.get("filePath")
            or item.get("document_id")
            or item.get("documentId")
        )
        if title:
            titles.append(title)
        if ref:
            refs.append(ref)
    return " | ".join(titles[:8]), " | ".join(refs[:8]), len(list(result.get("sources") or []))


def _serialize_row(
    query_row: dict[str, str],
    result: dict[str, Any],
    *,
    top_k: int,
    retrieval_mode: str,
    latency_ms: float,
) -> dict[str, str]:
    answer_verification = dict(result.get("answerVerification") or {})
    family_diag = dict(result.get("familyRouteDiagnostics") or {})
    router = dict(result.get("router") or {})
    selected = dict(router.get("selected") or {})
    source_titles, source_refs, source_count = _serialize_sources(result)
    return {
        "query": _clean_text(query_row.get("query")),
        "source": _clean_text(query_row.get("source")),
        "query_type": _clean_text(query_row.get("query_type")),
        "expected_primary_source": _clean_text(query_row.get("expected_primary_source")),
        "expected_answer_style": _clean_text(query_row.get("expected_answer_style")),
        "difficulty": _clean_text(query_row.get("difficulty")),
        "review_bucket": _clean_text(query_row.get("review_bucket")),
        "answer_status": _clean_text(result.get("status") or "unknown"),
        "answer_text": str(result.get("answer") or "").strip(),
        "answer_preview": _preview_text(result.get("answer")),
        "no_result": "1" if _clean_text(result.get("status")).lower() == "no_result" else "0",
        "needs_caution": "1" if bool(answer_verification.get("needsCaution")) else "0",
        "verification_status": _clean_text(answer_verification.get("status")),
        "verification_summary": _clean_text(answer_verification.get("summary")),
        "unsupported_claim_count": str(int(answer_verification.get("unsupportedClaimCount") or 0)),
        "source_count": str(int(source_count)),
        "source_titles": source_titles,
        "source_refs": source_refs,
        "runtime_used": _clean_text(family_diag.get("runtimeUsed") or family_diag.get("actualRuntimeUsed")),
        "answer_route": _clean_text(selected.get("route")),
        "router_provider": _clean_text(selected.get("provider")),
        "router_model": _clean_text(selected.get("model")),
        "latency_ms": str(round(float(latency_ms or 0.0), 3)),
        "top_k": str(max(1, int(top_k))),
        "retrieval_mode": _clean_text(retrieval_mode),
        "answer_backend": _clean_text(selected.get("route") or result.get("answerRoute") or "api"),
        "answer_backend_model": _clean_text(selected.get("model")),
        "packet_ref": "",
        "pred_label": "",
        "pred_groundedness": "",
        "pred_usefulness": "",
        "pred_readability": "",
        "pred_source_accuracy": "",
        "pred_should_abstain": "",
        "pred_confidence": "",
        "pred_reason": "",
        "judge_provider": "",
        "judge_model": "",
        "final_label": "",
        "final_groundedness": "",
        "final_usefulness": "",
        "final_readability": "",
        "final_source_accuracy": "",
        "final_should_abstain": "",
        "final_notes": "",
    }


def _error_row(
    query_row: dict[str, str],
    *,
    top_k: int,
    retrieval_mode: str,
    latency_ms: float,
    error: Exception,
) -> dict[str, str]:
    row = {field: "" for field in USER_ANSWER_EVAL_FIELDNAMES}
    row.update(
        {
            "query": _clean_text(query_row.get("query")),
            "source": _clean_text(query_row.get("source")),
            "query_type": _clean_text(query_row.get("query_type")),
            "expected_primary_source": _clean_text(query_row.get("expected_primary_source")),
            "expected_answer_style": _clean_text(query_row.get("expected_answer_style")),
            "difficulty": _clean_text(query_row.get("difficulty")),
            "review_bucket": _clean_text(query_row.get("review_bucket")),
            "answer_status": "error",
            "answer_text": f"error: {type(error).__name__}: {error}",
            "answer_preview": _preview_text(f"error: {type(error).__name__}: {error}"),
            "no_result": "0",
            "latency_ms": str(round(float(latency_ms or 0.0), 3)),
            "top_k": str(max(1, int(top_k))),
            "retrieval_mode": _clean_text(retrieval_mode),
            "final_notes": f"collector_error={type(error).__name__}: {error}",
        }
    )
    return row


def main() -> int:
    _auto_load_dotenv()
    parser = argparse.ArgumentParser(description="Collect user-facing answer eval rows with API answer generation.")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument(
        "--queries",
        default="eval/knowledgeos/queries/user_answer_eval_queries_v1.csv",
        help="CSV path with user-facing evaluation queries",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=8, help="Ask retrieval top-k")
    parser.add_argument("--mode", default="hybrid", choices=["semantic", "keyword", "hybrid"], help="Retrieval mode")
    parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid alpha")
    parser.add_argument(
        "--allow-external",
        action="store_true",
        default=True,
        help="Allow external/API generation. Default is enabled for this collector.",
    )
    parser.add_argument(
        "--no-allow-external",
        dest="allow_external",
        action="store_false",
        help="Disable external/API generation and keep the runtime local-only.",
    )
    parser.add_argument(
        "--force-api-route",
        action="store_true",
        default=True,
        help="Force answer generation onto the API route when external calls are allowed.",
    )
    parser.add_argument(
        "--no-force-api-route",
        dest="force_api_route",
        action="store_false",
        help="Let the router choose the final generation route.",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    factory = AppContextFactory(config_path=args.config)
    searcher = factory.get_searcher()

    rows: list[dict[str, str]] = []
    for item in _read_queries(queries_path):
        query = _clean_text(item.get("query"))
        source_type = _clean_text(item.get("source")) or None
        started = time.perf_counter()
        try:
            result = searcher.generate_answer(
                query,
                top_k=max(1, int(args.top_k)),
                source_type=source_type,
                retrieval_mode=str(args.mode),
                alpha=float(args.alpha),
                allow_external=bool(args.allow_external),
                answer_route_override="api" if args.allow_external and args.force_api_route else None,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            row = _serialize_row(
                item,
                result,
                top_k=max(1, int(args.top_k)),
                retrieval_mode=str(args.mode),
                latency_ms=latency_ms,
            )
        except Exception as error:
            latency_ms = (time.perf_counter() - started) * 1000.0
            row = _error_row(
                item,
                top_k=max(1, int(args.top_k)),
                retrieval_mode=str(args.mode),
                latency_ms=latency_ms,
                error=error,
            )
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=USER_ANSWER_EVAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    route_label = "api-forced" if args.allow_external and args.force_api_route else "router-default"
    print(
        f"Wrote user-answer eval sheet: {out_path} "
        f"({len(rows)} queries, mode={str(args.mode)}, top_k={max(1, int(args.top_k))}, route={route_label})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

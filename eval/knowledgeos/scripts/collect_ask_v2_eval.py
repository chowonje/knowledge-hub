#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from knowledge_hub.ai.ask_v2 import AskV2Service
from knowledge_hub.application.ask_v2_eval import ASK_V2_EVAL_FIELDNAMES, serialize_ask_v2_eval_row
from knowledge_hub.application.context import AppContextFactory


def _read_queries(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    items: list[dict[str, str]] = []
    for row in rows:
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        items.append({str(key): str(value or "").strip() for key, value in row.items()})
    return items


@contextmanager
def _readonly_ask_v2_runtime():
    original_paper = AskV2Service._ensure_paper_card
    original_web = AskV2Service._ensure_web_card
    original_vault = AskV2Service._ensure_vault_card
    try:
        AskV2Service._ensure_paper_card = lambda self, paper_id: self.sqlite_db.get_paper_card_v2(paper_id)  # type: ignore[method-assign]
        AskV2Service._ensure_web_card = lambda self, url: self.sqlite_db.get_web_card_v2_by_url(url)  # type: ignore[method-assign]
        AskV2Service._ensure_vault_card = lambda self, note_id: self.sqlite_db.get_vault_card_v2(note_id)  # type: ignore[method-assign]
        yield
    finally:
        AskV2Service._ensure_paper_card = original_paper  # type: ignore[method-assign]
        AskV2Service._ensure_web_card = original_web  # type: ignore[method-assign]
        AskV2Service._ensure_vault_card = original_vault  # type: ignore[method-assign]


def _error_row(query_row: dict[str, Any], *, top_k: int, retrieval_mode: str, latency_ms: float, error: Exception) -> dict[str, str]:
    row = serialize_ask_v2_eval_row(
        query_row,
        {
            "status": "error",
            "answer": f"error: {type(error).__name__}: {error}",
            "answerVerification": {"needsCaution": True},
            "v2": {
                "routing": {
                    "mode": "",
                    "intent": "",
                    "sourceKind": str(query_row.get("source") or "").strip(),
                    "matched_entities": [],
                    "selected_card_ids": [],
                },
                "evidenceVerification": {
                    "anchorIdsUsed": [],
                    "unsupportedFields": ["collector_error"],
                    "verificationStatus": "missing",
                },
                "fallback": {"used": False, "reason": "collector_error"},
                "consensus": {
                    "claimVerificationSummary": "",
                    "conflictCount": 0,
                    "weakClaimCount": 0,
                    "unsupportedClaimCount": 0,
                },
            },
        },
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        latency_ms=latency_ms,
    )
    row["notes"] = f"collector_error={type(error).__name__}: {error}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect ask-v2 manual-eval CSV rows from khub ask payloads.")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument(
        "--queries",
        default="eval/knowledgeos/queries/knowledgeos_ask_v2_eval_queries_v1.csv",
        help="CSV path with ask-v2 evaluation queries",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=6, help="Ask retrieval top-k")
    parser.add_argument("--mode", default="hybrid", choices=["semantic", "keyword", "hybrid"], help="Retrieval mode")
    parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid alpha")
    args = parser.parse_args()

    queries_path = Path(args.queries).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    factory = AppContextFactory(config_path=args.config)
    searcher = factory.get_searcher()

    rows: list[dict[str, str]] = []
    with _readonly_ask_v2_runtime():
        for item in _read_queries(queries_path):
            query = str(item.get("query") or "").strip()
            source_type = str(item.get("source") or "").strip() or None
            started = time.perf_counter()
            try:
                result = searcher.generate_answer(
                    query,
                    top_k=max(1, int(args.top_k)),
                    source_type=source_type,
                    retrieval_mode=str(args.mode),
                    alpha=float(args.alpha),
                    allow_external=False,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                row = serialize_ask_v2_eval_row(
                    item,
                    result,
                    top_k=max(1, int(args.top_k)),
                    retrieval_mode=str(args.mode),
                    latency_ms=latency_ms,
                )
            except Exception as error:  # pragma: no cover - collector should preserve failures in CSV
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
        writer = csv.DictWriter(handle, fieldnames=ASK_V2_EVAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote ask-v2 eval sheet: {out_path} "
        f"({len(rows)} queries, mode={str(args.mode)}, top_k={max(1, int(args.top_k))})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

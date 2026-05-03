#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


_REFUSAL_PATTERNS = (
    r"i(?:'m| am) unable to access",
    r"i cannot access",
    r"i do not have access",
    r"논문 전문을 직접 열람할 수 없",
    r"외부 문서.*읽을 수 없",
    r"원문을 업로드해",
    r"링크/p?df.*필요",
)

_VAULT_HUB_PATTERNS = (
    r"아틀라스",
    r"마인드맵",
    r"백링크",
    r"체크리스트",
    r"전체\s*정리",
    r"\b(?:hub|index|overview|atlas|mindmap|backlink|checklist)\b",
)


def _load_rows(path: Path, *, query_col: str = "query") -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    loaded: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get(query_col) or "").strip()
        rank = str(row.get("rank") or "1").strip()
        if not key or rank != "1":
            continue
        loaded[key] = {str(k): str(v or "") for k, v in row.items()}
    return loaded


def _normalize_boolish(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _final_label(row: dict[str, str]) -> str:
    return str(row.get("final_label") or row.get("label") or "").strip().lower()


def _reason_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^0-9A-Za-z가-힣_]+", str(text or "").strip().lower())
        if token
    }


def _classify_failure(machine: dict[str, str], review: dict[str, str]) -> str:
    final_label = _final_label(review)
    source = str(machine.get("source") or machine.get("expected_primary_source") or "").strip().lower()
    prefilter_reason = str(machine.get("memory_prefilter_reason") or "").strip().lower()
    insufficient_tokens = _reason_tokens(machine.get("insufficient_reasons") or "")
    if final_label == "good":
        return "good"
    if _normalize_boolish(review.get("final_wrong_era")):
        return "temporal_wrong_era"
    title_excerpt = " ".join([str(machine.get("top1_title") or ""), str(machine.get("top1_excerpt") or "")]).strip().lower()
    if str(machine.get("top1_rejected_reason") or "").strip().lower() == "vault_hub_noise" or (
        source == "vault" and any(re.search(pattern, title_excerpt, re.IGNORECASE) for pattern in _VAULT_HUB_PATTERNS)
    ):
        return "vault_hub_noise"
    if "non_substantive_evidence" in insufficient_tokens:
        return "non_substantive_evidence"
    if _normalize_boolish(machine.get("no_result")) or str(machine.get("answer_status") or "").strip().lower() == "no_result":
        if source == "vault" and prefilter_reason in {"no_memory_hits", "vault_chunk_fallback"}:
            return "vault_memory_miss"
        if source == "web" and prefilter_reason in {"prefilter_no_ranked_results", "web_chunk_fallback_success", "no_memory_hits"}:
            return "web_ranked_chunk_miss"
        if source == "all" and prefilter_reason in {"mixed_fallback_no_hit", "mixed_fallback"}:
            return "mixed_fallback_no_hit"
        return "no_result_or_coverage_miss"
    if _normalize_boolish(review.get("final_should_abstain")):
        if "missing_temporal_grounding" in insufficient_tokens:
            return "temporal_no_grounding"
    excerpt = str(machine.get("top1_excerpt") or "").strip().lower()
    if excerpt and any(re.search(pattern, excerpt, re.IGNORECASE) for pattern in _REFUSAL_PATTERNS):
        return "non_substantive_evidence"
    expected_source = str(machine.get("expected_primary_source") or "").strip().lower()
    actual_source = str(machine.get("top1_source_type") or "").strip().lower()
    if expected_source in {"paper", "vault", "web"} and actual_source and actual_source != expected_source:
        return "source_mismatch"
    if str(machine.get("memory_prefilter_reason") or "").strip() == "memory_prefilter_chunk_fallback" and final_label in {"partial", "good"}:
        return "memory_miss_chunk_fallback_success"
    return "other_failure"


def build_taxonomy(machine_csv: Path, review_csv: Path) -> dict[str, Any]:
    machine_rows = _load_rows(machine_csv)
    review_rows = _load_rows(review_csv)
    categories: Counter[str] = Counter()
    details: list[dict[str, Any]] = []

    for query, machine in machine_rows.items():
        review = review_rows.get(query, {})
        category = _classify_failure(machine, review)
        categories[category] += 1
        details.append(
            {
                "query": query,
                "category": category,
                "expected_primary_source": machine.get("expected_primary_source", ""),
                "top1_source_type": machine.get("top1_source_type", ""),
                "answer_status": machine.get("answer_status", ""),
                "memory_prefilter_reason": machine.get("memory_prefilter_reason", ""),
                "insufficient_reasons": machine.get("insufficient_reasons", ""),
                "final_label": review.get("final_label", ""),
                "final_wrong_era": review.get("final_wrong_era", ""),
                "final_should_abstain": review.get("final_should_abstain", ""),
            }
        )

    return {
        "schema": "knowledge-hub.memory-router.failure-taxonomy.v1",
        "machineCsvPath": str(machine_csv),
        "reviewCsvPath": str(review_csv),
        "queryCount": len(machine_rows),
        "categoryCounts": dict(sorted(categories.items(), key=lambda item: (-item[1], item[0]))),
        "rows": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a failure taxonomy summary from machine eval and human review CSVs.")
    parser.add_argument("--machine-csv", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    payload = build_taxonomy(Path(args.machine_csv), Path(args.review_csv))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "outJson": str(out_path), "queryCount": payload["queryCount"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

REVIEW_WINNERS = {"baseline", "candidate", "tie", "review"}
RELEVANCE_LABELS = {"good", "partial", "bad"}
LOCAL_PROVIDERS = {"ollama", "local", "sentence_transformers", "tei", "huggingface"}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def _read_query_payloads(run_dir: Path, model_label: str) -> dict[str, dict[str, Any]]:
    model_dir = run_dir / model_label
    if not model_dir.exists():
        raise FileNotFoundError(f"model results directory missing: {model_dir}")
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(model_dir.glob("q*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        query = str(payload.get("query") or "").strip()
        if query:
            payloads[query] = payload
    if not payloads:
        raise ValueError(f"no query payloads found for model {model_label!r} under {model_dir}")
    return payloads


def _bool_flag(value: bool) -> str:
    return "1" if value else "0"


def _parse_doc_ids(raw: str) -> list[str]:
    if not raw:
        return []
    normalized = raw.replace("|", ",").replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _top_result(payload: dict[str, Any]) -> dict[str, Any]:
    results = list(payload.get("results") or [])
    return dict(results[0] or {}) if results else {}


def _source_hit(results: list[dict[str, Any]], expected_source: str, *, top_k: int) -> bool:
    if not expected_source:
        return False
    return any(str(item.get("sourceType") or "").strip() == expected_source for item in results[:top_k])


def _source_mrr(results: list[dict[str, Any]], expected_source: str, *, top_k: int) -> float:
    if not expected_source:
        return 0.0
    for rank, item in enumerate(results[:top_k], start=1):
        if str(item.get("sourceType") or "").strip() == expected_source:
            return 1.0 / float(rank)
    return 0.0


def _doc_ids(results: list[dict[str, Any]], *, top_k: int) -> list[str]:
    return [str(item.get("documentId") or "").strip() for item in results[:top_k] if str(item.get("documentId") or "").strip()]


def _gold_hit_at1(results: list[dict[str, Any]], gold_doc_ids: set[str]) -> bool:
    if not results or not gold_doc_ids:
        return False
    return str(results[0].get("documentId") or "").strip() in gold_doc_ids


def _gold_hit_at3(results: list[dict[str, Any]], gold_doc_ids: set[str], *, top_k: int) -> bool:
    if not gold_doc_ids:
        return False
    return any(doc_id in gold_doc_ids for doc_id in _doc_ids(results, top_k=top_k))


def _gold_mrr(results: list[dict[str, Any]], gold_doc_ids: set[str], *, top_k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    for rank, item in enumerate(results[:top_k], start=1):
        if str(item.get("documentId") or "").strip() in gold_doc_ids:
            return 1.0 / float(rank)
    return 0.0


def _hard_negative_top1(results: list[dict[str, Any]], hard_negative_doc_ids: set[str]) -> bool:
    if not results or not hard_negative_doc_ids:
        return False
    return str(results[0].get("documentId") or "").strip() in hard_negative_doc_ids


def _hard_negative_top3(results: list[dict[str, Any]], hard_negative_doc_ids: set[str], *, top_k: int) -> bool:
    if not hard_negative_doc_ids:
        return False
    return any(doc_id in hard_negative_doc_ids for doc_id in _doc_ids(results, top_k=top_k))


def _predict_relevance(
    *,
    has_gold_labels: bool,
    top1_gold_hit: bool,
    top3_gold_hit: bool,
    top1_source_match: bool,
    top3_source_hit: bool,
) -> tuple[str, str]:
    if has_gold_labels:
        if top1_gold_hit:
            return ("good", "gold top1 hit")
        if top3_gold_hit:
            return ("partial", "gold top3 hit only")
        return ("bad", "no gold document found in top3")
    if top1_source_match:
        return ("good", "source-hit fallback: top1 source matches expected source")
    if top3_source_hit:
        return ("partial", "source-hit fallback: expected source appears within top3")
    return ("bad", "source-hit fallback: expected source absent within top3")


def _pairwise_winner(
    *,
    has_gold_labels: bool,
    baseline_hit1: bool,
    baseline_hit3: bool,
    baseline_mrr: float,
    baseline_hn1: bool,
    baseline_hn3: bool,
    baseline_no_result: bool,
    candidate_hit1: bool,
    candidate_hit3: bool,
    candidate_mrr: float,
    candidate_hn1: bool,
    candidate_hn3: bool,
    candidate_no_result: bool,
    baseline_top1_source_match: bool,
    baseline_top3_source_hit: bool,
    candidate_top1_source_match: bool,
    candidate_top3_source_hit: bool,
    same_top1: bool,
) -> tuple[str, str, str]:
    if candidate_hn1 and not baseline_hn1:
        return ("baseline", "0.95", "candidate top1 is a known hard negative while baseline top1 is not")
    if baseline_hn1 and not candidate_hn1:
        return ("candidate", "0.95", "baseline top1 is a known hard negative while candidate top1 is not")

    if has_gold_labels:
        baseline_tuple = (1 if baseline_hit1 else 0, 1 if baseline_hit3 else 0, baseline_mrr, 0 if baseline_hn3 else 1, 0 if baseline_no_result else 1)
        candidate_tuple = (1 if candidate_hit1 else 0, 1 if candidate_hit3 else 0, candidate_mrr, 0 if candidate_hn3 else 1, 0 if candidate_no_result else 1)
        if candidate_tuple > baseline_tuple:
            return ("candidate", "0.9", "candidate has stronger gold-document retrieval and fallback signals")
        if baseline_tuple > candidate_tuple:
            return ("baseline", "0.9", "baseline has stronger gold-document retrieval and fallback signals")
        if same_top1:
            return ("tie", "0.6", "both models return the same top1 document")
        return ("review", "0.45", "gold metrics tie; human review required")

    if candidate_top1_source_match and not baseline_top1_source_match:
        return ("candidate", "0.7", "source-hit fallback: candidate top1 source matches expected source")
    if baseline_top1_source_match and not candidate_top1_source_match:
        return ("baseline", "0.7", "source-hit fallback: baseline top1 source matches expected source")
    if candidate_top3_source_hit and not baseline_top3_source_hit:
        return ("candidate", "0.6", "source-hit fallback: candidate reaches expected source within top3")
    if baseline_top3_source_hit and not candidate_top3_source_hit:
        return ("baseline", "0.6", "source-hit fallback: baseline reaches expected source within top3")
    if baseline_no_result and not candidate_no_result:
        return ("candidate", "0.55", "candidate returns evidence while baseline has no result")
    if candidate_no_result and not baseline_no_result:
        return ("baseline", "0.55", "baseline returns evidence while candidate has no result")
    if same_top1:
        return ("tie", "0.4", "both models return the same top1 document")
    return ("review", "0.3", "missing gold labels; heuristic fallback is inconclusive")


def _model_metadata(payloads: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    first_payload = next(iter(payloads.values()))
    provider = str(first_payload.get("provider") or "")
    model = str(first_payload.get("model") or label)
    embedding_dimension = first_payload.get("embeddingDimension") or first_payload.get("vectorDim") or first_payload.get("dimension") or ""
    is_local = provider in LOCAL_PROVIDERS
    return {
        "label": label,
        "model": model,
        "provider": provider,
        "embedding_dimension": embedding_dimension,
        "is_local": is_local,
        "requires_hosted_api": not is_local,
    }


def _build_rows(
    *,
    query_rows: list[dict[str, str]],
    baseline_payloads: dict[str, dict[str, Any]],
    candidate_payloads: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for query_row in query_rows:
        query = str(query_row.get("query") or "").strip()
        if not query:
            continue
        baseline_payload = baseline_payloads.get(query)
        candidate_payload = candidate_payloads.get(query)
        if baseline_payload is None:
            raise KeyError(f"baseline payload missing for query: {query}")
        if candidate_payload is None:
            raise KeyError(f"candidate payload missing for query: {query}")

        baseline_results = list(baseline_payload.get("results") or [])
        candidate_results = list(candidate_payload.get("results") or [])
        baseline_top = _top_result(baseline_payload)
        candidate_top = _top_result(candidate_payload)
        expected_source = str(query_row.get("expected_primary_source") or "").strip()
        gold_doc_ids = set(_parse_doc_ids(str(query_row.get("gold_doc_ids") or "")))
        hard_negative_doc_ids = set(_parse_doc_ids(str(query_row.get("hard_negative_doc_ids") or "")))
        has_gold_labels = bool(gold_doc_ids)

        baseline_top1_source_match = bool(expected_source and str(baseline_top.get("sourceType") or "") == expected_source)
        candidate_top1_source_match = bool(expected_source and str(candidate_top.get("sourceType") or "") == expected_source)
        baseline_top3_source_hit = _source_hit(baseline_results, expected_source, top_k=top_k)
        candidate_top3_source_hit = _source_hit(candidate_results, expected_source, top_k=top_k)
        baseline_source_mrr = _source_mrr(baseline_results, expected_source, top_k=top_k)
        candidate_source_mrr = _source_mrr(candidate_results, expected_source, top_k=top_k)

        baseline_gold_hit1 = _gold_hit_at1(baseline_results, gold_doc_ids)
        candidate_gold_hit1 = _gold_hit_at1(candidate_results, gold_doc_ids)
        baseline_gold_hit3 = _gold_hit_at3(baseline_results, gold_doc_ids, top_k=top_k)
        candidate_gold_hit3 = _gold_hit_at3(candidate_results, gold_doc_ids, top_k=top_k)
        baseline_gold_mrr = _gold_mrr(baseline_results, gold_doc_ids, top_k=top_k)
        candidate_gold_mrr = _gold_mrr(candidate_results, gold_doc_ids, top_k=top_k)

        baseline_hard_negative_top1 = _hard_negative_top1(baseline_results, hard_negative_doc_ids)
        candidate_hard_negative_top1 = _hard_negative_top1(candidate_results, hard_negative_doc_ids)
        baseline_hard_negative_top3 = _hard_negative_top3(baseline_results, hard_negative_doc_ids, top_k=top_k)
        candidate_hard_negative_top3 = _hard_negative_top3(candidate_results, hard_negative_doc_ids, top_k=top_k)

        baseline_no_result = not baseline_results
        candidate_no_result = not candidate_results
        same_top1 = (
            str(baseline_top.get("documentId") or "").strip()
            and str(baseline_top.get("documentId") or "").strip() == str(candidate_top.get("documentId") or "").strip()
        )

        baseline_pred_top1_relevance, baseline_pred_top1_reason = _predict_relevance(
            has_gold_labels=has_gold_labels,
            top1_gold_hit=baseline_gold_hit1,
            top3_gold_hit=baseline_gold_hit3,
            top1_source_match=baseline_top1_source_match,
            top3_source_hit=baseline_top3_source_hit,
        )
        candidate_pred_top1_relevance, candidate_pred_top1_reason = _predict_relevance(
            has_gold_labels=has_gold_labels,
            top1_gold_hit=candidate_gold_hit1,
            top3_gold_hit=candidate_gold_hit3,
            top1_source_match=candidate_top1_source_match,
            top3_source_hit=candidate_top3_source_hit,
        )
        baseline_pred_top3_relevance = "good" if baseline_gold_hit3 else ("partial" if baseline_top3_source_hit else "bad")
        candidate_pred_top3_relevance = "good" if candidate_gold_hit3 else ("partial" if candidate_top3_source_hit else "bad")

        pred_pairwise_winner, pred_confidence, pred_reason = _pairwise_winner(
            has_gold_labels=has_gold_labels,
            baseline_hit1=baseline_gold_hit1,
            baseline_hit3=baseline_gold_hit3,
            baseline_mrr=baseline_gold_mrr,
            baseline_hn1=baseline_hard_negative_top1,
            baseline_hn3=baseline_hard_negative_top3,
            baseline_no_result=baseline_no_result,
            candidate_hit1=candidate_gold_hit1,
            candidate_hit3=candidate_gold_hit3,
            candidate_mrr=candidate_gold_mrr,
            candidate_hn1=candidate_hard_negative_top1,
            candidate_hn3=candidate_hard_negative_top3,
            candidate_no_result=candidate_no_result,
            baseline_top1_source_match=baseline_top1_source_match,
            baseline_top3_source_hit=baseline_top3_source_hit,
            candidate_top1_source_match=candidate_top1_source_match,
            candidate_top3_source_hit=candidate_top3_source_hit,
            same_top1=same_top1,
        )
        if pred_pairwise_winner not in REVIEW_WINNERS:
            raise ValueError(f"invalid predicted winner: {pred_pairwise_winner}")

        rows.append(
            {
                "query": query,
                "query_variant_group": str(query_row.get("query_variant_group") or "").strip(),
                "source": str(query_row.get("source") or "").strip(),
                "query_type": str(query_row.get("query_type") or "").strip(),
                "temporal_query": str(query_row.get("temporal_query") or "").strip(),
                "expected_primary_source": expected_source,
                "expected_answer_style": str(query_row.get("expected_answer_style") or "").strip(),
                "difficulty": str(query_row.get("difficulty") or "").strip(),
                "gold_doc_ids": ";".join(sorted(gold_doc_ids)),
                "hard_negative_doc_ids": ";".join(sorted(hard_negative_doc_ids)),
                "notes": str(query_row.get("notes") or "").strip(),
                "has_gold_labels": _bool_flag(has_gold_labels),
                "baseline_top1_doc_id": str(baseline_top.get("documentId") or ""),
                "baseline_top1_title": str(baseline_top.get("title") or ""),
                "baseline_top1_source_type": str(baseline_top.get("sourceType") or ""),
                "baseline_top1_score": str(baseline_top.get("score") or ""),
                "baseline_top3_doc_ids": ";".join(_doc_ids(baseline_results, top_k=top_k)),
                "baseline_top1_source_match": _bool_flag(baseline_top1_source_match),
                "baseline_top3_source_hit": _bool_flag(baseline_top3_source_hit),
                "baseline_source_mrr_at3": f"{baseline_source_mrr:.4f}",
                "baseline_gold_hit_at1": _bool_flag(baseline_gold_hit1),
                "baseline_gold_hit_at3": _bool_flag(baseline_gold_hit3),
                "baseline_gold_mrr_at3": f"{baseline_gold_mrr:.4f}",
                "baseline_hard_negative_top1": _bool_flag(baseline_hard_negative_top1),
                "baseline_hard_negative_top3": _bool_flag(baseline_hard_negative_top3),
                "baseline_pred_top1_relevance": baseline_pred_top1_relevance,
                "baseline_pred_top1_reason": baseline_pred_top1_reason,
                "baseline_pred_top3_relevance": baseline_pred_top3_relevance,
                "baseline_result_count": str(len(baseline_results)),
                "baseline_latency_ms": str(round(float(baseline_payload.get("latencyMs") or 0.0), 3)),
                "candidate_top1_doc_id": str(candidate_top.get("documentId") or ""),
                "candidate_top1_title": str(candidate_top.get("title") or ""),
                "candidate_top1_source_type": str(candidate_top.get("sourceType") or ""),
                "candidate_top1_score": str(candidate_top.get("score") or ""),
                "candidate_top3_doc_ids": ";".join(_doc_ids(candidate_results, top_k=top_k)),
                "candidate_top1_source_match": _bool_flag(candidate_top1_source_match),
                "candidate_top3_source_hit": _bool_flag(candidate_top3_source_hit),
                "candidate_source_mrr_at3": f"{candidate_source_mrr:.4f}",
                "candidate_gold_hit_at1": _bool_flag(candidate_gold_hit1),
                "candidate_gold_hit_at3": _bool_flag(candidate_gold_hit3),
                "candidate_gold_mrr_at3": f"{candidate_gold_mrr:.4f}",
                "candidate_hard_negative_top1": _bool_flag(candidate_hard_negative_top1),
                "candidate_hard_negative_top3": _bool_flag(candidate_hard_negative_top3),
                "candidate_pred_top1_relevance": candidate_pred_top1_relevance,
                "candidate_pred_top1_reason": candidate_pred_top1_reason,
                "candidate_pred_top3_relevance": candidate_pred_top3_relevance,
                "candidate_result_count": str(len(candidate_results)),
                "candidate_latency_ms": str(round(float(candidate_payload.get("latencyMs") or 0.0), 3)),
                "same_top1_document": _bool_flag(same_top1),
                "baseline_no_result": _bool_flag(baseline_no_result),
                "candidate_no_result": _bool_flag(candidate_no_result),
                "pred_pairwise_winner": pred_pairwise_winner,
                "pred_confidence": pred_confidence,
                "pred_reason": pred_reason,
                "final_pairwise_winner": "",
                "review_notes": "",
            }
        )
    return rows


def _metrics_for_rows(rows: list[dict[str, str]], prefix: str) -> dict[str, Any]:
    total = len(rows)
    no_result = sum(1 for row in rows if row[f"{prefix}_no_result"] == "1")
    top1_source_hit = sum(1 for row in rows if row[f"{prefix}_top1_source_match"] == "1")
    top3_source_hit = sum(1 for row in rows if row[f"{prefix}_top3_source_hit"] == "1")
    top1_gold = sum(1 for row in rows if row[f"{prefix}_gold_hit_at1"] == "1")
    top3_gold = sum(1 for row in rows if row[f"{prefix}_gold_hit_at3"] == "1")
    hard_negative_top1 = sum(1 for row in rows if row[f"{prefix}_hard_negative_top1"] == "1")
    hard_negative_top3 = sum(1 for row in rows if row[f"{prefix}_hard_negative_top3"] == "1")
    latency_values = [float(row[f"{prefix}_latency_ms"]) for row in rows if row[f"{prefix}_latency_ms"]]
    source_mrr_values = [float(row[f"{prefix}_source_mrr_at3"]) for row in rows if row[f"{prefix}_source_mrr_at3"]]
    gold_mrr_values = [float(row[f"{prefix}_gold_mrr_at3"]) for row in rows if row[f"{prefix}_gold_mrr_at3"]]
    return {
        "query_count": total,
        "top1_source_hit_rate": round(top1_source_hit / total, 4) if total else 0.0,
        "top3_source_hit_rate": round(top3_source_hit / total, 4) if total else 0.0,
        "gold_hit_at1_rate": round(top1_gold / total, 4) if total else 0.0,
        "gold_hit_at3_rate": round(top3_gold / total, 4) if total else 0.0,
        "hard_negative_top1_rate": round(hard_negative_top1 / total, 4) if total else 0.0,
        "hard_negative_top3_rate": round(hard_negative_top3 / total, 4) if total else 0.0,
        "no_result_rate": round(no_result / total, 4) if total else 0.0,
        "avg_latency_ms": round(sum(latency_values) / len(latency_values), 3) if latency_values else 0.0,
        "mean_source_mrr_at3": round(sum(source_mrr_values) / len(source_mrr_values), 4) if source_mrr_values else 0.0,
        "mean_gold_mrr_at3": round(sum(gold_mrr_values) / len(gold_mrr_values), 4) if gold_mrr_values else 0.0,
    }


def _promotion_gate(*, rows: list[dict[str, str]], baseline_meta: dict[str, Any], candidate_meta: dict[str, Any], baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> dict[str, Any]:
    total = len(rows)
    review_count = sum(1 for row in rows if row["pred_pairwise_winner"] == "review")
    candidate_wins = sum(1 for row in rows if row["pred_pairwise_winner"] == "candidate")
    baseline_wins = sum(1 for row in rows if row["pred_pairwise_winner"] == "baseline")
    has_complete_gold_labels = bool(rows) and all(row.get("has_gold_labels") == "1" for row in rows)
    relevance_not_worse = (
        candidate_metrics["gold_hit_at1_rate"] >= baseline_metrics["gold_hit_at1_rate"]
        and candidate_metrics["gold_hit_at3_rate"] >= baseline_metrics["gold_hit_at3_rate"]
        and candidate_metrics["mean_gold_mrr_at3"] >= baseline_metrics["mean_gold_mrr_at3"]
    )
    hard_negative_not_worse = candidate_metrics["hard_negative_top1_rate"] <= baseline_metrics["hard_negative_top1_rate"]
    no_result_not_worse = candidate_metrics["no_result_rate"] <= baseline_metrics["no_result_rate"]
    latency_within_2x = candidate_metrics["avg_latency_ms"] <= (baseline_metrics["avg_latency_ms"] * 2.0 if baseline_metrics["avg_latency_ms"] else float("inf"))
    review_fraction = round(review_count / total, 4) if total else 0.0
    review_fraction_acceptable = review_fraction <= 0.35
    locally_compatible = bool(candidate_meta["is_local"])
    strong_quality_gain = (
        candidate_metrics["gold_hit_at1_rate"] > baseline_metrics["gold_hit_at1_rate"]
        or candidate_metrics["gold_hit_at3_rate"] > baseline_metrics["gold_hit_at3_rate"]
        or candidate_metrics["mean_gold_mrr_at3"] > baseline_metrics["mean_gold_mrr_at3"]
    )
    hosted_only_blocked = candidate_meta["requires_hosted_api"] and not strong_quality_gain
    recommended = all(
        [
            has_complete_gold_labels,
            relevance_not_worse,
            hard_negative_not_worse,
            no_result_not_worse,
            latency_within_2x,
            review_fraction_acceptable,
            not hosted_only_blocked,
            candidate_wins >= baseline_wins,
        ]
    )
    reasons: list[str] = []
    if not has_complete_gold_labels:
        reasons.append("gold_doc_ids are incomplete; recommendation is blocked until the 30-query set is curated")
    if not relevance_not_worse:
        reasons.append("candidate gold relevance metrics are worse than baseline")
    if not hard_negative_not_worse:
        reasons.append("candidate hard-negative top1 rate is worse than baseline")
    if not no_result_not_worse:
        reasons.append("candidate no-result rate is worse than baseline")
    if not latency_within_2x:
        reasons.append("candidate average latency exceeds 2x baseline")
    if not review_fraction_acceptable:
        reasons.append("pairwise review fraction is too high for automatic promotion")
    if hosted_only_blocked:
        reasons.append("hosted-only candidate is blocked by local-first gate without clear quality gain")
    if candidate_wins < baseline_wins:
        reasons.append("candidate does not beat baseline in pairwise draft wins")
    return {
        "has_complete_gold_labels": has_complete_gold_labels,
        "relevance_not_worse": relevance_not_worse,
        "hard_negative_top1_not_worse": hard_negative_not_worse,
        "no_result_not_worse": no_result_not_worse,
        "latency_within_2x": latency_within_2x,
        "review_fraction": review_fraction,
        "review_fraction_acceptable": review_fraction_acceptable,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "hosted_only_blocked": hosted_only_blocked,
        "recommended": recommended,
        "reasons": reasons,
    }


def _default_output_paths(run_dir: Path, baseline_label: str, candidate_label: str) -> dict[str, Path]:
    stem = f"{baseline_label}__vs__{candidate_label}_embedding"
    return {
        "machine": run_dir / f"{stem}_machine_review.csv",
        "human": run_dir / f"{stem}_human_review.csv",
        "summary_json": run_dir / f"{stem}_summary.json",
        "summary_md": run_dir / f"{stem}_summary.md",
    }


def _write_csv(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(summary: dict[str, Any]) -> str:
    def _fmt_pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    def _fmt_ms(value: float) -> str:
        return f"{value:.1f}ms"

    def _fmt_delta(candidate: float, baseline: float, *, percent: bool = False) -> str:
        if percent:
            delta = (candidate - baseline) * 100.0
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta:.1f}pp"
        delta = candidate - baseline
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.4f}"

    baseline = summary["baseline"]
    candidate = summary["candidate"]
    gate = summary["promotionGate"]
    lines = [
        "# Embedding Review Summary",
        "",
        f"- baseline: `{summary['baselineLabel']}` ({summary['baselineMeta']['provider'] or 'unknown'})",
        f"- candidate: `{summary['candidateLabel']}` ({summary['candidateMeta']['provider'] or 'unknown'})",
        f"- queries: `{summary['queryCount']}`",
        "",
        "## Core Metrics",
        "",
        "| metric | baseline | candidate | delta |",
        "|---|---:|---:|---:|",
        f"| gold_hit_at1_rate | {_fmt_pct(baseline['gold_hit_at1_rate'])} | {_fmt_pct(candidate['gold_hit_at1_rate'])} | {_fmt_delta(candidate['gold_hit_at1_rate'], baseline['gold_hit_at1_rate'], percent=True)} |",
        f"| gold_hit_at3_rate | {_fmt_pct(baseline['gold_hit_at3_rate'])} | {_fmt_pct(candidate['gold_hit_at3_rate'])} | {_fmt_delta(candidate['gold_hit_at3_rate'], baseline['gold_hit_at3_rate'], percent=True)} |",
        f"| mean_gold_mrr_at3 | {baseline['mean_gold_mrr_at3']:.4f} | {candidate['mean_gold_mrr_at3']:.4f} | {_fmt_delta(candidate['mean_gold_mrr_at3'], baseline['mean_gold_mrr_at3'])} |",
        f"| hard_negative_top1_rate | {_fmt_pct(baseline['hard_negative_top1_rate'])} | {_fmt_pct(candidate['hard_negative_top1_rate'])} | {_fmt_delta(candidate['hard_negative_top1_rate'], baseline['hard_negative_top1_rate'], percent=True)} |",
        f"| no_result_rate | {_fmt_pct(baseline['no_result_rate'])} | {_fmt_pct(candidate['no_result_rate'])} | {_fmt_delta(candidate['no_result_rate'], baseline['no_result_rate'], percent=True)} |",
        f"| avg_latency_ms | {_fmt_ms(baseline['avg_latency_ms'])} | {_fmt_ms(candidate['avg_latency_ms'])} | {candidate['avg_latency_ms'] - baseline['avg_latency_ms']:+.1f}ms |",
        "",
        "## Source-Hit Backstop",
        "",
        f"- baseline top1/top3 source-hit: {_fmt_pct(baseline['top1_source_hit_rate'])} / {_fmt_pct(baseline['top3_source_hit_rate'])}",
        f"- candidate top1/top3 source-hit: {_fmt_pct(candidate['top1_source_hit_rate'])} / {_fmt_pct(candidate['top3_source_hit_rate'])}",
        "",
        "## Pairwise Draft Winner Distribution",
        "",
    ]
    for key, value in sorted(summary["pairwiseDraftCounts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Operating Metadata",
            "",
            f"- baseline local-first: `{baseline['is_local']}` / hosted API required: `{baseline['requires_hosted_api']}` / embedding dimension: `{baseline['embedding_dimension']}`",
            f"- candidate local-first: `{candidate['is_local']}` / hosted API required: `{candidate['requires_hosted_api']}` / embedding dimension: `{candidate['embedding_dimension']}`",
            "",
            "## Promotion Gate",
            "",
            f"- recommended: `{gate['recommended']}`",
            f"- has_complete_gold_labels: `{gate['has_complete_gold_labels']}`",
            f"- review_fraction: `{gate['review_fraction']}`",
            f"- candidate_wins / baseline_wins: `{gate['candidate_wins']}` / `{gate['baseline_wins']}`",
        ]
    )
    if gate["reasons"]:
        lines.extend(["", "### Hold Reasons", ""])
        for reason in gate["reasons"]:
            lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def _build_pair(
    *,
    query_rows: list[dict[str, str]],
    run_dir: Path,
    baseline_label: str,
    candidate_label: str,
    top_k: int,
    machine_out: Path,
    human_out: Path,
    summary_json: Path,
    summary_md: Path,
) -> dict[str, Any]:
    baseline_payloads = _read_query_payloads(run_dir, baseline_label)
    candidate_payloads = _read_query_payloads(run_dir, candidate_label)
    rows = _build_rows(
        query_rows=query_rows,
        baseline_payloads=baseline_payloads,
        candidate_payloads=candidate_payloads,
        top_k=top_k,
    )
    fieldnames = list(rows[0].keys()) if rows else []
    _write_csv(machine_out, rows, fieldnames=fieldnames)
    human_rows = [
        {
            "query": row["query"],
            "query_variant_group": row["query_variant_group"],
            "source": row["source"],
            "query_type": row["query_type"],
            "temporal_query": row["temporal_query"],
            "expected_primary_source": row["expected_primary_source"],
            "expected_answer_style": row["expected_answer_style"],
            "difficulty": row["difficulty"],
            "gold_doc_ids": row["gold_doc_ids"],
            "hard_negative_doc_ids": row["hard_negative_doc_ids"],
            "baseline_top1_doc_id": row["baseline_top1_doc_id"],
            "baseline_top1_title": row["baseline_top1_title"],
            "candidate_top1_doc_id": row["candidate_top1_doc_id"],
            "candidate_top1_title": row["candidate_top1_title"],
            "pred_pairwise_winner": row["pred_pairwise_winner"],
            "pred_confidence": row["pred_confidence"],
            "pred_reason": row["pred_reason"],
            "final_pairwise_winner": row["final_pairwise_winner"],
            "review_notes": row["review_notes"],
        }
        for row in rows
    ]
    _write_csv(human_out, human_rows, fieldnames=list(human_rows[0].keys()) if human_rows else [])

    baseline_meta = _model_metadata(baseline_payloads, baseline_label)
    candidate_meta = _model_metadata(candidate_payloads, candidate_label)
    baseline_metrics = _metrics_for_rows(rows, "baseline")
    candidate_metrics = _metrics_for_rows(rows, "candidate")
    pairwise_counts: dict[str, int] = {winner: 0 for winner in REVIEW_WINNERS}
    for row in rows:
        pairwise_counts[row["pred_pairwise_winner"]] += 1

    summary = {
        "baselineLabel": baseline_label,
        "candidateLabel": candidate_label,
        "queryCount": len(rows),
        "baselineMeta": baseline_meta,
        "candidateMeta": candidate_meta,
        "baseline": {**baseline_metrics, **{k: baseline_meta[k] for k in ("provider", "is_local", "requires_hosted_api", "embedding_dimension", "model")}},
        "candidate": {**candidate_metrics, **{k: candidate_meta[k] for k in ("provider", "is_local", "requires_hosted_api", "embedding_dimension", "model")}},
        "pairwiseDraftCounts": pairwise_counts,
        "promotionGate": _promotion_gate(
            rows=rows,
            baseline_meta=baseline_meta,
            candidate_meta=candidate_meta,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
        ),
        "paths": {
            "machineReviewCsv": str(machine_out),
            "humanReviewCsv": str(human_out),
            "summaryJson": str(summary_json),
            "summaryMarkdown": str(summary_md),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embedding promotion review sheets from A/B or multi-model retrieval run artifacts.")
    parser.add_argument("--query-csv", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", action="append", required=True, help="Repeat to compare multiple candidates against the current default.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--machine-out", type=Path)
    parser.add_argument("--review-out", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    query_rows = _read_csv_rows(args.query_csv)
    candidate_labels: list[str] = []
    for raw in args.candidate_label:
        for item in str(raw).split(","):
            label = item.strip()
            if label and label not in candidate_labels and label != args.baseline_label:
                candidate_labels.append(label)
    if not candidate_labels:
        raise SystemExit("at least one candidate label is required")
    if len(candidate_labels) > 1 and any([args.machine_out, args.review_out, args.summary_json, args.summary_md]):
        raise SystemExit("custom output paths are only supported for a single pairwise comparison")

    pairs: list[tuple[str, str]] = [(args.baseline_label, label) for label in candidate_labels]
    if len(candidate_labels) > 1:
        pairs.extend(itertools.combinations(candidate_labels, 2))

    pair_summaries: list[dict[str, Any]] = []
    for baseline_label, candidate_label in pairs:
        outputs = _default_output_paths(args.run_dir, baseline_label, candidate_label)
        if len(pairs) == 1:
            outputs = {
                "machine": args.machine_out or outputs["machine"],
                "human": args.review_out or outputs["human"],
                "summary_json": args.summary_json or outputs["summary_json"],
                "summary_md": args.summary_md or outputs["summary_md"],
            }
        summary = _build_pair(
            query_rows=query_rows,
            run_dir=args.run_dir,
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            top_k=args.top_k,
            machine_out=outputs["machine"],
            human_out=outputs["human"],
            summary_json=outputs["summary_json"],
            summary_md=outputs["summary_md"],
        )
        pair_summaries.append(summary)

    payload: dict[str, Any] = {
        "queryCsv": str(args.query_csv),
        "runDir": str(args.run_dir),
        "baselineLabel": args.baseline_label,
        "candidateLabels": candidate_labels,
        "pairSummaries": [
            {
                "baselineLabel": summary["baselineLabel"],
                "candidateLabel": summary["candidateLabel"],
                **summary["paths"],
                "recommended": summary["promotionGate"]["recommended"],
            }
            for summary in pair_summaries
        ],
    }
    if len(pair_summaries) == 1:
        payload.update(pair_summaries[0]["paths"])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

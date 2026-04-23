from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_eval_queries(path: str | Path) -> list[str]:
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in rows if line.strip() and not line.strip().startswith("#")]


def compute_precision_metrics(
    csv_path: str | Path,
    *,
    label_col: str,
    source_col: str,
    query_col: str = "query",
    rank_col: str = "rank",
    top_k: int = 5,
    hit_k: int = 3,
) -> dict[str, float | int]:
    rows = list(csv.DictReader(Path(csv_path).open("r", encoding="utf-8")))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(query_col, "")).strip()].append(row)

    precision_values: list[float] = []
    hit_values: list[float] = []
    vault_values: list[float] = []
    labeled_queries = 0

    for query, items in grouped.items():
        if not query:
            continue
        ordered = sorted(items, key=lambda row: int(str(row.get(rank_col, "0") or "0")))
        top_items = ordered[: max(1, int(top_k))]
        labeled_top = [row for row in top_items if str(row.get(label_col, "")).strip() in {"0", "1"}]
        if labeled_top:
            labeled_queries += 1
        labels = [int(str(row.get(label_col, "")).strip()) for row in labeled_top]
        precision_values.append((sum(labels) / len(labels)) if labels else 0.0)

        hit_items = ordered[: max(1, int(hit_k))]
        labeled_hit = [row for row in hit_items if str(row.get(label_col, "")).strip() in {"0", "1"}]
        hit_values.append(1.0 if any(int(str(row.get(label_col, "")).strip()) == 1 for row in labeled_hit) else 0.0)
        vault_values.append(
            1.0
            if any(
                int(str(row.get(label_col, "")).strip()) == 1 and str(row.get(source_col, "")).strip() == "vault"
                for row in labeled_hit
            )
            else 0.0
        )

    total_queries = len([query for query in grouped if query])
    return {
        "queryCount": total_queries,
        "labeledQueryCount": labeled_queries,
        "precisionAt5": (sum(precision_values) / len(precision_values)) if precision_values else 0.0,
        "hitAt3": (sum(hit_values) / len(hit_values)) if hit_values else 0.0,
        "vaultHitRatio": (sum(vault_values) / len(vault_values)) if vault_values else 0.0,
    }


def build_eval_report(
    csv_path: str | Path,
    *,
    label_col: str,
    source_col: str,
    query_col: str = "query",
    rank_col: str = "rank",
    top_k: int = 5,
    hit_k: int = 3,
    runtime_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = compute_precision_metrics(
        csv_path,
        label_col=label_col,
        source_col=source_col,
        query_col=query_col,
        rank_col=rank_col,
        top_k=top_k,
        hit_k=hit_k,
    )
    runtime = dict(runtime_diagnostics or {})
    degraded = bool(runtime.get("degraded"))
    return {
        "schema": "knowledge-hub.retrieval.eval.report.v1",
        "status": "degraded" if degraded else "ok",
        "dataset": {
            "csvPath": str(Path(csv_path)),
            "queryCol": query_col,
            "rankCol": rank_col,
            "labelCol": label_col,
            "sourceCol": source_col,
            "topK": int(top_k),
            "hitK": int(hit_k),
        },
        "metrics": metrics,
        "runtimeDiagnostics": runtime,
        "warnings": list(runtime.get("warnings") or []),
    }


__all__ = ["build_eval_report", "compute_precision_metrics", "read_eval_queries"]

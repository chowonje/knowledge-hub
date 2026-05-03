#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_ORDER = ["vault", "paper", "web", "all"]
STAGE_A_THRESHOLDS = {
    "overall_top1_hit_rate_min": 0.60,
    "overall_no_result_rate_max": 0.40,
    "overall_route_apply_rate_min": 0.60,
    "source_top1_hit_rate_min": 0.50,
}
STAGE_B_THRESHOLDS = {
    "good_plus_partial_min": 0.70,
    "temporal_wrong_era_max": 0.20,
    "abstention_agreement_min": 0.80,
    "answerable_rate_min": 0.01,
    "non_substantive_top1_rate_max": 0.30,
    "temporal_grounded_rate_min": 0.30,
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            fixed = {}
            for key, value in row.items():
                normalized = "query" if str(key).lstrip("\ufeff") == "query" else str(key)
                fixed[normalized] = str(value or "")
            rows.append(fixed)
        return rows


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _top1_exists(row: dict[str, str]) -> bool:
    return bool(str(row.get("top1_title") or "").strip())


def _is_no_result(row: dict[str, str]) -> bool:
    status = str(row.get("answer_status") or "").strip().lower()
    if status == "no_result":
        return True
    return _as_bool(row.get("no_result"))


def _split_reason_tokens(raw: str) -> list[str]:
    token = str(raw or "").strip()
    if not token:
        return []
    return [part.strip() for part in token.split("|") if part.strip()]


def _distribution(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value:
            counter[value] += 1
    return dict(counter)


def _insufficient_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for token in _split_reason_tokens(row.get("insufficient_reasons") or ""):
            counter[token] += 1
    return dict(counter)


def _stage_a_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    top1_count = sum(1 for row in rows if _top1_exists(row))
    no_result_count = sum(1 for row in rows if _is_no_result(row))
    route_apply_count = sum(1 for row in rows if _as_bool(row.get("memory_route_applied")))
    temporal_apply_count = sum(1 for row in rows if _as_bool(row.get("temporal_route_applied")))

    by_source: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source") or "").strip().lower()].append(row)
    for source in SOURCE_ORDER:
        subset = grouped.get(source, [])
        count = len(subset)
        by_source[source] = {
            "count": count,
            "top1_hit_rate": _rate(sum(1 for row in subset if _top1_exists(row)), count),
            "no_result_rate": _rate(sum(1 for row in subset if _is_no_result(row)), count),
            "route_apply_rate": _rate(sum(1 for row in subset if _as_bool(row.get("memory_route_applied"))), count),
            "temporal_apply_rate": _rate(sum(1 for row in subset if _as_bool(row.get("temporal_route_applied"))), count),
        }

    checks = {
        "overall_top1_hit_rate": _rate(top1_count, total) >= STAGE_A_THRESHOLDS["overall_top1_hit_rate_min"],
        "overall_no_result_rate": _rate(no_result_count, total) <= STAGE_A_THRESHOLDS["overall_no_result_rate_max"],
        "overall_route_apply_rate": _rate(route_apply_count, total) >= STAGE_A_THRESHOLDS["overall_route_apply_rate_min"],
        "vault_top1_hit_rate": by_source["vault"]["top1_hit_rate"] >= STAGE_A_THRESHOLDS["source_top1_hit_rate_min"],
        "web_top1_hit_rate": by_source["web"]["top1_hit_rate"] >= STAGE_A_THRESHOLDS["source_top1_hit_rate_min"],
        "all_top1_hit_rate": by_source["all"]["top1_hit_rate"] >= STAGE_A_THRESHOLDS["source_top1_hit_rate_min"],
    }

    return {
        "metrics": {
            "count": total,
            "top1_hit_rate": _rate(top1_count, total),
            "no_result_rate": _rate(no_result_count, total),
            "route_apply_rate": _rate(route_apply_count, total),
            "temporal_apply_rate": _rate(temporal_apply_count, total),
        },
        "by_source": by_source,
        "prefilter_reason_distribution": _distribution(rows, "memory_prefilter_reason"),
        "insufficient_reason_distribution": _insufficient_distribution(rows),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _stage_b_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    top1_rows = [row for row in rows if _top1_exists(row)]
    answerable_rate = _rate(sum(1 for row in top1_rows if _as_bool(row.get("answerable"))), len(top1_rows))
    non_substantive_top1_rate = _rate(sum(1 for row in top1_rows if not _as_bool(row.get("top1_substantive"))), len(top1_rows))
    temporal_top1_rows = [row for row in top1_rows if _as_bool(row.get("temporal_query"))]
    temporal_grounded_rate = _rate(
        sum(
            1
            for row in temporal_top1_rows
            if str(row.get("temporal_grounded_count") or "").strip() not in {"", "0"}
        ),
        len(temporal_top1_rows),
    )
    judged_rows = [row for row in top1_rows if str(row.get("pred_label") or "").strip()]
    label_counter = Counter(str(row.get("pred_label") or "").strip().lower() for row in judged_rows if str(row.get("pred_label") or "").strip())
    abstain_counter = Counter("1" if _as_bool(row.get("pred_should_abstain")) else "0" for row in judged_rows)
    temporal_rows = [row for row in judged_rows if _as_bool(row.get("temporal_query"))]
    wrong_era_rate = _rate(sum(1 for row in temporal_rows if _as_bool(row.get("pred_wrong_era"))), len(temporal_rows))
    abstention_rows = [
        row
        for row in judged_rows
        if str(row.get("query_type") or "").strip().lower() == "abstention"
        or str(row.get("expected_answer_style") or "").strip().lower() == "abstain"
    ]
    abstention_agreement = _rate(sum(1 for row in abstention_rows if _as_bool(row.get("pred_should_abstain"))), len(abstention_rows))
    good_plus_partial = _rate(label_counter.get("good", 0) + label_counter.get("partial", 0), len(judged_rows))
    pred_columns_present = any("pred_label" in row for row in rows)

    checks = {
        "answerable_rate": answerable_rate >= STAGE_B_THRESHOLDS["answerable_rate_min"] if top1_rows else False,
        "non_substantive_top1_rate": non_substantive_top1_rate <= STAGE_B_THRESHOLDS["non_substantive_top1_rate_max"] if top1_rows else False,
        "temporal_grounded_rate": temporal_grounded_rate >= STAGE_B_THRESHOLDS["temporal_grounded_rate_min"] if temporal_top1_rows else False,
        "good_plus_partial_rate": good_plus_partial >= STAGE_B_THRESHOLDS["good_plus_partial_min"] if judged_rows else False,
        "temporal_wrong_era_rate": wrong_era_rate <= STAGE_B_THRESHOLDS["temporal_wrong_era_max"] if temporal_rows else False,
        "abstention_agreement": abstention_agreement >= STAGE_B_THRESHOLDS["abstention_agreement_min"] if abstention_rows else False,
    }
    raw_checks = (
        checks["answerable_rate"],
        checks["non_substantive_top1_rate"],
        checks["temporal_grounded_rate"],
    )
    judged_checks = (
        checks["good_plus_partial_rate"],
        checks["temporal_wrong_era_rate"],
        checks["abstention_agreement"],
    )
    pass_value = all(raw_checks) and (all(judged_checks) if judged_rows else False)

    return {
        "metrics": {
            "top1_row_count": len(top1_rows),
            "judged_row_count": len(judged_rows),
            "answerable_rate": answerable_rate,
            "non_substantive_top1_rate": non_substantive_top1_rate,
            "temporal_grounded_rate": temporal_grounded_rate,
            "good_plus_partial_rate": good_plus_partial,
            "temporal_wrong_era_rate": wrong_era_rate,
            "abstention_agreement": abstention_agreement,
        },
        "pred_label_distribution": dict(label_counter),
        "pred_should_abstain_distribution": dict(abstain_counter),
        "pred_wrong_era_distribution": {
            "1": sum(1 for row in temporal_rows if _as_bool(row.get("pred_wrong_era"))),
            "0": sum(1 for row in temporal_rows if not _as_bool(row.get("pred_wrong_era"))),
        },
        "checks": checks,
        "pass": pass_value,
        "available": pred_columns_present and bool(judged_rows),
    }


def _delta_vs_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result = {
        "stageA": {},
        "stageB": {},
        "by_source": {},
    }
    for key in ("top1_hit_rate", "no_result_rate", "route_apply_rate", "temporal_apply_rate"):
        cand = float(candidate["stageA"]["metrics"].get(key, 0.0))
        base = float(baseline["stageA"]["metrics"].get(key, 0.0))
        result["stageA"][key] = {
            "candidate": cand,
            "baseline": base,
            "delta_pp": round((cand - base) * 100.0, 3),
        }
    for source in SOURCE_ORDER:
        cand = candidate["stageA"]["by_source"].get(source, {})
        base = baseline["stageA"]["by_source"].get(source, {})
        result["by_source"][source] = {}
        for key in ("top1_hit_rate", "no_result_rate", "route_apply_rate", "temporal_apply_rate"):
            cand_value = float(cand.get(key, 0.0))
            base_value = float(base.get(key, 0.0))
            result["by_source"][source][key] = {
                "candidate": cand_value,
                "baseline": base_value,
                "delta_pp": round((cand_value - base_value) * 100.0, 3),
            }
    for key in ("good_plus_partial_rate", "temporal_wrong_era_rate", "abstention_agreement"):
        cand = float(candidate["stageB"]["metrics"].get(key, 0.0))
        base = float(baseline["stageB"]["metrics"].get(key, 0.0))
        result["stageB"][key] = {
            "candidate": cand,
            "baseline": base,
            "delta_pp": round((cand - base) * 100.0, 3),
        }
    for key in ("answerable_rate", "non_substantive_top1_rate", "temporal_grounded_rate"):
        cand = float(candidate["stageB"]["metrics"].get(key, 0.0))
        base = float(baseline["stageB"]["metrics"].get(key, 0.0))
        result["stageB"][key] = {
            "candidate": cand,
            "baseline": base,
            "delta_pp": round((cand - base) * 100.0, 3),
        }
    return result


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}pp"


def _render_markdown(report: dict[str, Any]) -> str:
    candidate = report["candidate"]
    baseline = report["baseline"]
    delta = report["delta_vs_baseline"]
    lines: list[str] = []
    lines.append("# KnowledgeOS Eval Report v1")
    lines.append("")
    lines.append(f"- candidate: `{report['paths']['candidate']}`")
    lines.append(f"- baseline: `{report['paths']['baseline']}`")
    lines.append("")
    lines.append("## Stage A: Retrieval Health")
    lines.append("")
    lines.append(f"- PASS: `{candidate['stageA']['pass']}`")
    lines.append(f"- overall top1_hit_rate: `{_fmt_pct(candidate['stageA']['metrics']['top1_hit_rate'])}`")
    lines.append(f"- overall no_result_rate: `{_fmt_pct(candidate['stageA']['metrics']['no_result_rate'])}`")
    lines.append(f"- overall route_apply_rate: `{_fmt_pct(candidate['stageA']['metrics']['route_apply_rate'])}`")
    lines.append(f"- overall temporal_apply_rate: `{_fmt_pct(candidate['stageA']['metrics']['temporal_apply_rate'])}`")
    lines.append("")
    lines.append("| source | count | top1_hit_rate | no_result_rate | route_apply_rate | temporal_apply_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for source in SOURCE_ORDER:
        metrics = candidate["stageA"]["by_source"][source]
        lines.append(
            f"| {source} | {metrics['count']} | {_fmt_pct(metrics['top1_hit_rate'])} | {_fmt_pct(metrics['no_result_rate'])} | "
            f"{_fmt_pct(metrics['route_apply_rate'])} | {_fmt_pct(metrics['temporal_apply_rate'])} |"
        )
    lines.append("")
    lines.append(f"- prefilter reasons: `{json.dumps(candidate['stageA']['prefilter_reason_distribution'], ensure_ascii=False)}`")
    lines.append(f"- insufficient reasons: `{json.dumps(candidate['stageA']['insufficient_reason_distribution'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Stage B: Answer Quality")
    lines.append("")
    lines.append(f"- PASS: `{candidate['stageB']['pass']}`")
    lines.append(f"- available: `{candidate['stageB']['available']}`")
    lines.append(f"- judged top1 rows: `{candidate['stageB']['metrics']['judged_row_count']}` / `{candidate['stageB']['metrics']['top1_row_count']}`")
    lines.append(f"- answerable rate: `{_fmt_pct(candidate['stageB']['metrics']['answerable_rate'])}`")
    lines.append(f"- non_substantive top1 rate: `{_fmt_pct(candidate['stageB']['metrics']['non_substantive_top1_rate'])}`")
    lines.append(f"- temporal grounded rate: `{_fmt_pct(candidate['stageB']['metrics']['temporal_grounded_rate'])}`")
    lines.append(f"- good+partial rate: `{_fmt_pct(candidate['stageB']['metrics']['good_plus_partial_rate'])}`")
    lines.append(f"- temporal wrong-era rate: `{_fmt_pct(candidate['stageB']['metrics']['temporal_wrong_era_rate'])}`")
    lines.append(f"- abstention agreement: `{_fmt_pct(candidate['stageB']['metrics']['abstention_agreement'])}`")
    lines.append(f"- pred_label distribution: `{json.dumps(candidate['stageB']['pred_label_distribution'], ensure_ascii=False)}`")
    lines.append(f"- pred_should_abstain distribution: `{json.dumps(candidate['stageB']['pred_should_abstain_distribution'], ensure_ascii=False)}`")
    lines.append(f"- pred_wrong_era distribution: `{json.dumps(candidate['stageB']['pred_wrong_era_distribution'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Delta vs Baseline")
    lines.append("")
    for key, payload in delta["stageA"].items():
        lines.append(f"- Stage A {key}: `{_fmt_pp(payload['delta_pp'])}`")
    for key, payload in delta["stageB"].items():
        lines.append(f"- Stage B {key}: `{_fmt_pp(payload['delta_pp'])}`")
    lines.append("")
    lines.append("| source | top1_hit_rate Δ | no_result_rate Δ | route_apply_rate Δ | temporal_apply_rate Δ |")
    lines.append("|---|---:|---:|---:|---:|")
    for source in SOURCE_ORDER:
        payload = delta["by_source"][source]
        lines.append(
            f"| {source} | {_fmt_pp(payload['top1_hit_rate']['delta_pp'])} | {_fmt_pp(payload['no_result_rate']['delta_pp'])} | "
            f"{_fmt_pp(payload['route_apply_rate']['delta_pp'])} | {_fmt_pp(payload['temporal_apply_rate']['delta_pp'])} |"
        )
    lines.append("")
    lines.append("## Gate Thresholds")
    lines.append("")
    lines.append(f"- Stage A top1_hit_rate >= `{STAGE_A_THRESHOLDS['overall_top1_hit_rate_min']:.2f}`")
    lines.append(f"- Stage A no_result_rate <= `{STAGE_A_THRESHOLDS['overall_no_result_rate_max']:.2f}`")
    lines.append(f"- Stage A route_apply_rate >= `{STAGE_A_THRESHOLDS['overall_route_apply_rate_min']:.2f}`")
    lines.append(f"- Stage A source top1_hit_rate (vault/web/all) >= `{STAGE_A_THRESHOLDS['source_top1_hit_rate_min']:.2f}`")
    lines.append(f"- Stage B good+partial >= `{STAGE_B_THRESHOLDS['good_plus_partial_min']:.2f}`")
    lines.append(f"- Stage B temporal wrong-era <= `{STAGE_B_THRESHOLDS['temporal_wrong_era_max']:.2f}`")
    lines.append(f"- Stage B abstention agreement >= `{STAGE_B_THRESHOLDS['abstention_agreement_min']:.2f}`")
    lines.append(f"- Stage B answerable_rate >= `{STAGE_B_THRESHOLDS['answerable_rate_min']:.2f}`")
    lines.append(f"- Stage B non_substantive_top1_rate <= `{STAGE_B_THRESHOLDS['non_substantive_top1_rate_max']:.2f}`")
    lines.append(f"- Stage B temporal_grounded_rate >= `{STAGE_B_THRESHOLDS['temporal_grounded_rate_min']:.2f}`")
    return "\n".join(lines) + "\n"


def build_report(candidate_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]) -> dict[str, Any]:
    candidate = {
        "stageA": _stage_a_metrics(candidate_rows),
        "stageB": _stage_b_metrics(candidate_rows),
    }
    baseline = {
        "stageA": _stage_a_metrics(baseline_rows),
        "stageB": _stage_b_metrics(baseline_rows),
    }
    return {
        "schema": "knowledgeos.eval.report.v1",
        "candidate": candidate,
        "baseline": baseline,
        "delta_vs_baseline": _delta_vs_baseline(candidate, baseline),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage A/B eval report for KnowledgeOS candidate vs baseline CSVs.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()

    candidate_path = Path(args.candidate).expanduser()
    baseline_path = Path(args.baseline).expanduser()
    candidate_rows = _read_rows(candidate_path)
    baseline_rows = _read_rows(baseline_path)
    report = build_report(candidate_rows, baseline_rows)
    report["paths"] = {
        "candidate": str(candidate_path),
        "baseline": str(baseline_path),
    }

    reports_dir = candidate_path.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_stem = f"{candidate_path.stem}__vs__{baseline_path.stem}"
    json_path = reports_dir / f"{report_stem}.json"
    md_path = reports_dir / f"{report_stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    summary = {
        "status": "ok",
        "json": str(json_path),
        "markdown": str(md_path),
        "stageA_pass": bool(report["candidate"]["stageA"]["pass"]),
        "stageB_pass": bool(report["candidate"]["stageB"]["pass"]),
        "top1_hit_rate": report["candidate"]["stageA"]["metrics"]["top1_hit_rate"],
        "no_result_rate": report["candidate"]["stageA"]["metrics"]["no_result_rate"],
        "answerable_rate": report["candidate"]["stageB"]["metrics"]["answerable_rate"],
        "non_substantive_top1_rate": report["candidate"]["stageB"]["metrics"]["non_substantive_top1_rate"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from typing import Any


def _alert(
    *,
    code: str,
    severity: str,
    scope: str,
    summary: str,
    metric: str,
    observed: Any,
    threshold: str,
) -> dict[str, Any]:
    return {
        "code": str(code),
        "severity": str(severity),
        "scope": str(scope),
        "summary": str(summary),
        "metric": str(metric),
        "observed": observed,
        "threshold": str(threshold),
    }


def _action(
    *,
    action_type: str,
    scope: str,
    summary: str,
    reason_codes: list[str],
    command: str,
    args: list[str],
) -> dict[str, Any]:
    return {
        "actionType": str(action_type),
        "scope": str(scope),
        "summary": str(summary),
        "reasonCodes": [str(item) for item in reason_codes if str(item).strip()],
        "command": str(command),
        "args": [str(item) for item in args if str(item).strip()],
    }


def evaluate_ko_note_report_alerts(
    *,
    run_id: str,
    run_payload: dict[str, Any],
    apply_backlog_count: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alerts: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    review_queue = dict((run_payload.get("reviewQueue") or {}).get("combined") or {})
    remediation = dict(run_payload.get("remediation") or {})
    combined_quality = dict((run_payload.get("quality") or {}).get("combined") or {})

    review_total = int(review_queue.get("total") or 0)
    remediation_failures = int(remediation.get("failed") or 0) + int(remediation.get("regressed") or 0)
    remediation_full = int(remediation.get("recommendedFull") or 0)
    rejected_total = int(combined_quality.get("reject") or 0)

    if review_total > 0:
        alerts.append(
            _alert(
                code="ko_note_review_queue_pending",
                severity="warning",
                scope="ko_note",
                summary=f"review queue에 {review_total}개 항목이 남아 있습니다.",
                metric="reviewQueue.combined.total",
                observed=review_total,
                threshold="> 0",
            )
        )
        actions.append(
            _action(
                action_type="inspect_review_queue",
                scope="ko_note",
                summary="review queue 항목을 먼저 확인하세요.",
                reason_codes=["ko_note_review_queue_pending"],
                command="khub",
                args=["labs", "crawl", "ko-note-review-list", "--run-id", str(run_id)],
            )
        )

    if remediation_failures > 0:
        alerts.append(
            _alert(
                code="ko_note_remediation_failures",
                severity="critical",
                scope="ko_note",
                summary=f"remediation 실패/회귀 항목이 {remediation_failures}개 있습니다.",
                metric="remediation.failed+regressed",
                observed=remediation_failures,
                threshold="> 0",
            )
        )
        actions.append(
            _action(
                action_type="remediate_full",
                scope="ko_note",
                summary="회귀/실패 항목은 full remediation 경로를 점검하세요.",
                reason_codes=["ko_note_remediation_failures"],
                command="khub",
                args=["labs", "crawl", "ko-note-remediate", "--run-id", str(run_id), "--strategy", "full"],
            )
        )

    if remediation_full > 0:
        alerts.append(
            _alert(
                code="ko_note_full_remediation_recommended",
                severity="warning",
                scope="ko_note",
                summary=f"section remediation으로 해결되지 않은 항목이 {remediation_full}개 있습니다.",
                metric="remediation.recommendedFull",
                observed=remediation_full,
                threshold="> 0",
            )
        )
        actions.append(
            _action(
                action_type="remediate_full",
                scope="ko_note",
                summary="full remediation 권장 항목을 재보강하세요.",
                reason_codes=["ko_note_full_remediation_recommended"],
                command="khub",
                args=["labs", "crawl", "ko-note-remediate", "--run-id", str(run_id), "--strategy", "full"],
            )
        )

    if rejected_total > 0:
        alerts.append(
            _alert(
                code="ko_note_rejected_quality_items",
                severity="warning",
                scope="ko_note",
                summary=f"품질 reject 상태 항목이 {rejected_total}개 있습니다.",
                metric="quality.combined.reject",
                observed=rejected_total,
                threshold="> 0",
            )
        )

    if review_total > 0 and remediation_failures == 0 and remediation_full == 0:
        actions.append(
            _action(
                action_type="remediate_section",
                scope="ko_note",
                summary="review queue의 약한 노트를 section remediation으로 먼저 보강하세요.",
                reason_codes=["ko_note_review_queue_pending"],
                command="khub",
                args=["labs", "crawl", "ko-note-remediate", "--run-id", str(run_id), "--strategy", "section"],
            )
        )

    if apply_backlog_count > 0:
        actions.append(
            _action(
                action_type="apply_backlog",
                scope="ko_note",
                summary=f"적용 가능한 ko-note가 {apply_backlog_count}개 남아 있습니다.",
                reason_codes=["ko_note_apply_backlog"],
                command="khub",
                args=["crawl", "ko-note-apply", "--run-id", str(run_id)],
            )
        )

    return _dedupe_entries(alerts), _dedupe_entries(actions)


def evaluate_rag_report_alerts(
    *,
    days: int,
    limit: int,
    counts: dict[str, Any],
    verification: dict[str, Any],
    rates: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alerts: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    total = int(counts.get("total") or 0)
    needs_caution_rate = float(rates.get("needsCautionRate") or 0.0)
    unsupported_rate = float(rates.get("unsupportedClaimRate") or 0.0)
    fallback_rate = float(rates.get("conservativeFallbackRate") or 0.0)
    failed_or_skipped = int(verification.get("failed") or 0) + int(verification.get("skipped") or 0)

    report_args = ["rag-report", "--days", str(int(days)), "--limit", str(int(limit)), "--json"]

    if total >= 10 and needs_caution_rate >= 0.20:
        alerts.append(
            _alert(
                code="rag_high_caution_rate",
                severity="warning",
                scope="rag",
                summary=f"최근 답변의 caution 비율이 {needs_caution_rate:.0%}입니다.",
                metric="rates.needsCautionRate",
                observed=round(needs_caution_rate, 4),
                threshold=">= 0.20 with total >= 10",
            )
        )
        actions.append(
            _action(
                action_type="inspect_rag_samples",
                scope="rag",
                summary="caution 비율이 높습니다. 최근 샘플과 retrieval evidence를 점검하세요.",
                reason_codes=["rag_high_caution_rate"],
                command="khub",
                args=report_args,
            )
        )

    if total >= 10 and unsupported_rate >= 0.10:
        alerts.append(
            _alert(
                code="rag_high_unsupported_rate",
                severity="critical",
                scope="rag",
                summary=f"최근 답변의 unsupported claim 비율이 {unsupported_rate:.0%}입니다.",
                metric="rates.unsupportedClaimRate",
                observed=round(unsupported_rate, 4),
                threshold=">= 0.10 with total >= 10",
            )
        )
        actions.append(
            _action(
                action_type="inspect_rag_samples",
                scope="rag",
                summary="unsupported claim이 많습니다. retrieval 근거와 answer rewrite 품질을 점검하세요.",
                reason_codes=["rag_high_unsupported_rate"],
                command="khub",
                args=report_args,
            )
        )

    if failed_or_skipped > 0:
        alerts.append(
            _alert(
                code="rag_verification_failed_or_skipped",
                severity="warning",
                scope="rag",
                summary=f"verification failed/skipped 사례가 {failed_or_skipped}건 있습니다.",
                metric="verification.failed+skipped",
                observed=failed_or_skipped,
                threshold="> 0",
            )
        )
        actions.append(
            _action(
                action_type="inspect_verification_routes",
                scope="rag",
                summary="verification route와 policy/config 경로를 점검하세요.",
                reason_codes=["rag_verification_failed_or_skipped"],
                command="khub",
                args=["config", "list"],
            )
        )

    if total >= 10 and fallback_rate >= 0.05:
        alerts.append(
            _alert(
                code="rag_high_conservative_fallback_rate",
                severity="warning",
                scope="rag",
                summary=f"conservative fallback 비율이 {fallback_rate:.0%}입니다.",
                metric="rates.conservativeFallbackRate",
                observed=round(fallback_rate, 4),
                threshold=">= 0.05 with total >= 10",
            )
        )
        actions.append(
            _action(
                action_type="review_answer_routes",
                scope="rag",
                summary="answer prompt와 verification-rewrite 경로를 점검하세요.",
                reason_codes=["rag_high_conservative_fallback_rate"],
                command="khub",
                args=report_args,
            )
        )

    return _dedupe_entries(alerts), _dedupe_entries(actions)


def evaluate_paper_source_report_alerts(
    *,
    counts: dict[str, Any],
    items: list[dict[str, Any]],
    max_actions: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alerts: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    repairable_pending = int(counts.get("repairablePending") or 0)
    blocked_manual = int(counts.get("blockedManual") or 0)
    blocked_missing_canonical = int(counts.get("blockedMissingCanonical") or 0)

    if repairable_pending > 0:
        alerts.append(
            _alert(
                code="paper_source_repair_pending",
                severity="warning",
                scope="paper",
                summary=f"canonical relink가 필요한 paper source가 {repairable_pending}건 있습니다.",
                metric="counts.repairablePending",
                observed=repairable_pending,
                threshold="> 0",
            )
        )
        repair_items = [dict(item) for item in items if bool(item.get("needsRepair"))]
        for item in repair_items[: max(1, int(max_actions or 5))]:
            paper_id = str(item.get("paperId") or "").strip()
            if not paper_id:
                continue
            title = str(item.get("title") or paper_id).strip()
            actions.append(
                _action(
                    action_type="repair_paper_source",
                    scope="paper",
                    summary=f"paper source repair: {title}",
                    reason_codes=["paper_source_repair_pending"],
                    command="khub",
                    args=["paper", "repair-source", "--paper-id", paper_id],
                )
                | {"paperId": paper_id}
            )

    if blocked_manual > 0:
        alerts.append(
            _alert(
                code="paper_source_manual_fix_needed",
                severity="warning",
                scope="paper",
                summary=f"수동 source 복구가 필요한 paper가 {blocked_manual}건 있습니다.",
                metric="counts.blockedManual",
                observed=blocked_manual,
                threshold="> 0",
            )
        )

    if blocked_missing_canonical > 0:
        alerts.append(
            _alert(
                code="paper_source_missing_canonical",
                severity="critical",
                scope="paper",
                summary=f"canonical source가 없어 자동 repair를 못 하는 paper가 {blocked_missing_canonical}건 있습니다.",
                metric="counts.blockedMissingCanonical",
                observed=blocked_missing_canonical,
                threshold="> 0",
            )
        )

    return _dedupe_entries(alerts), _dedupe_entries(actions)


def evaluate_paper_report_alerts(
    *,
    counts: dict[str, Any],
    items: list[dict[str, Any]],
    document_memory_parser: str = "raw",
    rebuild: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alerts: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    repair_eligible = [dict(item) for item in items if bool(item.get("repairEligible"))]
    manual_fix_required = int(counts.get("manualFixRequired") or 0)
    canonical_missing = int(counts.get("canonicalMissing") or 0)
    parser_token = str(document_memory_parser or "").strip().lower() or "raw"

    if repair_eligible:
        alerts.append(
            _alert(
                code="paper_source_repair_pending",
                severity="warning",
                scope="paper",
                summary=f"deterministic paper source repair 후보가 {len(repair_eligible)}개 있습니다.",
                metric="counts.repairEligible",
                observed=len(repair_eligible),
                threshold="> 0",
            )
        )
        for item in repair_eligible:
            paper_id = str(item.get("paperId") or "").strip()
            if not paper_id:
                continue
            args = ["paper", "repair-source", "--paper-id", paper_id]
            if parser_token != "raw":
                args.extend(["--document-memory-parser", parser_token])
            if not rebuild:
                args.append("--no-rebuild")
            actions.append(
                _action(
                    action_type="repair_paper_source",
                    scope="paper",
                    summary=f"repair paper source for {str(item.get('title') or paper_id).strip()}",
                    reason_codes=["paper_source_repair_pending"],
                    command="khub",
                    args=args,
                )
            )

    if manual_fix_required > 0:
        alerts.append(
            _alert(
                code="paper_source_manual_fix_required",
                severity="warning",
                scope="paper",
                summary=f"manual source fix가 필요한 paper가 {manual_fix_required}개 있습니다.",
                metric="counts.manualFixRequired",
                observed=manual_fix_required,
                threshold="> 0",
            )
        )

    if canonical_missing > 0:
        alerts.append(
            _alert(
                code="paper_source_canonical_missing",
                severity="critical",
                scope="paper",
                summary=f"canonical source row가 없어 자동 repair를 못 하는 paper가 {canonical_missing}개 있습니다.",
                metric="counts.canonicalMissing",
                observed=canonical_missing,
                threshold="> 0",
            )
        )

    return _dedupe_entries(alerts), _dedupe_entries(actions)


def _dedupe_entries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for item in items:
        code = str(item.get("code") or item.get("actionType") or "")
        scope = str(item.get("scope") or "")
        args = tuple(str(arg) for arg in item.get("args", []))
        key = (code, scope, args)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped

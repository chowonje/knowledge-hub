"""Deterministic synthesis helpers above normalized claims.

The first KnowledgeOS vNext synthesis layer intentionally stays bounded and
inspectable. It composes existing normalized claim comparisons into compact
reports without changing the default runtime answer path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from knowledge_hub.knowledge.claim_normalization import (
    CLAIM_NORMALIZATION_VERSION,
    ClaimComparisonService,
    ClaimNormalizationService,
)
from knowledge_hub.knowledge.semantic_units import evidence_links_from_claim


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _unique_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _clean_text(value)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
    return out


def _comparison_label(kind: str, reason: str) -> str:
    if kind == "aligned":
        return "aligned claims"
    if kind == "conflict":
        return "conflict candidate"
    if kind == "incomparable":
        return "incomparable claims"
    return _clean_text(reason) or "comparison"


def _group_summary(kind: str, group: dict[str, Any]) -> str:
    key = dict(group.get("comparisonKey") or {})
    task = _clean_text(key.get("task"))
    metric = _clean_text(key.get("metric"))
    claims = list(group.get("claims") or [])
    paper_titles = _unique_strings([str(item.get("paperTitle") or "") for item in claims])
    reason = _clean_text(group.get("reason"))
    if kind == "aligned":
        body = f"{len(claims)} claims align on {task}/{metric}"
    elif kind == "conflict":
        body = f"{len(claims)} claims disagree on {task}/{metric}"
    else:
        body = f"{len(claims)} claims are not directly comparable for {task}/{metric}"
    if paper_titles:
        body = f"{body} across {', '.join(paper_titles[:3])}"
    if reason:
        body = f"{body} ({reason})"
    return body


def _compact_claim(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item.get("normalized") or {})
    return {
        "claimId": _clean_text(item.get("claimId")),
        "paperId": _clean_text(item.get("paperId")),
        "paperTitle": _clean_text(item.get("paperTitle")),
        "claimText": _clean_text(item.get("claimText")),
        "predicate": _clean_text(item.get("predicate")),
        "confidence": float(item.get("confidence") or 0.0),
        "evidenceSummary": _clean_text(item.get("evidenceSummary")),
        "normalizationStatus": _clean_text(item.get("normalizationStatus") or "failed"),
        "evidenceLinks": list(item.get("evidenceLinks") or []),
        "normalized": {
            "task": _clean_text(normalized.get("task")),
            "dataset": _clean_text(normalized.get("dataset")),
            "metric": _clean_text(normalized.get("metric")),
            "comparator": _clean_text(normalized.get("comparator")),
            "resultDirection": _clean_text(normalized.get("resultDirection")),
            "conditionText": _clean_text(normalized.get("conditionText")),
            "scopeText": _clean_text(normalized.get("scopeText")),
            "limitationText": _clean_text(normalized.get("limitationText")),
            "evidenceStrength": _clean_text(normalized.get("evidenceStrength")),
        },
    }


def _build_report_items(kind: str, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        key = dict(group.get("comparisonKey") or {})
        claims = [_compact_claim(item) for item in list(group.get("claims") or [])]
        items.append(
            {
                "reportId": f"{kind}:{index}",
                "kind": kind,
                "label": _comparison_label(kind, str(group.get("reason") or "")),
                "task": _clean_text(key.get("task")),
                "metric": _clean_text(key.get("metric")),
                "reason": _clean_text(group.get("reason")),
                "summary": _group_summary(kind, group),
                "claims": claims,
            }
        )
    return items


def _limitation_candidates(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for group in groups:
        for claim in list(group.get("claims") or []):
            normalized = dict(claim.get("normalized") or {})
            limitation = _clean_text(normalized.get("limitationText"))
            if not limitation:
                condition = _clean_text(normalized.get("conditionText"))
                scope = _clean_text(normalized.get("scopeText"))
                if condition and any(token in condition.casefold() for token in ("low recall", "domain shift", "top-k")):
                    limitation = condition
                elif scope and any(token in scope.casefold() for token in ("low-recall", "domain shift")):
                    limitation = scope
            if not limitation:
                continue
            bucket = buckets.setdefault(
                limitation.casefold(),
                {
                    "text": limitation,
                    "claimIds": [],
                    "paperIds": [],
                    "paperTitles": [],
                    "evidence": [],
                },
            )
            claim_id = _clean_text(claim.get("claimId"))
            paper_id = _clean_text(claim.get("paperId"))
            paper_title = _clean_text(claim.get("paperTitle"))
            evidence_summary = _clean_text(claim.get("evidenceSummary"))
            if claim_id:
                bucket["claimIds"].append(claim_id)
            if paper_id:
                bucket["paperIds"].append(paper_id)
            if paper_title:
                bucket["paperTitles"].append(paper_title)
            if evidence_summary:
                bucket["evidence"].append(
                    {
                        "claimId": claim_id,
                        "paperId": paper_id,
                        "paperTitle": paper_title,
                        "evidenceSummary": evidence_summary,
                        "evidenceLinks": list(claim.get("evidenceLinks") or []),
                    }
                )
    items: list[dict[str, Any]] = []
    for index, bucket in enumerate(sorted(buckets.values(), key=lambda item: (-len(item["claimIds"]), item["text"])), start=1):
        paper_titles = _unique_strings(list(bucket.get("paperTitles") or []))
        items.append(
            {
                "summaryId": f"limitation:{index}",
                "limitation": bucket["text"],
                "count": len(_unique_strings(list(bucket.get("claimIds") or []))),
                "paperIds": _unique_strings(list(bucket.get("paperIds") or [])),
                "paperTitles": paper_titles,
                "summary": (
                    f"{bucket['text']} appears across {len(paper_titles) or len(_unique_strings(list(bucket.get('paperIds') or [])))} "
                    "paper contexts"
                ),
                "evidence": list(bucket.get("evidence") or [])[:5],
            }
        )
    return items


def _selected_claim_items(
    db: Any,
    config: Any,
    *,
    claim_ids: list[str] | None,
    paper_ids: list[str] | None,
    task: str,
    dataset: str,
    metric: str,
    limit: int,
) -> list[dict[str, Any]]:
    normalizer = ClaimNormalizationService(db, config)
    selected = normalizer._candidate_claims(
        claim_ids=claim_ids,
        paper_ids=paper_ids,
        limit=limit,
    )
    items: list[dict[str, Any]] = []
    for claim in selected:
        claim_id = _clean_text(claim.get("claim_id"))
        normalization = db.get_claim_normalization(claim_id, normalization_version=CLAIM_NORMALIZATION_VERSION)
        payload = (
            normalizer.normalize_claim(claim, persist=True, allow_external=False, llm_mode="fallback-only")
            if not normalization
            else {
                "claimId": claim_id,
                "paperId": _clean_text((normalization.get("normalized_payload") or {}).get("paper_id")),
                "paperTitle": _clean_text((normalization.get("normalized_payload") or {}).get("paper_title")),
                "status": _clean_text(normalization.get("status") or "failed"),
                "task": _clean_text(normalization.get("task")),
                "dataset": _clean_text(normalization.get("dataset")),
                "metric": _clean_text(normalization.get("metric")),
                "comparator": _clean_text(normalization.get("comparator")),
                "resultDirection": _clean_text(normalization.get("result_direction")),
                "conditionText": _clean_text(normalization.get("condition_text")),
                "scopeText": _clean_text(normalization.get("scope_text")),
                "limitationText": _clean_text(normalization.get("limitation_text")),
                "evidenceStrength": _clean_text(normalization.get("evidence_strength")),
                "evidenceSummary": _clean_text((normalization.get("normalized_payload") or {}).get("evidence_summary")),
            }
        )
        if task and _clean_text(payload.get("task")) != _clean_text(task):
            continue
        if dataset and _clean_text(payload.get("dataset")) != _clean_text(dataset):
            continue
        if metric and _clean_text(payload.get("metric")) != _clean_text(metric):
            continue
        items.append(
            {
                "claimId": claim_id,
                "paperId": _clean_text(payload.get("paperId")),
                "paperTitle": _clean_text(payload.get("paperTitle")),
                "claimText": _clean_text(claim.get("claim_text")),
                "predicate": _clean_text(claim.get("predicate")),
                "confidence": float(claim.get("confidence") or 0.0),
                "evidenceSummary": _clean_text(payload.get("evidenceSummary")),
                "normalizationStatus": _clean_text(payload.get("status")),
                "evidenceLinks": evidence_links_from_claim(
                    claim,
                    paper_id=_clean_text(payload.get("paperId")),
                    paper_title=_clean_text(payload.get("paperTitle")),
                ),
                "normalized": {
                    "task": _clean_text(payload.get("task")),
                    "dataset": _clean_text(payload.get("dataset")),
                    "metric": _clean_text(payload.get("metric")),
                    "comparator": _clean_text(payload.get("comparator")),
                    "resultDirection": _clean_text(payload.get("resultDirection")),
                    "conditionText": _clean_text(payload.get("conditionText")),
                    "scopeText": _clean_text(payload.get("scopeText")),
                    "limitationText": _clean_text(payload.get("limitationText")),
                    "evidenceStrength": _clean_text(payload.get("evidenceStrength")),
                },
            }
        )
    return items


def _conflict_explanations(conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, group in enumerate(conflicts, start=1):
        key = dict(group.get("comparisonKey") or {})
        claims = [_compact_claim(item) for item in list(group.get("claims") or [])]
        datasets = _unique_strings([str(item.get("normalized", {}).get("dataset") or "") for item in claims])
        comparators = _unique_strings([str(item.get("normalized", {}).get("comparator") or "") for item in claims])
        conditions = _unique_strings([str(item.get("normalized", {}).get("conditionText") or "") for item in claims])
        scopes = _unique_strings([str(item.get("normalized", {}).get("scopeText") or "") for item in claims])
        summary = (
            f"Claims disagree on {_clean_text(key.get('task'))}/{_clean_text(key.get('metric'))} "
            "under otherwise comparable axes."
        )
        items.append(
            {
                "conflictId": f"conflict:{index}",
                "task": _clean_text(key.get("task")),
                "metric": _clean_text(key.get("metric")),
                "severity": _clean_text(group.get("severity") or "unknown"),
                "reason": _clean_text(group.get("reason") or "direction_conflict"),
                "summary": summary,
                "axes": {
                    "datasets": datasets,
                    "comparators": comparators,
                    "conditions": conditions,
                    "scopes": scopes,
                },
                "claims": claims,
            }
        )
    return items


@dataclass(slots=True)
class ClaimSynthesisService:
    db: Any
    config: Any

    def synthesize(
        self,
        *,
        claim_ids: list[str] | None = None,
        paper_ids: list[str] | None = None,
        task: str = "",
        dataset: str = "",
        metric: str = "",
        limit: int = 200,
    ) -> dict[str, Any]:
        compare_result = ClaimComparisonService(self.db, self.config).compare(
            claim_ids=claim_ids,
            paper_ids=paper_ids,
            task=task,
            dataset=dataset,
            metric=metric,
            limit=limit,
        )
        aligned_groups = list(compare_result.get("alignedGroups") or [])
        conflict_candidates = list(compare_result.get("conflictCandidates") or [])
        incomparable_groups = list(compare_result.get("incomparableGroups") or [])
        selected_items = _selected_claim_items(
            self.db,
            self.config,
            claim_ids=claim_ids,
            paper_ids=paper_ids,
            task=task,
            dataset=dataset,
            metric=metric,
            limit=limit,
        )
        report_items = (
            _build_report_items("aligned", aligned_groups)
            + _build_report_items("conflict", conflict_candidates)
            + _build_report_items("incomparable", incomparable_groups)
        )
        limitation_summary = _limitation_candidates(
            [*aligned_groups, *conflict_candidates, *incomparable_groups, {"claims": selected_items}]
        )
        conflict_explanations = _conflict_explanations(conflict_candidates)
        diagnostics = {
            "selectedCount": int(compare_result.get("selectedCount") or 0),
            "alignedGroups": len(aligned_groups),
            "conflictCandidates": len(conflict_candidates),
            "incomparableGroups": len(incomparable_groups),
            "skippedClaims": len(compare_result.get("skippedClaims") or []),
            "normalizationVersion": CLAIM_NORMALIZATION_VERSION,
        }
        return {
            "version": CLAIM_NORMALIZATION_VERSION,
            "comparisonReport": report_items,
            "commonLimitationSummary": limitation_summary,
            "conflictExplanations": conflict_explanations,
            "diagnostics": diagnostics,
            "compareResult": {
                "alignedGroups": aligned_groups,
                "conflictCandidates": conflict_candidates,
                "incomparableGroups": incomparable_groups,
                "skippedClaims": list(compare_result.get("skippedClaims") or []),
            },
        }


__all__ = ["ClaimSynthesisService"]

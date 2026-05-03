from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _section_payload(
    payload: dict[str, Any] | None,
    section_key: str,
    *,
    known_keys: set[str],
) -> dict[str, Any]:
    data = dict(payload or {})
    if isinstance(data.get(section_key), dict):
        return dict(data.get(section_key) or {})
    if any(key in data for key in known_keys):
        return data
    return {}


def _split_known(data: dict[str, Any], known_keys: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    known: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for key, value in dict(data or {}).items():
        if key in known_keys:
            known[key] = value
        else:
            extras[key] = value
    return known, extras


@dataclass
class KoNoteQuality:
    score: int = 0
    max_score: int = 1
    flag: str = "unscored"
    missing_sections: list[str] = field(default_factory=list)
    banned_phrase_hits: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    scored_at: str = ""
    version: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "KoNoteQuality":
        known_keys = {
            "score",
            "max_score",
            "flag",
            "missing_sections",
            "banned_phrase_hits",
            "checks",
            "scored_at",
            "version",
        }
        raw = _section_payload(payload, "quality", known_keys=known_keys)
        known, extras = _split_known(raw, known_keys)
        return cls(
            score=int(known.get("score") or 0),
            max_score=max(1, int(known.get("max_score") or 1)),
            flag=str(known.get("flag") or "unscored").strip() or "unscored",
            missing_sections=[str(item).strip() for item in (known.get("missing_sections") or []) if str(item).strip()],
            banned_phrase_hits=[str(item).strip() for item in (known.get("banned_phrase_hits") or []) if str(item).strip()],
            checks=dict(known.get("checks") or {}),
            scored_at=str(known.get("scored_at") or ""),
            version=str(known.get("version") or ""),
            extras=extras,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **dict(self.extras),
            "score": int(self.score),
            "max_score": max(1, int(self.max_score)),
            "flag": str(self.flag or "unscored"),
            "missing_sections": list(self.missing_sections),
            "banned_phrase_hits": list(self.banned_phrase_hits),
            "checks": dict(self.checks),
            "scored_at": str(self.scored_at or ""),
            "version": str(self.version or ""),
        }


@dataclass
class KoNoteReviewDecision:
    status: str = ""
    note: str = ""
    reviewer: str = ""
    reviewed_at: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "KoNoteReviewDecision":
        known_keys = {"status", "note", "reviewer", "reviewedAt"}
        raw = _section_payload(payload, "decision", known_keys=known_keys)
        known, extras = _split_known(raw, known_keys)
        return cls(
            status=str(known.get("status") or ""),
            note=str(known.get("note") or ""),
            reviewer=str(known.get("reviewer") or ""),
            reviewed_at=str(known.get("reviewedAt") or ""),
            extras=extras,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **dict(self.extras),
            "status": str(self.status or ""),
            "note": str(self.note or ""),
            "reviewer": str(self.reviewer or ""),
            "reviewedAt": str(self.reviewed_at or ""),
        }

    def is_approved(self) -> bool:
        return str(self.status or "") == "approved"

    def is_rejected(self) -> bool:
        return str(self.status or "") == "rejected"

    def is_empty(self) -> bool:
        return not any(
            (
                str(self.status or "").strip(),
                str(self.note or "").strip(),
                str(self.reviewer or "").strip(),
                str(self.reviewed_at or "").strip(),
            )
        )


@dataclass
class KoNoteRemediation:
    attempt_count: int = 0
    last_attempt_at: str = ""
    last_attempt_status: str = ""
    last_attempt_warnings: list[str] = field(default_factory=list)
    last_attempt_quality_flag: str = ""
    last_attempt_score: float = 0.0
    last_improved: bool = False
    last_run_id: str = ""
    strategy: str = ""
    target_sections: list[str] = field(default_factory=list)
    patched_sections: list[str] = field(default_factory=list)
    preserved_sections_count: int = 0
    last_patched_line_count: int = 0
    last_preserved_line_count: int = 0
    section_no_improvement_count: int = 0
    recommended_strategy: str = ""
    field_diagnostics: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "KoNoteRemediation":
        known_keys = {
            "attemptCount",
            "lastAttemptAt",
            "lastAttemptStatus",
            "lastAttemptWarnings",
            "lastAttemptQualityFlag",
            "lastAttemptScore",
            "lastImproved",
            "lastRunId",
            "strategy",
            "targetSections",
            "patchedSections",
            "preservedSectionsCount",
            "lastPatchedLineCount",
            "lastPreservedLineCount",
            "sectionNoImprovementCount",
            "recommendedStrategy",
            "fieldDiagnostics",
        }
        raw = _section_payload(payload, "remediation", known_keys=known_keys)
        known, extras = _split_known(raw, known_keys)
        return cls(
            attempt_count=int(known.get("attemptCount") or 0),
            last_attempt_at=str(known.get("lastAttemptAt") or ""),
            last_attempt_status=str(known.get("lastAttemptStatus") or ""),
            last_attempt_warnings=[str(item).strip() for item in (known.get("lastAttemptWarnings") or []) if str(item).strip()],
            last_attempt_quality_flag=str(known.get("lastAttemptQualityFlag") or ""),
            last_attempt_score=float(known.get("lastAttemptScore") or 0.0),
            last_improved=bool(known.get("lastImproved")),
            last_run_id=str(known.get("lastRunId") or ""),
            strategy=str(known.get("strategy") or ""),
            target_sections=[str(item).strip() for item in (known.get("targetSections") or []) if str(item).strip()],
            patched_sections=[str(item).strip() for item in (known.get("patchedSections") or []) if str(item).strip()],
            preserved_sections_count=int(known.get("preservedSectionsCount") or 0),
            last_patched_line_count=int(known.get("lastPatchedLineCount") or 0),
            last_preserved_line_count=int(known.get("lastPreservedLineCount") or 0),
            section_no_improvement_count=int(known.get("sectionNoImprovementCount") or 0),
            recommended_strategy=str(known.get("recommendedStrategy") or ""),
            field_diagnostics=dict(known.get("fieldDiagnostics") or {}),
            extras=extras,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **dict(self.extras),
            "attemptCount": int(self.attempt_count),
            "lastAttemptAt": str(self.last_attempt_at or ""),
            "lastAttemptStatus": str(self.last_attempt_status or ""),
            "lastAttemptWarnings": list(self.last_attempt_warnings),
            "lastAttemptQualityFlag": str(self.last_attempt_quality_flag or ""),
            "lastAttemptScore": float(self.last_attempt_score or 0.0),
            "lastImproved": bool(self.last_improved),
            "lastRunId": str(self.last_run_id or ""),
            "strategy": str(self.strategy or ""),
            "targetSections": list(self.target_sections),
            "patchedSections": list(self.patched_sections),
            "preservedSectionsCount": int(self.preserved_sections_count),
            "lastPatchedLineCount": int(self.last_patched_line_count),
            "lastPreservedLineCount": int(self.last_preserved_line_count),
            "sectionNoImprovementCount": int(self.section_no_improvement_count),
            "recommendedStrategy": str(self.recommended_strategy or ""),
            "fieldDiagnostics": dict(self.field_diagnostics),
        }

    def needs_full_strategy(self) -> bool:
        return str(self.recommended_strategy or "") == "full"

    def has_active_failure(self) -> bool:
        return str(self.last_attempt_status or "") in {"failed", "regressed"}


@dataclass
class KoNoteReview:
    queue: bool = False
    reasons: list[str] = field(default_factory=list)
    patch_hints: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    decision: KoNoteReviewDecision | None = None
    remediation: KoNoteRemediation = field(default_factory=KoNoteRemediation)
    generated_at: str = ""
    version: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "KoNoteReview":
        known_keys = {
            "queue",
            "reasons",
            "patchHints",
            "suggestedActions",
            "decision",
            "remediation",
            "generatedAt",
            "version",
        }
        raw = _section_payload(payload, "review", known_keys=known_keys)
        known, extras = _split_known(raw, known_keys)
        decision_payload = known.get("decision")
        remediation_payload = known.get("remediation")
        return cls(
            queue=bool(known.get("queue")),
            reasons=[str(item).strip() for item in (known.get("reasons") or []) if str(item).strip()],
            patch_hints=[str(item).strip() for item in (known.get("patchHints") or []) if str(item).strip()],
            suggested_actions=[str(item).strip() for item in (known.get("suggestedActions") or []) if str(item).strip()],
            decision=KoNoteReviewDecision.from_payload(decision_payload) if isinstance(decision_payload, dict) else None,
            remediation=KoNoteRemediation.from_payload(remediation_payload) if isinstance(remediation_payload, dict) else KoNoteRemediation(),
            generated_at=str(known.get("generatedAt") or ""),
            version=str(known.get("version") or ""),
            extras=extras,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **dict(self.extras),
            "queue": bool(self.queue),
            "reasons": list(self.reasons),
            "patchHints": list(self.patch_hints),
            "suggestedActions": list(self.suggested_actions),
            "decision": self.decision.to_payload() if self.decision is not None and not self.decision.is_empty() else None,
            "remediation": self.remediation.to_payload(),
            "generatedAt": str(self.generated_at or ""),
            "version": str(self.version or ""),
        }

    def has_decision(self, status: str) -> bool:
        return self.decision is not None and str(self.decision.status or "") == str(status or "")

    def with_decision(
        self,
        *,
        status: str,
        reviewer: str,
        note: str,
        reviewed_at: str,
        queue: bool | None = None,
    ) -> "KoNoteReview":
        self.decision = KoNoteReviewDecision(
            status=str(status or ""),
            reviewer=str(reviewer or ""),
            note=str(note or ""),
            reviewed_at=str(reviewed_at or ""),
        )
        if queue is not None:
            self.queue = bool(queue)
        return self


@dataclass
class KoNoteApproval:
    mode: str = ""
    policy_version: str = ""
    approved_at: str = ""
    approved_by: str = ""
    reasons: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "KoNoteApproval":
        known_keys = {"mode", "policyVersion", "approvedAt", "approvedBy", "reasons", "signals"}
        raw = _section_payload(payload, "approval", known_keys=known_keys)
        known, extras = _split_known(raw, known_keys)
        return cls(
            mode=str(known.get("mode") or ""),
            policy_version=str(known.get("policyVersion") or ""),
            approved_at=str(known.get("approvedAt") or ""),
            approved_by=str(known.get("approvedBy") or ""),
            reasons=[str(item).strip() for item in (known.get("reasons") or []) if str(item).strip()],
            signals=[str(item).strip() for item in (known.get("signals") or []) if str(item).strip()],
            extras=extras,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **dict(self.extras),
            "mode": str(self.mode or ""),
            "policyVersion": str(self.policy_version or ""),
            "approvedAt": str(self.approved_at or ""),
            "approvedBy": str(self.approved_by or ""),
            "reasons": list(self.reasons),
            "signals": list(self.signals),
        }

    def is_auto(self) -> bool:
        return str(self.mode or "") == "auto"

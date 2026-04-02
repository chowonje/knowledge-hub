"""Local-first policy utilities for Learning Coach."""

from __future__ import annotations

from typing import Iterable
from uuid import uuid4

from knowledge_hub.core.sanitizer import classify_payload_level, pointer_from_snippet
from knowledge_hub.learning.models import EvidencePointer, PolicyStatus

def pointer_from_text(raw: str, fallback_path: str) -> EvidencePointer:
    """Backward-compatible wrapper for pointer conversion."""
    converted = pointer_from_snippet(raw, fallback_path=fallback_path)
    return converted


def ensure_pointer_list(values: Iterable[EvidencePointer | dict]) -> list[EvidencePointer]:
    pointers: list[EvidencePointer] = []
    for value in values:
        if isinstance(value, EvidencePointer):
            pointers.append(value)
            continue
        if isinstance(value, dict):
            pointers.append(
                EvidencePointer(
                    type=str(value.get("type", "note")),
                    path=str(value.get("path", "")),
                    heading=str(value.get("heading", "")),
                    block_id=str(value.get("block_id", value.get("blockId", ""))),
                    snippet_hash=str(value.get("snippet_hash", value.get("snippetHash", ""))),
                )
            )
    return pointers


def evaluate_policy_for_payload(
    allow_external: bool,
    raw_texts: list[str],
    mode: str = "local-only",
) -> PolicyStatus:
    classification = classify_payload_level(raw_texts)
    warnings: list[str] = []
    trace_id = f"policy_{uuid4().hex[:12]}"
    base_rule = "allow_non_p0"

    if allow_external and classification == "P0":
        return PolicyStatus(
            mode=mode,
            allowed=False,
            classification="P0",
            rule="deny_p0_raw_sensitive",
            trace_id=trace_id,
            blocked_reason="p0-detected-for-external-call",
            policy_errors=["policy deny: P0 raw content cannot be sent to external calls"],
            warnings=warnings,
        )

    if classification == "P1":
        warnings.append("P1 structured facts detected")
        base_rule = "allow_p1_with_warning"

    if not allow_external:
        warnings.append("allow_external=false: local-only mode enforced")
        base_rule = "local_only_no_external"

    return PolicyStatus(
        mode=mode,
        allowed=True,
        classification=classification,
        rule=base_rule,
        trace_id=trace_id,
        blocked_reason=None,
        policy_errors=[],
        warnings=warnings,
    )

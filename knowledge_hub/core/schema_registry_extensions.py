from __future__ import annotations

from typing import Final


SCHEMA_NAME_BY_ID_EXTENSIONS: Final = {
    "knowledge-hub.evidence-packet.input-completeness.v1": "evidence-packet-input-completeness.v1.json",
    "knowledge-hub.paper-understanding-readback.v1": "paper-understanding-readback.v1.json",
    "knowledge-hub.paper-understanding-profile-readiness.v1": "paper-understanding-profile-readiness.v1.json",
    "knowledge-hub.paper-answer-quality-harness.v1": "paper-answer-quality-harness.v1.json",
    "knowledge-hub.paper-real-answer-quality-gate.v1": "paper-real-answer-quality-gate.v1.json",
}


__all__ = ["SCHEMA_NAME_BY_ID_EXTENSIONS"]

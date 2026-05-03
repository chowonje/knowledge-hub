from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


DEFINITIONS_DIR = Path(__file__).resolve().parent / "definitions"


@dataclass(frozen=True)
class TransformationDefinition:
    id: str
    version: str
    title: str
    description: str
    prompt_template: str
    declared_inputs: dict[str, Any]
    declared_output: dict[str, Any]


def _definition_from_payload(payload: dict[str, Any]) -> TransformationDefinition:
    return TransformationDefinition(
        id=str(payload["id"]),
        version=str(payload["version"]),
        title=str(payload.get("title") or payload["id"]),
        description=str(payload.get("description") or ""),
        prompt_template=str(payload["prompt_template"]),
        declared_inputs=dict(payload.get("declared_inputs") or {}),
        declared_output=dict(payload.get("declared_output") or {}),
    )


def load_transformations() -> list[TransformationDefinition]:
    items: list[TransformationDefinition] = []
    for path in sorted(DEFINITIONS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items.append(_definition_from_payload(payload))
    return items


def get_transformation(transformation_id: str) -> TransformationDefinition | None:
    target = str(transformation_id or "").strip()
    if not target:
        return None
    for item in load_transformations():
        if item.id == target:
            return item
    return None

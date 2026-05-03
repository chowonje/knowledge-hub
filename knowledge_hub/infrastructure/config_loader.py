from __future__ import annotations

import os
from pathlib import Path

from knowledge_hub.infrastructure.config import DEFAULT_CONFIG_PATH

KHUB_CONFIG_ENV = "KHUB_CONFIG"


def resolve_config_path(config_path: str | None = None, *, project_root: str | Path | None = None) -> str | None:
    if config_path:
        return str(Path(config_path).expanduser())

    env_path = os.getenv(KHUB_CONFIG_ENV, "").strip()
    if env_path:
        return str(Path(env_path).expanduser())

    if project_root:
        candidate = Path(project_root) / "config.yaml"
        if candidate.exists():
            return str(candidate)

    if DEFAULT_CONFIG_PATH.exists():
        return str(DEFAULT_CONFIG_PATH)

    return None

"""Pipeline and policy helpers for the web ingest facade."""

from __future__ import annotations

import os
from pathlib import Path


def pipeline_root(service) -> Path:
    raw = str(
        service.config.get_nested(
            "pipeline",
            "storage",
            "root",
            default="~/.khub/knowledge_os",
        )
        or "~/.khub/knowledge_os"
    ).strip()
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def pipeline_profile(service, profile: str | None = None) -> tuple[str, dict[str, int]]:
    configured = str(
        profile
        or service.config.get_nested("pipeline", "profile", default="safe")
        or "safe"
    ).strip().lower()
    defaults = {
        "workers": 2,
        "download_concurrency": 2,
        "normalize_concurrency": 2,
        "embed_batch_size": 4,
    }
    profile_cfg = service.config.get_nested("pipeline", "profiles", configured, default={})
    if not isinstance(profile_cfg, dict):
        profile_cfg = {}
    merged = dict(defaults)
    for key, value in profile_cfg.items():
        try:
            merged[str(key)] = int(value)
        except Exception:
            continue
    return configured, merged


def resource_thresholds(service) -> tuple[float, float, float, float]:
    mem_high = float(
        service.config.get_nested(
            "pipeline",
            "resource",
            "memory_high_watermark",
            default=0.82,
        )
        or 0.82
    )
    cpu_high = float(
        service.config.get_nested(
            "pipeline",
            "resource",
            "cpu_pause_threshold",
            default=0.90,
        )
        or 0.90
    )
    backoff_base = float(
        service.config.get_nested(
            "pipeline",
            "resource",
            "backoff_base_sec",
            default=1.0,
        )
        or 1.0
    )
    backoff_max = float(
        service.config.get_nested(
            "pipeline",
            "resource",
            "backoff_max_sec",
            default=30.0,
        )
        or 30.0
    )
    return (
        max(0.05, min(mem_high, 0.99)),
        max(0.05, min(cpu_high, 0.99)),
        max(0.1, backoff_base),
        max(1.0, backoff_max),
    )


def sample_resource_ratio(service) -> tuple[float, float]:
    memory_ratio = 0.0
    try:
        import psutil  # type: ignore

        memory_ratio = max(0.0, min(1.0, float(psutil.virtual_memory().percent) / 100.0))
    except Exception:
        memory_ratio = 0.0

    cpu_ratio = 0.0
    try:
        load1 = float(os.getloadavg()[0])
        cpu_count = max(1, int(os.cpu_count() or 1))
        cpu_ratio = max(0.0, min(1.0, load1 / cpu_count))
    except Exception:
        cpu_ratio = 0.0
    return memory_ratio, cpu_ratio


def domain_allowlist(service) -> set[str]:
    allowlist = service.config.get_nested("pipeline", "allowlist_domains", default=[])
    if not isinstance(allowlist, list):
        return set()
    return {str(item).strip().lower() for item in allowlist if str(item).strip()}


def is_domain_allowed(
    service,
    sqlite_db,
    *,
    domain: str,
    source_policy: str,
    allowlist: set[str],
) -> tuple[bool, str]:
    del service
    domain_name = str(domain or "").strip().lower()
    if not domain_name:
        return False, "invalid_domain"

    # CLI exposes fixed | hybrid | keyword; only "fixed" is allowlist-only. "keyword" shares
    # the hybrid path (allowlist OR approved row OR pending/rejected in crawl_domain_policy).
    policy = str(source_policy or "hybrid").strip().lower()
    if policy == "fixed":
        if domain_name in allowlist:
            sqlite_db.upsert_crawl_domain_policy(domain_name, "approved", reason="allowlist")
            return True, "approved_allowlist"
        sqlite_db.upsert_crawl_domain_policy(domain_name, "pending", reason="not-in-allowlist")
        return False, "pending_domain"

    if domain_name in allowlist:
        sqlite_db.upsert_crawl_domain_policy(domain_name, "approved", reason="allowlist")
        return True, "approved_allowlist"

    row = sqlite_db.get_crawl_domain_policy(domain_name)
    if row:
        status = str(row.get("status", "pending")).strip().lower()
        if status == "approved":
            return True, "approved"
        if status == "rejected":
            return False, "rejected_domain"
        return False, "pending_domain"

    sqlite_db.upsert_crawl_domain_policy(domain_name, "pending", reason="new-domain-needs-approval")
    return False, "pending_domain"


def pipeline_paths(
    service,
    source: str,
    fetched_at: str | None,
    record_id: str,
    *,
    day_parts_fn,
) -> tuple[Path, Path, Path]:
    y, m, d = day_parts_fn(fetched_at)
    root = pipeline_root(service)
    raw_dir = root / "raw" / source / y / m / d / record_id
    normalized_path = root / "normalized" / source / y / m / d / f"{record_id}.json"
    indexed_path = root / "indexed" / source / y / m / d / f"{record_id}.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    indexed_path.parent.mkdir(parents=True, exist_ok=True)
    return raw_dir, normalized_path, indexed_path


__all__ = [
    "domain_allowlist",
    "is_domain_allowed",
    "pipeline_paths",
    "pipeline_profile",
    "pipeline_root",
    "resource_thresholds",
    "sample_resource_ratio",
]

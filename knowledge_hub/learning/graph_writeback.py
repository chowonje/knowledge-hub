"""Obsidian writeback for learning graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_hub.learning.mapper import slugify_topic
from knowledge_hub.learning.obsidian_writeback import (
    FileSystemVaultAdapter,
    ObsidianCliPreferredAdapter,
    VaultWriteAdapter,
    _read,
    _split_frontmatter,
    _upsert_marked_section,
    _write,
)


def _resolve_adapter(vault_path: str, backend: str = "filesystem", cli_binary: str = "obsidian", vault_name: str = "") -> VaultWriteAdapter:
    mode = (backend or "filesystem").strip().lower()
    if mode in {"cli-preferred", "cli", "obsidian-cli"}:
        return ObsidianCliPreferredAdapter(vault_path=vault_path, cli_binary=cli_binary, vault_name=vault_name)
    return FileSystemVaultAdapter()


def _concept_note_path(vault_path: str, canonical_name: str) -> Path:
    vault = Path(vault_path).expanduser().resolve()
    return vault / "Projects" / "AI" / "AI_Papers" / "Concepts" / f"{canonical_name}.md"


def _paper_note_path(vault_path: str, title: str) -> Path:
    vault = Path(vault_path).expanduser().resolve()
    return vault / "Projects" / "AI" / "AI_Papers" / "Papers" / f"{title}.md"


def _topic_index_path(vault_path: str, topic: str) -> Path:
    vault = Path(vault_path).expanduser().resolve()
    return vault / "Projects" / "AI" / "AI_Papers" / "Indexes" / "Learning_Paths" / f"{slugify_topic(topic)}.md"


def _review_index_path(vault_path: str, topic: str, stem: str) -> Path:
    vault = Path(vault_path).expanduser().resolve()
    return vault / "Projects" / "AI" / "AI_Papers" / "Indexes" / "Reviews" / f"{stem}_{slugify_topic(topic)}.md"


def write_concept_learning_section(
    vault_path: str,
    canonical_name: str,
    payload: dict[str, Any],
    *,
    backend: str = "filesystem",
    cli_binary: str = "obsidian",
    vault_name: str = "",
) -> Path | None:
    target = _concept_note_path(vault_path, canonical_name)
    if not target.exists():
        return None
    adapter = _resolve_adapter(vault_path, backend=backend, cli_binary=cli_binary, vault_name=vault_name)
    content = _read(target, adapter=adapter)
    body = "\n".join(
        [
            "## Learning",
            f"- difficulty: `{payload.get('difficultyLevel', 'intermediate')}`",
            f"- stage: `{payload.get('stage', payload.get('difficultyLevel', 'intermediate'))}`",
            f"- prerequisites: {', '.join(payload.get('prerequisites', []) or []) or '-'}",
            f"- study next: {', '.join(payload.get('studyNext', []) or []) or '-'}",
            f"- representative papers: {', '.join(payload.get('representativePapers', []) or []) or '-'}",
        ]
    )
    updated = _upsert_marked_section(content, "learning-graph", body)
    _write(target, updated, adapter=adapter)
    return target


def write_paper_learning_context(
    vault_path: str,
    title: str,
    payload: dict[str, Any],
    *,
    backend: str = "filesystem",
    cli_binary: str = "obsidian",
    vault_name: str = "",
) -> Path | None:
    target = _paper_note_path(vault_path, title)
    if not target.exists():
        return None
    adapter = _resolve_adapter(vault_path, backend=backend, cli_binary=cli_binary, vault_name=vault_name)
    content = _read(target, adapter=adapter)
    body = "\n".join(
        [
            "## Learning Context",
            f"- under concept: {payload.get('underConcept', '-')}",
            f"- recommended reading stage: `{payload.get('readingStage', '-')}`",
            f"- prerequisite concepts: {', '.join(payload.get('prerequisites', []) or []) or '-'}",
            f"- why read this paper now: {payload.get('whyReadNow', '-')}",
        ]
    )
    updated = _upsert_marked_section(content, "learning-context", body)
    _write(target, updated, adapter=adapter)
    return target


def write_topic_learning_path(
    vault_path: str,
    topic: str,
    path_payload: dict[str, Any],
    *,
    backend: str = "filesystem",
    cli_binary: str = "obsidian",
    vault_name: str = "",
) -> Path:
    target = _topic_index_path(vault_path, topic)
    adapter = _resolve_adapter(vault_path, backend=backend, cli_binary=cli_binary, vault_name=vault_name)
    lines: list[str] = [
        f"# Learning Path: {topic}",
        "",
        f"- status: {path_payload.get('status', 'approved')}",
        "",
    ]
    for stage_name in ("beginner", "intermediate", "advanced"):
        lines.append(f"## {stage_name.title()}")
        stage_items = path_payload.get("stages", {}).get(stage_name, [])
        if not stage_items:
            lines.append("- (none)")
            lines.append("")
            continue
        for idx, item in enumerate(stage_items, start=1):
            lines.append(f"{idx}. [[{item.get('canonicalName')}]]")
            papers = item.get("papers") or []
            for paper in papers[:5]:
                lines.append(f"   - paper: `{paper.get('link_type')}` -> `{paper.get('resource_node_id')}`")
        lines.append("")
    _write(target, "\n".join(lines).rstrip() + "\n", adapter=adapter)
    return target


def write_learning_review_notes(
    vault_path: str,
    topic: str,
    review_payload: dict[str, Any],
    *,
    backend: str = "filesystem",
    cli_binary: str = "obsidian",
    vault_name: str = "",
) -> dict[str, str]:
    adapter = _resolve_adapter(vault_path, backend=backend, cli_binary=cli_binary, vault_name=vault_name)
    claims_path = _review_index_path(vault_path, topic, "Claims_to_Review")
    concepts_path = _review_index_path(vault_path, topic, "Top_Concepts")
    queue_path = _review_index_path(vault_path, topic, "Reading_Queue")

    claim_lines = [f"# Claims to Review: {topic}", "", "## Pending Claims"]
    for item in review_payload.get("claims", []) or []:
        reason = item.get("reason_json") if isinstance(item.get("reason_json"), dict) else {}
        claim_text = reason.get("claim_text") or reason.get("claimText") or item.get("predicate_id") or "claim"
        source_entity = item.get("source_entity_id") or "-"
        predicate = item.get("predicate_id") or "-"
        claim_lines.append(
            f"- confidence `{float(item.get('confidence', 0.0)):.2f}` "
            f"[`{source_entity}` / `{predicate}`]: {claim_text}"
        )
    if len(claim_lines) == 3:
        claim_lines.append("- (none)")
    claim_lines.extend(["", "## Merge Proposals"])
    for item in review_payload.get("mergeProposals", []) or []:
        source_name = item.get("source_canonical_name") or item.get("source_entity_id") or "-"
        target_name = item.get("target_canonical_name") or item.get("target_entity_id") or "-"
        cluster_size = item.get("duplicate_cluster_size") or 0
        claim_lines.append(
            f"- confidence `{float(item.get('confidence', 0.0)):.2f}`: "
            f"`{source_name}` -> `{target_name}` (cluster `{cluster_size}`)"
        )
    if not (review_payload.get("mergeProposals") or []):
        claim_lines.append("- (none)")
    claim_lines.extend(["", "## Contradiction Hotspots"])
    for item in review_payload.get("contradictionHotspots", []) or []:
        concept_name = item.get("feature_name") or item.get("feature_key") or "-"
        claim_lines.append(
            f"- [[{concept_name}]] contradiction `{float(item.get('contradiction_score', 0.0)):.2f}` "
            f"(importance `{float(item.get('importance_score', 0.0)):.2f}`)"
        )
    if not (review_payload.get("contradictionHotspots") or []):
        claim_lines.append("- (none)")
    claim_lines.extend(["", "## Predicate Validation"])
    for item in review_payload.get("predicateValidation", []) or []:
        source_name = item.get("source_name") or item.get("source_entity_id") or "-"
        target_name = item.get("target_name") or item.get("target_entity_id") or "-"
        issues = ", ".join(str(issue) for issue in (item.get("issues") or [])) or "-"
        claim_lines.append(
            f"- `{item.get('predicate_id', '-')}` `{source_name}` -> `{target_name}`: {issues}"
        )
    if not (review_payload.get("predicateValidation") or []):
        claim_lines.append("- (none)")
    claim_lines.extend(["", "## Learning Pending"])
    for item in review_payload.get("learningPending", []) or []:
        reason = item.get("reason") or item.get("item_type") or "pending"
        claim_lines.append(
            f"- `{item.get('item_type', '-')}` confidence `{float(item.get('confidence', 0.0)):.2f}`: {reason}"
        )
    if not (review_payload.get("learningPending") or []):
        claim_lines.append("- (none)")
    _write(claims_path, "\n".join(claim_lines).rstrip() + "\n", adapter=adapter)

    concept_lines = [f"# Top Concepts: {topic}", ""]
    for idx, item in enumerate(review_payload.get("topConcepts", []) or [], start=1):
        concept_name = item.get("feature_name") or item.get("feature_key")
        support_doc_count = int(item.get("support_doc_count") or 0)
        claim_density = float(item.get("claim_density") or 0.0)
        concept_lines.append(
            f"{idx}. [[{concept_name}]] "
            f"(importance `{float(item.get('importance_score', 0.0)):.2f}`, "
            f"contradiction `{float(item.get('contradiction_score', 0.0)):.2f}`, "
            f"support `{support_doc_count}`, claim density `{claim_density:.2f}`)"
        )
    if len(concept_lines) == 2:
        concept_lines.append("- (none)")
    _write(concepts_path, "\n".join(concept_lines).rstrip() + "\n", adapter=adapter)

    queue_lines = [f"# Reading Queue: {topic}", ""]
    queue_lines.append("## Top Study Next")
    for item in review_payload.get("studyNextItems", []) or []:
        queue_lines.append(
            f"- [[{item.get('concept')}]] "
            f"(`{item.get('stage', '-')}` / `{item.get('difficulty', '-')}`)"
        )
        prerequisites = item.get("prerequisites") or []
        if prerequisites:
            queue_lines.append(f"  - prerequisites: {', '.join(prerequisites)}")
    if not (review_payload.get("studyNextItems") or []):
        queue_lines.append("- (none)")
    queue_lines.append("")
    for stage_name in ("beginner", "intermediate", "advanced"):
        queue_lines.append(f"## {stage_name.title()}")
        items = review_payload.get("readingQueue", {}).get(stage_name, []) or []
        if not items:
            queue_lines.append("- (none)")
            queue_lines.append("")
            continue
        for item in items:
            queue_lines.append(f"- [[{item.get('concept')}]]")
            prerequisites = item.get("prerequisites") or []
            if prerequisites:
                queue_lines.append(f"  - prerequisites: {', '.join(prerequisites)}")
            for paper in item.get("papers", [])[:5]:
                queue_lines.append(f"  - {paper}")
        queue_lines.append("")
    _write(queue_path, "\n".join(queue_lines).rstrip() + "\n", adapter=adapter)

    return {
        "claimsToReview": str(claims_path),
        "topConcepts": str(concepts_path),
        "readingQueue": str(queue_path),
    }

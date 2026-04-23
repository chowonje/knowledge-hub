"""Vault organization utilities.

The organizer builds topic-based collections from existing markdown notes so users
can review material in a coherent order, even when the original folders are
mixed.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import frontmatter
except Exception:  # pragma: no cover - optional fallback
    frontmatter = None
import yaml

from knowledge_hub.core.keywords import STOP_ENGLISH, STOP_KOREAN, extract_keywords_from_text
from knowledge_hub.core.vault_paths import resolve_vault_exclude_folders

log = logging.getLogger("khub.vault.organizer")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(value: str, fallback: str) -> str:
    normalized = re.sub(r"\s+", " ", (value or "").strip().lower())
    slug = re.sub(r"[^a-z0-9가-힣]+", "-", normalized).strip("-")
    return slug or fallback


def _normalize_token(value: str) -> str:
    token = re.sub(r"\s+", " ", (value or "").strip().lower())
    return token


def _tokenize_terms(text: str) -> set[str]:
    raw = set(re.findall(r"[A-Za-z][A-Za-z0-9]+|[가-힣]{2,}", text or ""))
    out: set[str] = set()
    for token in raw:
        low = token.lower()
        if len(low) < 2:
            continue
        if low in STOP_ENGLISH or low in STOP_KOREAN:
            continue
        out.add(low)
    return out


def _tokenize_title(title: str) -> set[str]:
    return _tokenize_terms(title)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _extract_tags(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("tags", [])
    tags: list[str] = []
    if isinstance(raw, str):
        tags = [part.strip() for part in re.split(r"[, ]+", raw) if part.strip()]
    elif isinstance(raw, list):
        tags = [str(item).strip() for item in raw if str(item).strip()]
    normalized: list[str] = []
    for tag in tags:
        clean = _normalize_token(tag.lstrip("#"))
        if not clean:
            continue
        normalized.append(clean)
    return normalized


def _parse_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if frontmatter:
        try:
            post = frontmatter.loads(text)
            metadata = dict(post.metadata or {})
            content = (post.content or "").strip()
            if metadata:
                return metadata, content
            # Some indented frontmatter blocks can be missed by python-frontmatter.
            # Fall through to regex parser before treating the file as plain markdown.
        except Exception:
            log.warning("frontmatter parse failed for %s; fallback parser used", path)
    match = re.match(r"^\s*---\s*\n(.*?)\n\s*---\s*\n?", text, flags=re.DOTALL)
    if match:
        frontmatter_block = match.group(1)
        metadata: dict[str, Any] = {}
        try:
            parsed = yaml.safe_load(frontmatter_block) or {}
            if isinstance(parsed, dict):
                metadata = parsed
        except Exception:
            metadata = {}
        body = text[match.end() :].strip()
        return metadata, body
    return {}, text.strip()


def _estimate_difficulty(metadata: dict[str, Any], content: str) -> int:
    raw = str(metadata.get("difficulty", "")).strip().lower()
    mapping = {
        "beginner": 1,
        "basic": 1,
        "intermediate": 2,
        "medium": 2,
        "advanced": 3,
        "expert": 4,
        "입문": 1,
        "기초": 1,
        "중급": 2,
        "심화": 3,
        "고급": 4,
    }
    if raw in mapping:
        return mapping[raw]

    words = len(re.findall(r"[A-Za-z0-9가-힣]+", content or ""))
    score = 1
    if words >= 500:
        score += 1
    if words >= 1200:
        score += 1
    if len(re.findall(r"```", content or "")) >= 4:
        score += 1
    return max(1, min(4, score))


@dataclass
class NoteProfile:
    title: str
    abs_path: Path
    rel_path: Path
    source_folder: str
    tags: list[str]
    keywords: list[str]
    keyword_set: set[str]
    token_set: set[str]
    difficulty: int
    word_count: int


class VaultOrganizer:
    """Organize markdown notes into topic collections."""

    def __init__(self, vault_path: str, exclude_folders: list[str] | None = None):
        self.vault_path = Path(vault_path).expanduser().resolve()
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {self.vault_path}")
        self.exclude_folders = resolve_vault_exclude_folders(exclude_folders)

    def _scan_notes(self, output_dir: Path) -> list[NoteProfile]:
        notes: list[NoteProfile] = []
        output_parts = output_dir.parts
        for md_path in self.vault_path.rglob("*.md"):
            rel = md_path.relative_to(self.vault_path)
            if any(excluded in rel.parts for excluded in self.exclude_folders):
                continue
            if output_parts and tuple(rel.parts[: len(output_parts)]) == output_parts:
                continue

            metadata, content = _parse_markdown(md_path)
            if not content.strip():
                continue

            title = str(metadata.get("title") or md_path.stem).strip()
            tags = _extract_tags(metadata)
            keywords = extract_keywords_from_text(
                f"{title}\n\n{content}",
                max_keywords=14,
            )
            token_set: set[str] = set()
            for item in keywords[:10]:
                token_set.update(_tokenize_terms(item))
            token_set.update(_tokenize_title(title))
            for tag in tags:
                normalized_tag = _normalize_token(tag)
                if normalized_tag:
                    token_set.add(normalized_tag)
                    token_set.update(_tokenize_terms(normalized_tag))

            keyword_set: set[str] = set()
            for item in keywords[:6]:
                keyword_set.update(_tokenize_terms(item))
            if not keyword_set:
                keyword_set = set(token_set)

            notes.append(
                NoteProfile(
                    title=title,
                    abs_path=md_path,
                    rel_path=rel,
                    source_folder=rel.parent.as_posix() if rel.parent.as_posix() != "." else "",
                    tags=tags,
                    keywords=keywords,
                    keyword_set=keyword_set,
                    token_set=token_set,
                    difficulty=_estimate_difficulty(metadata, content),
                    word_count=len(re.findall(r"[A-Za-z0-9가-힣]+", content)),
                )
            )
        return notes

    @staticmethod
    def _note_similarity(left: NoteProfile, right: NoteProfile) -> float:
        token_score = _jaccard(left.token_set, right.token_set)
        key_overlap = left.keyword_set & right.keyword_set
        denom = max(1, min(len(left.keyword_set), len(right.keyword_set)))
        keyword_score = len(key_overlap) / denom
        tag_overlap = bool(set(left.tags) & set(right.tags))
        score = 0.55 * token_score + 0.35 * keyword_score + (0.15 if tag_overlap else 0.0)
        return min(1.0, score)

    @classmethod
    def _cluster_notes(cls, notes: list[NoteProfile], similarity_threshold: float) -> list[list[NoteProfile]]:
        if not notes:
            return []

        parent = list(range(len(notes)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for left in range(len(notes)):
            for right in range(left + 1, len(notes)):
                similarity = cls._note_similarity(notes[left], notes[right])
                if similarity >= similarity_threshold:
                    union(left, right)

        grouped: dict[int, list[NoteProfile]] = defaultdict(list)
        for idx, note in enumerate(notes):
            grouped[find(idx)].append(note)

        clusters = list(grouped.values())
        clusters.sort(
            key=lambda members: (-len(members), members[0].title.lower()),
        )
        return clusters

    @staticmethod
    def _cluster_label(cluster: list[NoteProfile], cluster_index: int) -> tuple[str, str]:
        counter: Counter[str] = Counter()
        for note in cluster:
            for token in note.keywords[:6]:
                normalized = _normalize_token(token)
                if not normalized or normalized in STOP_ENGLISH or normalized in STOP_KOREAN:
                    continue
                counter[normalized] += 1
            for tag in note.tags:
                normalized = _normalize_token(tag)
                if normalized:
                    counter[normalized] += 2

        top_terms = [token for token, _ in counter.most_common(3)]
        if top_terms:
            label = " / ".join(top_terms[:2])
            slug_seed = "-".join(top_terms[:2])
        else:
            label = cluster[0].title
            slug_seed = cluster[0].title

        fallback = f"cluster-{cluster_index:02d}"
        return label, _slugify(slug_seed, fallback=fallback)

    @staticmethod
    def _ordered_members(cluster: list[NoteProfile]) -> list[NoteProfile]:
        return sorted(
            cluster,
            key=lambda note: (
                note.difficulty,
                note.word_count,
                note.title.lower(),
            ),
        )

    def organize(
        self,
        apply: bool = False,
        relocate: bool = False,
        output_dir: str = "LearningHub/Collections",
        similarity_threshold: float = 0.16,
        min_cluster_size: int = 2,
    ) -> dict[str, Any]:
        output_root = Path(output_dir.strip("/")) if output_dir.strip() else Path("LearningHub/Collections")
        notes = self._scan_notes(output_root)
        clusters = self._cluster_notes(notes, similarity_threshold=max(0.0, min(1.0, similarity_threshold)))
        notes_total = sum(len(cluster) for cluster in clusters)
        singletons = sum(1 for cluster in clusters if len(cluster) < max(1, min_cluster_size))

        slug_counter: Counter[str] = Counter()
        cluster_rows: list[dict[str, Any]] = []
        moved_count = 0
        moved_files: list[dict[str, str]] = []

        for idx, cluster in enumerate(clusters, start=1):
            label, base_slug = self._cluster_label(cluster, idx)
            slug_counter[base_slug] += 1
            suffix = "" if slug_counter[base_slug] == 1 else f"-{slug_counter[base_slug]}"
            slug = f"{base_slug}{suffix}"

            collection_rel_dir = output_root / f"{idx:02d}-{slug}"
            collection_abs_dir = self.vault_path / collection_rel_dir
            index_rel_path = collection_rel_dir / "00_Index.md"
            index_abs_path = self.vault_path / index_rel_path

            ordered_notes = self._ordered_members(cluster)
            rows: list[dict[str, Any]] = []
            for order, note in enumerate(ordered_notes, start=1):
                current_abs = note.abs_path
                current_rel = note.rel_path
                if apply and relocate:
                    notes_rel_dir = collection_rel_dir / "notes"
                    notes_abs_dir = self.vault_path / notes_rel_dir
                    notes_abs_dir.mkdir(parents=True, exist_ok=True)
                    target_abs = notes_abs_dir / current_abs.name
                    if target_abs.resolve() != current_abs.resolve():
                        if target_abs.exists():
                            stem = target_abs.stem
                            suffix_idx = 2
                            while True:
                                candidate = target_abs.with_name(f"{stem}-{suffix_idx}{target_abs.suffix}")
                                if not candidate.exists():
                                    target_abs = candidate
                                    break
                                suffix_idx += 1
                        shutil.move(str(current_abs), str(target_abs))
                        moved_count += 1
                        moved_files.append(
                            {
                                "from": current_rel.as_posix(),
                                "to": target_abs.relative_to(self.vault_path).as_posix(),
                            }
                        )
                    current_abs = target_abs
                    current_rel = current_abs.relative_to(self.vault_path)

                rows.append(
                    {
                        "title": note.title,
                        "relPath": current_rel.as_posix(),
                        "sourceFolder": note.source_folder,
                        "difficulty": note.difficulty,
                        "wordCount": note.word_count,
                        "recommendedOrder": order,
                    }
                )

            cluster_rows.append(
                {
                    "id": f"cluster-{idx:03d}",
                    "label": label,
                    "slug": slug,
                    "size": len(cluster),
                    "singleton": len(cluster) < max(1, min_cluster_size),
                    "collectionPath": collection_rel_dir.as_posix(),
                    "indexPath": index_rel_path.as_posix(),
                    "notes": rows,
                    "sourceFolders": sorted({row["sourceFolder"] for row in rows if row["sourceFolder"]}),
                }
            )

            if apply:
                collection_abs_dir.mkdir(parents=True, exist_ok=True)
                self._write_collection_index(index_abs_path, label, rows, idx)

        summary_rel_path = output_root / "00_Collections.md"
        manifest_rel_path = output_root / "_organize_manifest.json"
        summary_abs_path = self.vault_path / summary_rel_path
        manifest_abs_path = self.vault_path / manifest_rel_path

        if apply:
            summary_abs_path.parent.mkdir(parents=True, exist_ok=True)
            summary_abs_path.write_text(
                self._render_summary(cluster_rows),
                encoding="utf-8",
            )
            manifest_abs_path.write_text(
                json.dumps(
                    {
                        "schema": "knowledge-hub.vault.organize.manifest.v1",
                        "generatedAt": _now_iso(),
                        "vaultPath": str(self.vault_path),
                        "mode": "apply",
                        "relocate": relocate,
                        "clusterCount": len(cluster_rows),
                        "noteCount": notes_total,
                        "singletonCount": singletons,
                        "movedCount": moved_count,
                        "movedFiles": moved_files,
                        "clusters": cluster_rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        return {
            "schema": "knowledge-hub.vault.organize.result.v1",
            "status": "ok",
            "mode": "apply" if apply else "preview",
            "relocate": relocate,
            "vaultPath": str(self.vault_path),
            "outputDir": output_root.as_posix(),
            "clusterCount": len(cluster_rows),
            "noteCount": notes_total,
            "singletonCount": singletons,
            "summaryPath": summary_rel_path.as_posix(),
            "manifestPath": manifest_rel_path.as_posix(),
            "movedCount": moved_count,
            "clusters": cluster_rows,
        }

    @staticmethod
    def _write_collection_index(
        index_path: Path,
        label: str,
        rows: list[dict[str, Any]],
        cluster_number: int,
    ) -> None:
        lines: list[str] = [
            f"# Collection {cluster_number:02d}: {label}",
            "",
            f"- Generated: {_now_iso()}",
            f"- Notes: {len(rows)}",
            "",
            "## Review Order",
        ]
        for row in rows:
            rel_no_ext = Path(row["relPath"]).with_suffix("").as_posix()
            lines.append(
                f"{row['recommendedOrder']}. [[{rel_no_ext}|{row['title']}]] "
                f"(difficulty {row['difficulty']}, words {row['wordCount']})"
            )
        lines.append("")
        index_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _render_summary(clusters: list[dict[str, Any]]) -> str:
        lines = [
            "# Organized Collections",
            "",
            f"- Generated: {_now_iso()}",
            f"- Collections: {len(clusters)}",
            "",
            "## Collections",
        ]
        for cluster in clusters:
            index_no_ext = Path(cluster["indexPath"]).with_suffix("").as_posix()
            lines.append(
                f"- [[{index_no_ext}|{cluster['label']}]] "
                f"({cluster['size']} notes)"
            )
        lines.append("")
        return "\n".join(lines)

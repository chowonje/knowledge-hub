"""Projects/AI vault organizer for paper, concept, web, and agent notes."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from knowledge_hub.notes.templates import split_frontmatter
import yaml

log = logging.getLogger("khub.vault.ai_organizer")

INDEX_KEYWORDS = {"index", "overview", "roadmap", "map"}
SPECIAL_DIR_MOVES = {
    Path("ai agent"): Path("AI_Agents"),
    Path("AI_Papers") / "scripts": Path("AI_Papers") / "Operations" / "scripts",
    Path("AI_Papers") / "study_packs": Path("AI_Papers") / "Operations" / "study_packs",
}


class AIVaultOrganizer:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path).expanduser().resolve()
        self.ai_root = self.vault_path / "Projects" / "AI"
        if not self.ai_root.exists():
            raise ValueError(f"Projects/AI path does not exist: {self.ai_root}")

    def organize_projects_ai(self, *, apply: bool = False) -> dict[str, Any]:
        markdown_files = self._collect_markdown_files()
        duplicate_groups = self._detect_duplicates(markdown_files)
        file_moves: list[dict[str, str]] = []
        manual_review: list[dict[str, str]] = []

        for source_path in markdown_files:
            rel = source_path.relative_to(self.ai_root)
            target_rel = self._classify_markdown(rel, source_path)
            if target_rel is None:
                manual_review.append({
                    "path": rel.as_posix(),
                    "reason": "unclassified",
                })
                continue
            if rel == target_rel:
                continue
            file_moves.append({
                "from": rel.as_posix(),
                "to": target_rel.as_posix(),
            })

        dir_moves = [
            {"from": old.as_posix(), "to": new.as_posix()}
            for old, new in SPECIAL_DIR_MOVES.items()
            if (self.ai_root / old).exists() and old != new
        ]

        conflict_review = self._detect_destination_conflicts(file_moves, dir_moves)
        if conflict_review:
            manual_review.extend(conflict_review)
        blocked_paths = {item["path"] for item in manual_review if item.get("reason") != "unclassified"}
        file_moves = [item for item in file_moves if item["from"] not in blocked_paths]
        dir_moves = [item for item in dir_moves if item["from"] not in blocked_paths]

        applied_moves: list[dict[str, str]] = []
        rewritten_links = 0
        if apply:
            applied_moves.extend(self._apply_directory_moves(dir_moves, manual_review))
            applied_moves.extend(self._apply_file_moves(file_moves, manual_review))
            replacements = self._build_replacements(applied_moves)
            rewritten_links = self._rewrite_links(replacements)

        return {
            "schema": "knowledge-hub.vault.organize-ai.result.v1",
            "status": "ok",
            "scope": "projects-ai",
            "mode": "apply" if apply else "dry-run",
            "vaultPath": str(self.vault_path),
            "rootPath": str(self.ai_root),
            "plannedMoveCount": len(file_moves) + len(dir_moves),
            "appliedMoveCount": len(applied_moves),
            "rewrittenLinks": rewritten_links,
            "fileMoves": file_moves,
            "directoryMoves": dir_moves,
            "duplicateGroups": duplicate_groups,
            "manualReview": manual_review,
        }

    def _collect_markdown_files(self) -> list[Path]:
        files: list[Path] = []
        for md_path in self.ai_root.rglob("*.md"):
            rel = md_path.relative_to(self.ai_root)
            if self._is_under_special_dir(rel):
                continue
            files.append(md_path)
        return sorted(files)

    @staticmethod
    def _is_under_special_dir(rel_path: Path) -> bool:
        for source_dir in SPECIAL_DIR_MOVES:
            if rel_path == source_dir or source_dir in rel_path.parents:
                return True
        return False

    def _classify_markdown(self, rel_path: Path, source_path: Path) -> Path | None:
        frontmatter, body = self._read_frontmatter_with_fallback(source_path)
        type_token = str(frontmatter.get("type") or "").strip().lower()
        status = str(frontmatter.get("status") or "").strip().lower()
        policy = str(frontmatter.get("policy") or "").strip().lower()
        archived = bool(frontmatter.get("archived"))
        stem = source_path.stem
        stem_lower = stem.lower()
        body_lower = body.lower()

        if rel_path.parts[:2] == ("AI_Papers", "Concepts"):
            return rel_path
        if rel_path.parts[:2] == ("AI_Papers", "Web_Sources"):
            return rel_path
        if rel_path.parts[:2] == ("AI_Papers", "Papers"):
            return rel_path
        if rel_path.parts[:2] == ("AI_Papers", "Indexes"):
            return rel_path
        if rel_path.parts[:2] == ("AI_Papers", "Operations"):
            return rel_path
        if rel_path.parts[:2] == ("AI_Papers", "Archives"):
            return rel_path

        if archived or status == "archived" or policy == "deprecated-merged":
            return Path("AI_Papers") / "Archives" / "Redirects" / rel_path.name
        if type_token == "paper" or frontmatter.get("arxiv_id"):
            return Path("AI_Papers") / "Papers" / rel_path.name
        if type_token == "concept":
            return Path("AI_Papers") / "Concepts" / rel_path.name
        if type_token == "web-source":
            return Path("AI_Papers") / "Web_Sources" / rel_path.name
        if self._looks_like_index(stem_lower, body_lower):
            return Path("AI_Papers") / "Indexes" / rel_path.name
        if self._looks_like_scratch_concept(frontmatter, body_lower):
            return Path("AI_Papers") / "Concepts" / rel_path.name
        return None

    @staticmethod
    def _read_frontmatter_with_fallback(source_path: Path) -> tuple[dict[str, Any], str]:
        text = source_path.read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(text)
        if frontmatter:
            return frontmatter, body
        lines = text.splitlines()
        pseudo_lines: list[str] = []
        for line in lines[:12]:
            stripped = line.strip()
            if not stripped:
                break
            if stripped == "---":
                break
            if ":" not in stripped:
                break
            pseudo_lines.append(line)
        if not pseudo_lines:
            return frontmatter, body
        try:
            parsed = yaml.safe_load("\n".join(pseudo_lines)) or {}
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict) or not parsed:
            return frontmatter, body
        remainder = text.splitlines()[len(pseudo_lines):]
        return parsed, "\n".join(remainder).lstrip("\n")

    @staticmethod
    def _looks_like_index(stem_lower: str, body_lower: str) -> bool:
        if any(keyword in stem_lower for keyword in INDEX_KEYWORDS):
            return True
        return "concept index" in body_lower or "개념 링크 목록" in body_lower

    @staticmethod
    def _looks_like_scratch_concept(frontmatter: dict[str, Any], body_lower: str) -> bool:
        if frontmatter.get("type") or frontmatter.get("arxiv_id"):
            return False
        if "## related notes" in body_lower:
            return True
        if "multilayer perceptron" in body_lower or "embedding" in body_lower:
            return True
        return False

    def _detect_destination_conflicts(
        self,
        file_moves: list[dict[str, str]],
        dir_moves: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        manual_review: list[dict[str, str]] = []
        claimed_targets: dict[str, str] = {}
        for move in dir_moves:
            target_abs = self.ai_root / move["to"]
            if target_abs.exists():
                manual_review.append({
                    "path": move["from"],
                    "target": move["to"],
                    "reason": "target-directory-exists",
                })
        for move in file_moves:
            target_abs = self.ai_root / move["to"]
            existing_claim = claimed_targets.get(move["to"])
            if existing_claim and existing_claim != move["from"]:
                manual_review.append({
                    "path": move["from"],
                    "target": move["to"],
                    "reason": "duplicate-target-planned",
                })
                continue
            claimed_targets[move["to"]] = move["from"]
            if target_abs.exists() and target_abs.resolve() != (self.ai_root / move["from"]).resolve():
                manual_review.append({
                    "path": move["from"],
                    "target": move["to"],
                    "reason": "target-file-exists",
                })
        return manual_review

    def _detect_duplicates(self, markdown_files: list[Path]) -> list[dict[str, Any]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for md_path in markdown_files:
            rel = md_path.relative_to(self.ai_root)
            key = re.sub(r"\s+", " ", md_path.stem.lower().replace("-", " ").replace("_", " ")).strip()
            groups[key].append(rel.as_posix())
        results: list[dict[str, Any]] = []
        for key, items in sorted(groups.items()):
            if len(items) > 1:
                results.append({"key": key, "items": sorted(items)})
        return results

    def _apply_directory_moves(
        self,
        dir_moves: list[dict[str, str]],
        manual_review: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        applied: list[dict[str, str]] = []
        for move in dir_moves:
            source_abs = self.ai_root / move["from"]
            target_abs = self.ai_root / move["to"]
            if not source_abs.exists():
                continue
            if target_abs.exists():
                manual_review.append({
                    "path": move["from"],
                    "target": move["to"],
                    "reason": "target-directory-exists-at-apply",
                })
                continue
            target_abs.parent.mkdir(parents=True, exist_ok=True)
            source_abs.rename(target_abs)
            applied.append(move)
        return applied

    def _apply_file_moves(
        self,
        file_moves: list[dict[str, str]],
        manual_review: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        applied: list[dict[str, str]] = []
        for move in file_moves:
            source_abs = self.ai_root / move["from"]
            target_abs = self.ai_root / move["to"]
            if not source_abs.exists():
                continue
            if target_abs.exists() and target_abs.resolve() != source_abs.resolve():
                manual_review.append({
                    "path": move["from"],
                    "target": move["to"],
                    "reason": "target-file-exists-at-apply",
                })
                continue
            target_abs.parent.mkdir(parents=True, exist_ok=True)
            source_abs.rename(target_abs)
            applied.append(move)
        return applied

    def _build_replacements(self, applied_moves: list[dict[str, str]]) -> list[tuple[str, str]]:
        replacements: list[tuple[str, str]] = []
        for move in applied_moves:
            old_rel = f"Projects/AI/{move['from']}"
            new_rel = f"Projects/AI/{move['to']}"
            replacements.append((old_rel, new_rel))
            replacements.append((old_rel.replace("Projects/", "", 1), new_rel.replace("Projects/", "", 1)))
            if old_rel.startswith("Projects/AI/AI_Papers/"):
                replacements.append(
                    (old_rel.replace("Projects/AI/", "", 1), new_rel.replace("Projects/AI/", "", 1))
                )
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in replacements:
            if item[0] == item[1] or item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _rewrite_links(self, replacements: list[tuple[str, str]]) -> int:
        if not replacements:
            return 0
        rewritten = 0
        for md_path in self.ai_root.rglob("*.md"):
            try:
                original = md_path.read_text(encoding="utf-8")
            except Exception as error:
                log.warning("failed to read %s for link rewrite: %s", md_path, error)
                continue
            updated = original
            for old, new in replacements:
                updated = self._rewrite_single_reference(updated, old, new)
            if updated != original:
                md_path.write_text(updated, encoding="utf-8")
                rewritten += 1
        return rewritten

    @staticmethod
    def _rewrite_single_reference(text: str, old: str, new: str) -> str:
        old_md = old if old.endswith(".md") else f"{old}.md"
        new_md = new if new.endswith(".md") else f"{new}.md"
        old_no_ext = old[:-3] if old.endswith(".md") else old
        new_no_ext = new[:-3] if new.endswith(".md") else new
        updated = text
        updated = updated.replace(f"[[{old_no_ext}]]", f"[[{new_no_ext}]]")
        updated = updated.replace(f"[[{old_no_ext}|", f"[[{new_no_ext}|")
        updated = updated.replace(f"[[{old_no_ext}#", f"[[{new_no_ext}#")
        updated = updated.replace(f"]({old_md})", f"]({new_md})")
        updated = updated.replace(f"]({old_no_ext})", f"]({new_no_ext})")
        updated = updated.replace(f"({old}/", f"({new}/")
        updated = updated.replace(f"[[{old}/", f"[[{new}/")
        return updated

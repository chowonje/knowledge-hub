from __future__ import annotations

from pathlib import Path, PurePosixPath

_SKIP_WIKILINK_ROOTS = {
    "archives",
    "learninghub",
    "papers",
    "projects",
}


def iter_concept_note_paths(concepts_dir: Path) -> list[Path]:
    root = Path(concepts_dir)
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*.md") if path.is_file()),
        key=lambda path: str(path.relative_to(root)).casefold(),
    )


def normalize_concept_wikilink_target(raw: str) -> str:
    token = str(raw or "").strip()
    if not token:
        return ""
    token = token.split("|", 1)[0].strip()
    if token.endswith(".md"):
        token = token[:-3].strip()
    token = token.replace("\\", "/").strip().strip("/")
    if not token:
        return ""
    first = token.split("/", 1)[0].casefold()
    if first in _SKIP_WIKILINK_ROOTS:
        return ""
    return PurePosixPath(token).name.strip()


def find_concept_note_path(concepts_dir: Path, concept_name: str) -> Path | None:
    root = Path(concepts_dir)
    token = str(concept_name or "").strip()
    if not token or not root.exists():
        return None

    direct = root / token
    if direct.suffix.lower() != ".md":
        direct = direct.with_suffix(".md")
    if direct.exists():
        return direct

    normalized = token.replace("\\", "/").strip().strip("/")
    if normalized:
        direct_posix = root.joinpath(*[part for part in normalized.split("/") if part]).with_suffix(".md")
        if direct_posix.exists():
            return direct_posix

    target_name = normalize_concept_wikilink_target(token).casefold()
    if not target_name:
        return None
    for path in iter_concept_note_paths(root):
        if path.stem.casefold() == target_name:
            return path
    return None

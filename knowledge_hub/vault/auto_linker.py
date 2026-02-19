"""
Obsidian 자동 링크 생성

고립된 노트(그래프에서 연결 없는 노트)를 찾아
키워드 기반으로 관련 노트 [[링크]]를 자동 삽입합니다.

Obsidian 그래프 뷰에서 바로 반영됩니다.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set, Optional

from rich.console import Console
from rich.table import Table

try:
    import frontmatter as fm_lib
except ImportError:
    fm_lib = None

console = Console()

LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
AUTO_BLOCK_START = "<!-- AUTO_LINK_START -->"
AUTO_BLOCK_END = "<!-- AUTO_LINK_END -->"

EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "this", "that",
    "with", "from", "for", "into", "about", "is", "are", "was", "were",
    "be", "been", "have", "has", "had", "do", "does", "did", "to", "in",
    "on", "at", "by", "of", "as", "it", "its", "we", "you", "they", "he",
    "she", "i", "me", "my", "our", "your", "their", "not", "will", "can",
}
KO_STOPWORDS = {
    "그", "이", "저", "것", "들", "에서", "하는", "있다", "없다", "있는",
    "때", "및", "또는", "그리고", "하지만", "그러나", "수", "때문에", "위해", "등",
}


@dataclass
class VaultNote:
    note_id: str
    rel_path: Path
    title: str
    content: str = ""
    aliases: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    raw_links: List[str] = field(default_factory=list)
    keywords: Set[str] = field(default_factory=set)
    outbound: Set[str] = field(default_factory=set)


def _normalize_key(value: str) -> str:
    text = value.strip().split("#", 1)[0].split("^", 1)[0]
    if text.lower().endswith(".md"):
        text = text[:-3]
    return re.sub(r"\s+", " ", text.lower().strip())


def _parse_frontmatter(text: str):
    if fm_lib:
        try:
            post = fm_lib.loads(text)
            return dict(post.metadata or {}), post.content or ""
        except Exception:
            pass
    if text.startswith("---\n") and "\n---\n" in text[4:]:
        end = text.find("\n---\n", 4)
        return {}, text[end + 5:]
    return {}, text


def _extract_tags(metadata: dict, content: str) -> Set[str]:
    tags = set()
    meta_tags = metadata.get("tags", [])
    if isinstance(meta_tags, list):
        tags.update(str(t).strip() for t in meta_tags if str(t).strip())
    elif isinstance(meta_tags, str):
        tags.add(meta_tags.strip())
    for m in re.findall(r"#([\w가-힣]+(?:[_-][\w가-힣]+)*)", content):
        tags.add(m.strip())
    return tags


def _extract_keywords(text: str, top_k: int = 5) -> Set[str]:
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.S)
    cleaned = re.sub(r"`[^`]+`", " ", cleaned)
    cleaned = re.sub(r"\[\[[^\]]+\]\]", " ", cleaned)
    cleaned = re.sub(r"[^\w가-힣\s]", " ", cleaned)
    tokens = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}", cleaned, re.UNICODE)

    counts: Counter = Counter()
    for t in tokens:
        low = t.lower()
        if len(low) < 2 or low in EN_STOPWORDS or low in KO_STOPWORDS:
            continue
        counts[low] += 1
    return {w for w, _ in counts.most_common(top_k)}


def scan_vault(vault_path: Path, exclude_folders: List[str]) -> List[VaultNote]:
    """vault의 모든 마크다운 노트 스캔"""
    excludes = set(exclude_folders)
    notes = []

    for md_file in vault_path.rglob("*.md"):
        if any(ex in md_file.parts for ex in excludes):
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        metadata, content = _parse_frontmatter(text)
        title = metadata.get("title", md_file.stem)
        raw_links = [_normalize_key(m.split("|", 1)[0]) for m in LINK_PATTERN.findall(content) if m.strip()]

        aliases = {md_file.stem, _normalize_key(str(title))}
        meta_aliases = metadata.get("aliases") or metadata.get("alias") or []
        if isinstance(meta_aliases, list):
            aliases.update(str(a).strip() for a in meta_aliases if str(a).strip())
        elif isinstance(meta_aliases, str):
            aliases.add(meta_aliases.strip())

        tags = _extract_tags(metadata, content)

        notes.append(VaultNote(
            note_id=_normalize_key(str(md_file.relative_to(vault_path).with_suffix(""))),
            rel_path=md_file.relative_to(vault_path),
            title=str(title),
            content=content.strip(),
            aliases={a for a in aliases if a},
            tags={_normalize_key(t) for t in tags if t},
            raw_links=raw_links,
        ))
    return notes


def build_graph(notes: List[VaultNote]):
    """링크 그래프 구축 + 고립 노트 식별"""
    by_path = {n.note_id: n.note_id for n in notes}
    by_stem: Dict[str, Set[str]] = defaultdict(set)
    by_alias: Dict[str, Set[str]] = defaultdict(set)

    for n in notes:
        by_stem[n.rel_path.stem.lower()].add(n.note_id)
        for a in n.aliases:
            key = _normalize_key(a)
            if key:
                by_alias[key].add(n.note_id)

    inbound: Dict[str, Set[str]] = defaultdict(set)
    for n in notes:
        resolved = set()
        for raw in n.raw_links:
            key = _normalize_key(raw)
            if key in by_path:
                resolved.add(by_path[key])
            else:
                stem_hits = by_stem.get(key.rsplit("/", 1)[-1], set())
                if len(stem_hits) == 1:
                    resolved.update(stem_hits)
                else:
                    alias_hits = by_alias.get(key, set())
                    if len(alias_hits) == 1:
                        resolved.update(alias_hits)
        n.outbound = resolved
        for target in resolved:
            inbound[target].add(n.note_id)

    orphans = [n for n in notes if not n.outbound and not inbound.get(n.note_id)]
    return inbound, orphans


def recommend_links(source: VaultNote, notes_by_id: Dict[str, VaultNote],
                    top_k: int = 5, min_score: float = 1.0) -> List[str]:
    """키워드/태그 기반으로 관련 노트 추천"""
    source_tokens = set(source.keywords) | {_normalize_key(w) for w in re.findall(r"\w+", source.title.lower())}
    source_tags = set(source.tags)
    scored = []

    for nid, candidate in notes_by_id.items():
        if nid == source.note_id:
            continue
        kw_overlap = len(source_tokens & candidate.keywords)
        tag_overlap = len(source_tags & candidate.tags)
        title_overlap = len(source_tokens & {_normalize_key(w) for w in re.findall(r"\w+", candidate.title.lower())})
        score = kw_overlap * 4 + tag_overlap * 2 + title_overlap

        if score == 0:
            sim = SequenceMatcher(None, source.title.lower(), candidate.title.lower()).ratio()
            if sim < 0.2:
                continue
            score = sim

        if score < min_score:
            continue
        scored.append((score, candidate.rel_path.with_suffix("").as_posix()))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [target for _, target in scored[:top_k]]


def upsert_auto_block(text: str, summary: str, links: List[str]) -> str:
    """파일 끝에 자동 링크 블록 삽입/갱신"""
    if links:
        link_text = ", ".join(f"[[{lnk}]]" for lnk in links)
    else:
        link_text = "후보 없음"

    new_block = "\n".join([
        AUTO_BLOCK_START,
        f"**관련 노트:** {link_text}",
        AUTO_BLOCK_END,
    ])

    pattern = re.compile(rf"{re.escape(AUTO_BLOCK_START)}.*?{re.escape(AUTO_BLOCK_END)}", re.S)
    if pattern.search(text):
        return pattern.sub(new_block, text)

    body = text.rstrip()
    return f"{body}\n\n{new_block}\n" if body else f"{new_block}\n"


class AutoLinker:
    """Obsidian 자동 링크 관리"""

    def __init__(self, vault_path: str, exclude_folders: List[str] = None):
        self.vault_path = Path(vault_path)
        self.excludes = exclude_folders or [".obsidian", ".trash", "templates"]

    def run(self, apply: bool = False, top_k: int = 5, all_notes: bool = False):
        """
        자동 링크 실행

        Args:
            apply: True면 실제 파일에 [[링크]] 삽입, False면 미리보기만
            top_k: 추천 링크 수
            all_notes: True면 고립 노트뿐 아니라 전체 대상
        """
        console.print(f"[cyan]vault 스캔 중: {self.vault_path}[/cyan]")
        notes = scan_vault(self.vault_path, self.excludes)
        console.print(f"총 {len(notes)}개 노트 발견")

        notes_by_id = {n.note_id: n for n in notes}

        # 키워드 추출
        for n in notes:
            n.keywords = _extract_keywords(n.content, 5)

        # 그래프 구축
        inbound, orphans = build_graph(notes)
        targets = notes if all_notes else orphans

        console.print(f"고립 노트: {len(orphans)}개 / 전체: {len(notes)}개")
        if not targets:
            console.print("[green]고립 노트가 없습니다![/green]")
            return

        console.print(f"\n[bold]{'전체' if all_notes else '고립'} 노트 {len(targets)}개 처리 중...[/bold]\n")

        updated = 0
        table = Table(title="자동 링크 결과")
        table.add_column("노트", max_width=40)
        table.add_column("추천 링크", max_width=60)
        table.add_column("상태", style="green")

        for note in targets:
            links = recommend_links(note, notes_by_id, top_k=top_k)
            if not links:
                continue

            link_display = ", ".join(f"[[{l.split('/')[-1]}]]" for l in links[:3])
            if len(links) > 3:
                link_display += f" (+{len(links) - 3})"

            if apply:
                full_path = self.vault_path / note.rel_path
                original = full_path.read_text(encoding="utf-8")
                summary = f"{note.title} 관련 키워드: {', '.join(sorted(note.keywords)[:5])}"
                new_content = upsert_auto_block(original, summary, links)

                if new_content != original:
                    full_path.write_text(new_content, encoding="utf-8")
                    updated += 1
                    table.add_row(note.title[:40], link_display, "적용됨")
            else:
                table.add_row(note.title[:40], link_display, "미리보기")

        console.print(table)

        if apply:
            console.print(f"\n[bold green]{updated}개 노트에 [[링크]] 삽입 완료![/bold green]")
            console.print("[dim]Obsidian 그래프 뷰를 새로고침하면 연결이 보입니다.[/dim]")
        else:
            console.print(f"\n[yellow]미리보기 모드입니다. 실제 적용하려면 --apply 를 추가하세요.[/yellow]")

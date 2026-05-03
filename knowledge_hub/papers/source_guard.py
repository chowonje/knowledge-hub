from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.source_cleanup import (
    EXCLUDE_UNTIL_MANUAL_FIX,
    RELINK_TO_CANONICAL,
    REVIEWED_KEEP_CURRENT_SOURCE,
    build_source_cleanup_plan,
)
from knowledge_hub.papers.source_text import extract_pdf_text_excerpt, extract_salient_paper_text

_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
    "paper",
    "learning",
    "networks",
    "network",
}


def _normalize_title(value: Any) -> str:
    return " ".join(_TITLE_TOKEN_RE.findall(str(value or "").strip().lower()))


def _meaningful_title_tokens(title: str) -> list[str]:
    tokens = [token for token in _normalize_title(title).split() if token]
    meaningful = [token for token in tokens if len(token) >= 3 and token not in _TITLE_STOPWORDS]
    return meaningful or tokens[:2]


def _title_collision(title: str, candidate_title: str) -> tuple[bool, str]:
    left = _normalize_title(title)
    right = _normalize_title(candidate_title)
    if not left or not right:
        return False, ""
    if left == right:
        return True, "exact"
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if len(shorter) < 4:
        return False, ""
    if len(shorter.split()) == 1 and len(shorter) < 4:
        return False, ""
    if longer.startswith(shorter):
        return True, "prefix"
    return False, ""


def _preview_from_paths(*, pdf_path: str, text_path: str, max_chars: int = 1600) -> str:
    text_file = Path(str(text_path or "").strip()).expanduser() if str(text_path or "").strip() else None
    if text_file and text_file.exists():
        return extract_salient_paper_text(text_file.read_text(encoding="utf-8", errors="ignore"), max_chars=max_chars)
    return extract_pdf_text_excerpt(str(pdf_path or ""), max_chars=max_chars)


def _preview_matches_title(preview: str, title: str) -> bool:
    preview_norm = _normalize_title(preview)
    title_norm = _normalize_title(title)
    if not preview_norm or not title_norm:
        return False
    if title_norm in preview_norm:
        return True
    tokens = _meaningful_title_tokens(title)
    if not tokens:
        return False
    matched = sum(1 for token in tokens[:6] if token in preview_norm)
    required = 1 if len(tokens) <= 2 else 2
    return matched >= required


def find_title_collision_candidates(sqlite_db, *, paper_id: str, title: str, limit: int = 8) -> list[dict[str, Any]]:
    token = str(paper_id or "").strip()
    title_token = str(title or "").strip()
    if not title_token:
        return []
    candidates: list[dict[str, Any]] = []
    for row in list(sqlite_db.list_papers(limit=5000) or []):
        other_id = str(row.get("arxiv_id") or "").strip()
        other_title = str(row.get("title") or "").strip()
        if not other_id or other_id == token or not other_title:
            continue
        collided, match_type = _title_collision(title_token, other_title)
        if not collided:
            continue
        candidates.append(
            {
                "paperId": other_id,
                "title": other_title,
                "matchType": match_type,
                "pdfPath": str(row.get("pdf_path") or ""),
                "textPath": str(row.get("text_path") or ""),
            }
        )
    candidates.sort(key=lambda item: (0 if item.get("matchType") == "exact" else 1, len(str(item.get("title") or ""))))
    return candidates[: max(1, int(limit or 8))]


def inspect_title_collisions(sqlite_db, *, paper_id: str, title: str) -> dict[str, Any]:
    candidates = find_title_collision_candidates(sqlite_db, paper_id=paper_id, title=title)
    if not candidates:
        return {
            "status": "clear",
            "decision": "allow_clean",
            "reason": "no title collision candidates detected",
            "candidates": [],
            "signals": {"candidateCount": 0},
        }
    return {
        "status": "suspected",
        "decision": "pending_review",
        "reason": "title collision candidates detected; defer to downloaded source preview",
        "candidates": candidates,
        "signals": {"candidateCount": len(candidates)},
    }


def merge_source_guard(existing: Any, payload: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(existing or {})
    if not payload:
        return merged
    merged.update(dict(payload))
    return merged


def stage_source_guard(sqlite_db, *, paper_id: str, title: str, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    return merge_source_guard(
        existing,
        inspect_title_collisions(sqlite_db, paper_id=str(paper_id or "").strip(), title=str(title or "").strip()),
    )


def finalize_source_guard(
    sqlite_db,
    *,
    paper_id: str,
    title: str,
    pdf_path: str,
    text_path: str,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token = str(paper_id or "").strip()
    title_token = str(title or "").strip()
    current = dict(existing or {})
    candidates = list(current.get("candidates") or find_title_collision_candidates(sqlite_db, paper_id=token, title=title_token))
    preview = _preview_from_paths(pdf_path=str(pdf_path or ""), text_path=str(text_path or ""))
    preview_excerpt = str(preview or "")[:400]
    matched_candidate_ids = [
        str(item.get("paperId") or "")
        for item in candidates
        if _preview_matches_title(preview, str(item.get("title") or ""))
    ]

    if token in RELINK_TO_CANONICAL or token in REVIEWED_KEEP_CURRENT_SOURCE or token in EXCLUDE_UNTIL_MANUAL_FIX:
        decision = build_source_cleanup_plan(
            [
                {
                    "paperId": token,
                    "title": title_token,
                    "oldPdfPath": str(pdf_path or ""),
                    "oldTextPath": str(text_path or ""),
                    "recommendedParser": "raw",
                }
            ],
            sqlite_db=sqlite_db,
        )[0]
        action = str(decision.get("action") or "")
        if action == "relink_to_canonical" and str(decision.get("status") or "") == "resolved":
            return {
                "status": "reviewed",
                "decision": "relink_to_canonical",
                "reason": str(decision.get("resolutionReason") or ""),
                "candidates": candidates,
                "canonicalPaperId": str(decision.get("canonicalPaperId") or ""),
                "canonicalTitle": str(decision.get("canonicalTitle") or ""),
                "newPdfPath": str(decision.get("newPdfPath") or ""),
                "newTextPath": str(decision.get("newTextPath") or ""),
                "previewExcerpt": preview_excerpt,
                "signals": {
                    "candidateCount": len(candidates),
                    "matchedCandidateIds": matched_candidate_ids,
                    "previewMatchesImportedTitle": _preview_matches_title(preview, title_token),
                },
            }
        if action == "keep_current_source":
            return {
                "status": "reviewed",
                "decision": "allow_duplicate" if candidates else "allow_clean",
                "reason": str(decision.get("resolutionReason") or ""),
                "candidates": candidates,
                "previewExcerpt": preview_excerpt,
                "signals": {
                    "candidateCount": len(candidates),
                    "matchedCandidateIds": matched_candidate_ids,
                    "previewMatchesImportedTitle": _preview_matches_title(preview, title_token),
                },
            }
        if action in {"exclude_until_manual_fix", "manual_review_required"}:
            return {
                "status": "blocked",
                "decision": "block_manual_review",
                "reason": str(decision.get("resolutionReason") or "source guard requires manual review"),
                "candidates": candidates,
                "previewExcerpt": preview_excerpt,
                "signals": {
                    "candidateCount": len(candidates),
                    "matchedCandidateIds": matched_candidate_ids,
                    "previewMatchesImportedTitle": _preview_matches_title(preview, title_token),
                },
            }

    if not candidates:
        return {
            "status": "clear",
            "decision": "allow_clean",
            "reason": "no title collision candidates detected",
            "candidates": [],
            "previewExcerpt": preview_excerpt,
            "signals": {"candidateCount": 0},
        }
    if not preview_excerpt:
        return {
            "status": "blocked",
            "decision": "block_manual_review",
            "reason": "title collision candidates exist but no source preview was available",
            "candidates": candidates,
            "previewExcerpt": "",
            "signals": {"candidateCount": len(candidates), "matchedCandidateIds": matched_candidate_ids},
        }
    if _preview_matches_title(preview, title_token) or matched_candidate_ids:
        return {
            "status": "reviewed",
            "decision": "allow_duplicate",
            "reason": "source preview matches the imported title closely enough",
            "candidates": candidates,
            "previewExcerpt": preview_excerpt,
            "signals": {
                "candidateCount": len(candidates),
                "matchedCandidateIds": matched_candidate_ids,
                "previewMatchesImportedTitle": _preview_matches_title(preview, title_token),
            },
        }
    return {
        "status": "blocked",
        "decision": "block_manual_review",
        "reason": "title collision candidates exist but downloaded source preview does not match the imported title",
        "candidates": candidates,
        "previewExcerpt": preview_excerpt,
        "signals": {
            "candidateCount": len(candidates),
            "matchedCandidateIds": matched_candidate_ids,
            "previewMatchesImportedTitle": False,
        },
    }


def review_downloaded_source(
    sqlite_db,
    *,
    paper_id: str,
    title: str,
    pdf_path: str,
    text_path: str,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    guard = stage_source_guard(sqlite_db, paper_id=paper_id, title=title, existing=existing)
    guard = merge_source_guard(
        guard,
        finalize_source_guard(
            sqlite_db,
            paper_id=str(paper_id or "").strip(),
            title=str(title or "").strip(),
            pdf_path=str(pdf_path or ""),
            text_path=str(text_path or ""),
            existing=guard,
        ),
    )
    return {
        "guard": guard,
        "blocked": str(guard.get("decision") or "") == "block_manual_review",
        "finalPdfPath": str(guard.get("newPdfPath") or pdf_path or ""),
        "finalTextPath": str(guard.get("newTextPath") or text_path or ""),
    }


__all__ = [
    "find_title_collision_candidates",
    "finalize_source_guard",
    "inspect_title_collisions",
    "merge_source_guard",
    "review_downloaded_source",
    "stage_source_guard",
]

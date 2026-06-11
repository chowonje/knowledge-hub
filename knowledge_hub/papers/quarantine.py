"""Surface-level quarantine for wrong-document papers.

The 2026-06-11 parsed-store content-identity audit
(parsed-store-content-identity-audit-20260611) confirmed 13 registered papers
whose on-disk PDF / parsed text is a different document than their metadata.
Until those sources are re-acquired and re-parsed, ask/search/compare surfaces
must not select or cite them.

2026-06-11 repair tranche: the 12 arXiv-backed ids were re-acquired from
arxiv.org, passed ``enforce_parse_identity`` (watermark/title gate), had their
contaminated parsed/derived/vector layers purged and rebuilt, and were
store-verified — they are lifted from the quarantine set.
``Gemini_Embedding_Generalizable_b5cf39ed`` remains quarantined: its registered
pdf_path points into the iCloud vault (off-limits) and re-registration to a
``~/.khub/papers/`` location is a pending manual product decision.

Quarantine blocks exposure only: stored rows, parsed artifacts, vectors, and
derived cards are left untouched so the repair flow can re-acquire and verify
them. Membership is keyed by the registered paper id in any of its runtime
token forms (bare id, ``paper:<id>`` document id, ``paper_<id>_<n>`` chunk id).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

QUARANTINE_AUDIT_REF = "parsed-store-content-identity-audit-20260611"
QUARANTINE_REASON_CODE = "quarantined_wrong_document"

QUARANTINED_PAPER_IDS: frozenset[str] = frozenset(
    {
        "Gemini_Embedding_Generalizable_b5cf39ed",
    }
)

_CHUNK_DOC_ID_RE = re.compile(r"^paper_(?P<paper_id>.+?)_(?P<chunk>\d+)$")


class QuarantinedPaperTargetError(RuntimeError):
    """Raised when an explicitly requested ask/compare target is quarantined."""

    def __init__(self, paper_ids: Iterable[str]):
        self.paper_ids = sorted({str(item).strip() for item in paper_ids if str(item).strip()})
        super().__init__(f"{QUARANTINE_REASON_CODE}: {','.join(self.paper_ids)}")


def normalize_paper_id_token(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    if token.startswith("paper:"):
        token = token[len("paper:") :].strip()
    match = _CHUNK_DOC_ID_RE.match(token)
    if match:
        token = match.group("paper_id")
    return token


def is_quarantined_paper(value: Any) -> bool:
    return normalize_paper_id_token(value) in QUARANTINED_PAPER_IDS


def quarantined_paper_ids(values: Iterable[Any]) -> list[str]:
    flagged: list[str] = []
    for value in values or []:
        token = normalize_paper_id_token(value)
        if token in QUARANTINED_PAPER_IDS and token not in flagged:
            flagged.append(token)
    return flagged


def resolve_quarantined_paper_targets(
    resolved_paper_ids: Iterable[Any],
    *,
    compare: bool = False,
) -> list[str]:
    """Drop quarantined ids from an explicit target list, failing closed.

    Raises :class:`QuarantinedPaperTargetError` when every target is
    quarantined, or — for compare-shaped requests — when any member of the
    target set is quarantined (silently substituting a different paper into a
    comparison would answer a different question than the user asked).
    """

    ids = [str(item).strip() for item in (resolved_paper_ids or []) if str(item or "").strip()]
    flagged = quarantined_paper_ids(ids)
    if not flagged:
        return ids
    clean = [item for item in ids if not is_quarantined_paper(item)]
    if compare or not clean:
        raise QuarantinedPaperTargetError(flagged)
    return clean


def _search_result_paper_token(result: Any) -> str:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("paper_id", "arxiv_id", "doc_id"):
            token = normalize_paper_id_token(metadata.get(key))
            if token:
                return token
    return normalize_paper_id_token(getattr(result, "document_id", ""))


def filter_quarantined_search_results(results: Iterable[Any]) -> tuple[list[Any], list[str]]:
    """Remove quarantined-paper hits from retrieval results.

    Non-paper results (vault/web/notes) pass through untouched because their
    tokens never resolve into the deny-list.
    """

    kept: list[Any] = []
    dropped: list[str] = []
    for result in results or []:
        token = _search_result_paper_token(result)
        if token in QUARANTINED_PAPER_IDS:
            if token not in dropped:
                dropped.append(token)
            continue
        kept.append(result)
    return kept, dropped


def filter_quarantined_cards(
    cards: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for card in cards or []:
        if not isinstance(card, dict):
            kept.append(card)
            continue
        token = normalize_paper_id_token(card.get("paper_id") or card.get("arxiv_id"))
        if token in QUARANTINED_PAPER_IDS:
            if token not in dropped:
                dropped.append(token)
            continue
        kept.append(card)
    return kept, dropped


__all__ = [
    "QUARANTINE_AUDIT_REF",
    "QUARANTINE_REASON_CODE",
    "QUARANTINED_PAPER_IDS",
    "QuarantinedPaperTargetError",
    "filter_quarantined_cards",
    "filter_quarantined_search_results",
    "is_quarantined_paper",
    "normalize_paper_id_token",
    "quarantined_paper_ids",
    "resolve_quarantined_paper_targets",
]

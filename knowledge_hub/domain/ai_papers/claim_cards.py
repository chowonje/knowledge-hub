from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
import re
from typing import Any

from knowledge_hub.ai.ask_v2_support import clean_text, query_terms, text_overlap, trim_text
from knowledge_hub.knowledge.claim_normalization import _materially_different


_ALIAS_SEEDS: tuple[tuple[str, str, str, str, str], ...] = (
    ("dataset", "natural questions", "Natural Questions", "Natural Questions", ""),
    ("dataset", "nq", "Natural Questions", "Natural Questions", ""),
    ("dataset", "triviaqa", "TriviaQA", "TriviaQA", ""),
    ("dataset", "hotpotqa", "HotpotQA", "HotpotQA", ""),
    ("dataset", "squad", "SQuAD", "SQuAD", ""),
    ("dataset", "squad 2.0", "SQuAD 2.0", "SQuAD", "2.0"),
    ("dataset", "squad v2", "SQuAD 2.0", "SQuAD", "2.0"),
    ("dataset", "squad2", "SQuAD 2.0", "SQuAD", "2.0"),
    ("dataset", "mmlu", "MMLU", "MMLU", ""),
    ("dataset", "mmlu-pro", "MMLU-Pro", "MMLU", "Pro"),
    ("dataset", "wmt 2014 en-de", "WMT 2014 en-de", "WMT 2014 en-de", ""),
    ("dataset", "wmt14 en-de", "WMT 2014 en-de", "WMT 2014 en-de", ""),
    ("metric", "f1", "f1", "", ""),
    ("metric", "f1 score", "f1", "", ""),
    ("metric", "f1-score", "f1", "", ""),
    ("metric", "accuracy", "accuracy", "", ""),
    ("metric", "acc", "accuracy", "", ""),
    ("metric", "top-1 accuracy", "accuracy", "", ""),
    ("metric", "exact match", "exact_match", "", ""),
    ("metric", "em", "exact_match", "", ""),
    ("metric", "bleu", "BLEU", "", ""),
    ("metric", "bleu-4", "BLEU", "", ""),
    ("metric", "sacrebleu", "BLEU", "", ""),
    ("task", "question answering", "question_answering", "", ""),
    ("task", "qa", "question_answering", "", ""),
    ("task", "open-domain question answering", "open_domain_qa", "", ""),
    ("task", "open-domain qa", "open_domain_qa", "", ""),
    ("task", "retrieval augmented generation", "retrieval_augmented_generation", "", ""),
    ("task", "rag", "retrieval_augmented_generation", "", ""),
    ("comparator", "baseline", "baseline", "", ""),
    ("comparator", "seq2seq baseline", "seq2seq baseline", "", ""),
    ("comparator", "previous sota", "previous SOTA", "", ""),
    ("comparator", "previous state of the art", "previous SOTA", "", ""),
    ("comparator", "sota", "SOTA", "", ""),
    ("comparator", "vanilla transformer", "Transformer", "", ""),
    ("comparator", "standard transformer", "Transformer", "", ""),
    ("comparator", "base transformer", "Transformer", "", ""),
)


def _stable_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _clean_list(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = clean_text(value)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
    return out


def _latest_normalizations(sqlite_db: Any, claim_ids: list[str]) -> dict[str, dict[str, Any]]:
    rows = sqlite_db.list_claim_normalizations(
        claim_ids=claim_ids,
        limit=max(10, len(claim_ids) * 4),
    )
    result: dict[str, dict[str, Any]] = {}
    ranking = {"normalized": 3, "partial": 2, "failed": 1}
    for row in rows:
        claim_id = clean_text(row.get("claim_id"))
        if not claim_id:
            continue
        existing = result.get(claim_id)
        if existing is None:
            result[claim_id] = dict(row)
            continue
        current_rank = ranking.get(clean_text(row.get("status")).casefold(), 0)
        existing_rank = ranking.get(clean_text(existing.get("status")).casefold(), 0)
        if current_rank > existing_rank or (
            current_rank == existing_rank and clean_text(row.get("updated_at")) > clean_text(existing.get("updated_at"))
        ):
            result[claim_id] = dict(row)
    return result


def _derive_claim_type(*, claim_text: str, predicate: str, role: str, normalization: dict[str, Any]) -> str:
    role_token = clean_text(role).casefold()
    claim_body = f"{clean_text(predicate)} {clean_text(claim_text)}".casefold()
    if role_token in {"comparison"} or clean_text(normalization.get("comparator")):
        return "comparison"
    if role_token in {"limitation", "when_not_to_use"} or clean_text(normalization.get("limitation_text")):
        return "limitation"
    if role_token in {"scope"} or clean_text(normalization.get("scope_text")) or clean_text(normalization.get("condition_text")):
        return "scope"
    if role_token in {"method", "implementation"} or any(token in claim_body for token in ("method", "architecture", "pipeline", "approach", "module", "implemented", "introduce", "propose")):
        return "method"
    if any(token in claim_body for token in ("define", "definition", "refers to", "is a", "means")):
        return "definition"
    if any(token in claim_body for token in ("theorem", "lemma", "proof", "bound")):
        return "theoretical"
    if role_token in {"result", "metric", "dataset", "supporting"} or clean_text(normalization.get("metric")) or normalization.get("result_value_numeric") is not None:
        return "empirical"
    return "unknown"


def _claim_status(*, confidence: float, evidence_strength: str, anchor_count: int) -> str:
    if anchor_count <= 0:
        return "unsupported"
    if confidence < 0.45:
        return "weak"
    if clean_text(evidence_strength).casefold() == "weak":
        return "weak"
    return "accepted"


def _quality_flag(*, normalization: dict[str, Any], anchor_count: int, status: str) -> str:
    if status in {"contradicted", "unsupported"}:
        return "needs_review"
    normalized = clean_text(normalization.get("status")).casefold() == "normalized"
    if normalized and anchor_count > 0 and status == "accepted":
        return "ok"
    if anchor_count > 0:
        return "needs_review"
    return "unscored"


def _summary_text(*, claim_text: str, normalization: dict[str, Any], anchors: list[dict[str, Any]], source_title: str) -> str:
    parts: list[str] = []
    if clean_text(claim_text):
        parts.append(clean_text(claim_text))
    normalized_bits = [
        clean_text(normalization.get("task")),
        clean_text(normalization.get("dataset")),
        clean_text(normalization.get("metric")),
        clean_text(normalization.get("result_value_text")),
    ]
    normalized_bits = [item for item in normalized_bits if item]
    if normalized_bits:
        parts.append(", ".join(normalized_bits[:4]))
    if source_title:
        parts.append(source_title)
    anchor_excerpt = next((clean_text(anchor.get("excerpt")) for anchor in anchors if clean_text(anchor.get("excerpt"))), "")
    if anchor_excerpt:
        parts.append(anchor_excerpt)
    return trim_text(" | ".join(parts), max_chars=260)


def _matched_entity_ids(claim: dict[str, Any]) -> list[str]:
    values = [
        claim.get("subject_entity_id"),
        claim.get("object_entity_id"),
    ]
    return _clean_list(values)


def _build_search_text(*, claim_text: str, normalization: dict[str, Any], source_title: str, anchors: list[dict[str, Any]]) -> str:
    parts = [
        claim_text,
        clean_text(normalization.get("task")),
        clean_text(normalization.get("dataset")),
        clean_text(normalization.get("metric")),
        clean_text(normalization.get("comparator")),
        clean_text(normalization.get("result_value_text")),
        clean_text(normalization.get("scope_text")),
        clean_text(normalization.get("condition_text")),
        source_title,
    ]
    for anchor in anchors[:3]:
        parts.append(clean_text(anchor.get("excerpt")))
    return clean_text(" ".join(part for part in parts if clean_text(part)))


def _snapshot_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_version_from_text(value: str) -> str:
    match = re.search(r"\b(v?\d+(?:\.\d+)?)\b", clean_text(value), flags=re.IGNORECASE)
    return clean_text(match.group(1)) if match else ""


def _trust_level(*, origin: str, status: str, evidence_strength: str) -> str:
    if origin == "synthetic_fallback":
        return "low"
    if origin == "project_ephemeral":
        return "medium"
    if status in {"unsupported", "contradicted", "weak"} or clean_text(evidence_strength).casefold() == "weak":
        return "medium"
    return "high"


class ClaimCardAlignmentService:
    @staticmethod
    def canonical_frame(card: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            clean_text(card.get("task_canonical") or card.get("task")),
            clean_text(card.get("dataset_canonical") or card.get("dataset")),
            clean_text(card.get("metric_canonical") or card.get("metric")),
            clean_text(card.get("comparator_canonical") or card.get("comparator")),
        )

    @staticmethod
    def frame_key(card: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            clean_text(card.get("task")),
            clean_text(card.get("dataset")),
            clean_text(card.get("metric")),
            clean_text(card.get("comparator")),
        )

    @classmethod
    def group_key(cls, card: dict[str, Any]) -> str:
        frame = cls.canonical_frame(card)
        condition = clean_text(card.get("condition_text")).casefold()
        dataset_family = clean_text(card.get("dataset_family"))
        dataset_version = clean_text(card.get("dataset_version"))
        return "||".join([*frame, condition, dataset_family, dataset_version])

    @staticmethod
    def _numeric_delta(left: dict[str, Any], right: dict[str, Any]) -> float | None:
        left_value = left.get("result_value_numeric")
        right_value = right.get("result_value_numeric")
        if left_value is None or right_value is None:
            return None
        try:
            return float(left_value) - float(right_value)
        except Exception:
            return None

    @classmethod
    def _alignment_type(cls, left: dict[str, Any], right: dict[str, Any]) -> str:
        if _materially_different(clean_text(left.get("condition_text")), clean_text(right.get("condition_text"))):
            return "condition_split"
        left_family = clean_text(left.get("dataset_family"))
        right_family = clean_text(right.get("dataset_family"))
        left_version = clean_text(left.get("dataset_version"))
        right_version = clean_text(right.get("dataset_version"))
        if left_family and right_family and left_family == right_family and left_version != right_version:
            return "family_related"
        left_direction = clean_text(left.get("result_direction")).casefold()
        right_direction = clean_text(right.get("result_direction")).casefold()
        if left_direction and right_direction and left_direction not in {"", "unknown"} and right_direction not in {"", "unknown"} and left_direction != right_direction:
            return "conflict"
        delta = cls._numeric_delta(left, right)
        if delta is not None and not math.isclose(delta, 0.0, abs_tol=1e-9):
            return "aligned"
        return "aligned"

    @classmethod
    def build_alignment_refs(cls, card: dict[str, Any], peers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frame = cls.frame_key(card)
        canonical_frame = cls.canonical_frame(card)
        if not any(frame):
            return []
        source_diversity = len({clean_text(item.get("source_id")) for item in peers if clean_text(item.get("source_id"))})
        refs: list[dict[str, Any]] = []
        for order, peer in enumerate(peers, 1):
            peer_id = clean_text(peer.get("claim_card_id"))
            if not peer_id or peer_id == clean_text(card.get("claim_card_id")):
                continue
            alignment_type = cls._alignment_type(card, peer)
            refs.append(
                {
                    "aligned_claim_card_id": peer_id,
                    "task": frame[0],
                    "dataset": frame[1],
                    "metric": frame[2],
                    "comparator": frame[3],
                    "task_canonical": canonical_frame[0],
                    "dataset_canonical": canonical_frame[1],
                    "dataset_family": clean_text(card.get("dataset_family")),
                    "dataset_version": clean_text(card.get("dataset_version")),
                    "metric_canonical": canonical_frame[2],
                    "comparator_canonical": canonical_frame[3],
                    "condition_text": clean_text(card.get("condition_text")),
                    "group_key": cls.group_key(card),
                    "family_relation_note": (
                        "same_dataset_family_different_version"
                        if alignment_type == "family_related"
                        else ""
                    ),
                    "alignment_type": alignment_type,
                    "value_delta": cls._numeric_delta(card, peer),
                    "source_diversity": source_diversity,
                    "evidence_order": order,
                }
            )
        return refs

    @classmethod
    def group_claim_cards(cls, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for card in cards:
            frame = cls.canonical_frame(card)
            if any(frame):
                grouped[cls.group_key(card)].append(card)
        summaries: list[dict[str, Any]] = []
        for group_key, items in grouped.items():
            first = items[0]
            frame = cls.frame_key(first)
            canonical_frame = cls.canonical_frame(first)
            directions = {clean_text(item.get("result_direction")).casefold() for item in items if clean_text(item.get("result_direction"))}
            numeric_values = [item.get("result_value_numeric") for item in items if item.get("result_value_numeric") is not None]
            evidence_strengths = [clean_text(item.get("evidence_strength")).casefold() or "weak" for item in items]
            summaries.append(
                {
                    "groupKey": group_key,
                    "frame": {
                        "task": frame[0],
                        "dataset": frame[1],
                        "metric": frame[2],
                        "comparator": frame[3],
                    },
                    "canonicalFrame": {
                        "task": canonical_frame[0],
                        "dataset": canonical_frame[1],
                        "datasetFamily": clean_text(first.get("dataset_family")),
                        "datasetVersion": clean_text(first.get("dataset_version")),
                        "metric": canonical_frame[2],
                        "comparator": canonical_frame[3],
                    },
                    "claimCardIds": [clean_text(item.get("claim_card_id")) for item in items if clean_text(item.get("claim_card_id"))],
                    "supportingClaimCount": sum(1 for item in items if clean_text(item.get("status")).casefold() == "accepted"),
                    "conflictingClaimCount": 1 if len({direction for direction in directions if direction not in {"", "unknown"}}) > 1 else 0,
                    "conditionText": clean_text(first.get("condition_text")),
                    "valueSpread": (
                        {"min": min(float(value) for value in numeric_values), "max": max(float(value) for value in numeric_values)}
                        if numeric_values
                        else {}
                    ),
                    "sourceDiversity": len({clean_text(item.get("source_id")) for item in items if clean_text(item.get("source_id"))}),
                    "evidenceOrdering": sorted(evidence_strengths, key=lambda value: {"strong": 0, "medium": 1, "weak": 2}.get(value, 3)),
                }
            )
        summaries.sort(key=lambda item: (-len(item.get("claimCardIds") or []), item["frame"].get("task") or "", item["frame"].get("dataset") or ""))
        return summaries


def build_project_claim_cards(*, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claim_cards: list[dict[str, Any]] = []
    type_map = {
        "module": "definition",
        "symbol_owner": "definition",
        "call_flow": "method",
        "integration_boundary": "scope",
    }
    for card in cards:
        source_card_id = clean_text(card.get("card_id"))
        relative_path = clean_text(card.get("relative_path"))
        source_id = relative_path
        anchors = list(card.get("anchors") or [])
        for rank, anchor in enumerate(anchors, 1):
            role = clean_text(anchor.get("evidence_role"))
            excerpt = clean_text(anchor.get("excerpt"))
            if not excerpt:
                continue
            claim_cards.append(
                {
                    "claim_card_id": f"claim-card-v1:project:{source_card_id}:{role}",
                    "claim_id": f"project-claim:{relative_path}:{role}",
                    "source_kind": "project",
                    "source_id": source_id,
                    "document_id": clean_text(anchor.get("file_path") or relative_path),
                    "paper_id": "",
                    "source_card_id": source_card_id,
                    "claim_text": excerpt,
                    "claim_type": type_map.get(role, "unknown"),
                    "status": "accepted",
                    "summary_text": trim_text(f"{relative_path} | {excerpt}", max_chars=260),
                    "scope_text": clean_text(card.get("file_role_core")) if role == "integration_boundary" else "",
                    "condition_text": "",
                    "limitation_text": "",
                    "negative_scope_text": "",
                    "task": "",
                    "dataset": "",
                    "metric": "",
                    "comparator": "",
                    "result_direction": "unknown",
                    "result_value_text": "",
                    "result_value_numeric": None,
                    "evidence_strength": "medium",
                    "evidence_anchor_ids": _clean_list([anchor.get("anchor_id")]),
                    "section_paths": _clean_list([anchor.get("section_path")]),
                    "matched_entity_ids": [],
                    "search_text": clean_text(f"{relative_path} {role} {excerpt} {card.get('search_text')}"),
                    "quality_flag": "ok" if clean_text(excerpt) else "needs_review",
                    "confidence": min(0.99, max(0.2, _stable_float(anchor.get("score")))),
                    "origin": "project_ephemeral",
                    "trust_level": "medium",
                    "built_at": _snapshot_now(),
                    "source_updated_at_snapshot": "",
                    "normalization_updated_at_snapshot": "",
                    "task_canonical": "",
                    "dataset_canonical": "",
                    "dataset_family": "",
                    "dataset_version": "",
                    "metric_canonical": "",
                    "comparator_canonical": "",
                    "updated_at": "",
                    "rank": rank,
                    "role": role,
                    "selection_reason": role,
                    "anchors": [dict(anchor)],
                }
            )
    return claim_cards


class ClaimCardBuilder:
    def __init__(self, sqlite_db: Any):
        self.sqlite_db = sqlite_db
        self.alignment = ClaimCardAlignmentService()
        self._alias_seed_attempted = False

    def _ensure_aliases_seeded(self) -> None:
        if self._alias_seed_attempted:
            return
        self._alias_seed_attempted = True
        upsert_alias = getattr(self.sqlite_db, "upsert_normalization_alias", None)
        if not callable(upsert_alias):
            return
        try:
            self._seed_aliases(upsert_alias)
        except (AttributeError, NotImplementedError):
            # Alias-store capability is optional for read-mostly ask_v2 paths.
            return

    @staticmethod
    def _seed_aliases(upsert_alias: Any) -> None:
        for alias_type, alias, canonical, dataset_family, dataset_version in _ALIAS_SEEDS:
            upsert_alias(
                alias_type=alias_type,
                alias=alias,
                canonical=canonical,
                dataset_family=dataset_family,
                dataset_version=dataset_version,
            )

    def _alias_rows(self, alias_type: str) -> dict[str, dict[str, Any]]:
        list_aliases = getattr(self.sqlite_db, "list_normalization_aliases", None)
        if not callable(list_aliases):
            return {}
        try:
            rows = list_aliases(alias_type=alias_type)
        except (AttributeError, NotImplementedError):
            return {}
        return {
            clean_text(row.get("alias")).casefold(): dict(row)
            for row in rows
            if clean_text(row.get("alias"))
        }

    def _canonical_term(self, *, alias_type: str, value: str) -> tuple[str, str, str]:
        token = clean_text(value)
        if not token:
            return "", "", ""
        rows = self._alias_rows(alias_type)
        lowered = token.casefold()
        row = rows.get(lowered)
        if row:
            return (
                clean_text(row.get("canonical")),
                clean_text(row.get("dataset_family")),
                clean_text(row.get("dataset_version")),
            )
        for alias, payload in rows.items():
            if alias and (alias in lowered or lowered in alias):
                return (
                    clean_text(payload.get("canonical")),
                    clean_text(payload.get("dataset_family")),
                    clean_text(payload.get("dataset_version")),
                )
        if alias_type == "dataset":
            version = _dataset_version_from_text(token)
            return token, token if not version else token.split(version, 1)[0].strip(" -_"), version
        return token, "", ""

    def _canonical_fields(self, normalization: dict[str, Any]) -> dict[str, str]:
        task_canonical, _, _ = self._canonical_term(alias_type="task", value=clean_text(normalization.get("task")))
        dataset_canonical, dataset_family, dataset_version = self._canonical_term(alias_type="dataset", value=clean_text(normalization.get("dataset")))
        metric_canonical, _, _ = self._canonical_term(alias_type="metric", value=clean_text(normalization.get("metric")))
        comparator_canonical, _, _ = self._canonical_term(alias_type="comparator", value=clean_text(normalization.get("comparator")))
        return {
            "task_canonical": task_canonical or clean_text(normalization.get("task")),
            "dataset_canonical": dataset_canonical or clean_text(normalization.get("dataset")),
            "dataset_family": dataset_family,
            "dataset_version": dataset_version,
            "metric_canonical": metric_canonical or clean_text(normalization.get("metric")),
            "comparator_canonical": comparator_canonical or clean_text(normalization.get("comparator")),
        }

    def _latest_normalization_snapshot(self, normalizations: dict[str, dict[str, Any]]) -> str:
        latest = ""
        for row in normalizations.values():
            updated_at = clean_text(row.get("updated_at"))
            if updated_at > latest:
                latest = updated_at
        return latest

    def _needs_rebuild_for_source_card(
        self,
        *,
        source_kind: str,
        source_card: dict[str, Any],
        stored_cards: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        normalizations: dict[str, dict[str, Any]],
    ) -> bool:
        if not stored_cards:
            return True
        source_snapshot = clean_text(source_card.get("updated_at"))
        normalization_snapshot = self._latest_normalization_snapshot(normalizations)
        if any(clean_text(item.get("source_updated_at_snapshot")) < source_snapshot for item in stored_cards):
            return True
        if any(clean_text(item.get("normalization_updated_at_snapshot")) < normalization_snapshot for item in stored_cards):
            return True
        if rows and any(clean_text(item.get("origin")) != "extracted" for item in stored_cards):
            return True
        if not rows and any(clean_text(item.get("origin")) == "extracted" for item in stored_cards):
            return True
        stored_ids = {clean_text(item.get("claim_id")) for item in stored_cards if clean_text(item.get("claim_id"))}
        current_ids = {clean_text(item.get("claim", {}).get("claim_id")) for item in rows if clean_text(item.get("claim", {}).get("claim_id"))}
        return bool(current_ids and stored_ids != current_ids)

    def _claim_rows(self, *, source_kind: str, source_card_id: str, source_card: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        refs: list[dict[str, Any]] = []
        if source_kind == "paper":
            refs = list(self.sqlite_db.list_paper_card_claim_refs_v2(card_id=source_card_id))
        elif source_kind == "web":
            refs = list(self.sqlite_db.list_web_card_claim_refs_v2(card_id=source_card_id))
        elif source_kind == "vault":
            refs = list(self.sqlite_db.list_vault_card_claim_refs_v2(card_id=source_card_id))
        claim_ids = [clean_text(ref.get("claim_id")) for ref in refs if clean_text(ref.get("claim_id"))]
        normalizations = _latest_normalizations(self.sqlite_db, claim_ids)
        claims: dict[str, dict[str, Any]] = {}
        for claim_id in claim_ids:
            claim = self.sqlite_db.get_claim(claim_id) or {}
            if claim:
                claims[claim_id] = dict(claim)
        anchors_by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for claim_id in claim_ids:
            if source_kind == "paper":
                anchors = self.sqlite_db.list_evidence_anchors_v2(card_id=source_card_id, claim_ids=[claim_id])
            elif source_kind == "web":
                anchors = self.sqlite_db.list_web_evidence_anchors_v2(card_id=source_card_id, claim_ids=[claim_id])
            else:
                anchors = self.sqlite_db.list_vault_evidence_anchors_v2(card_id=source_card_id, claim_ids=[claim_id])
            anchors_by_claim[claim_id].extend(dict(item) for item in anchors)
        rows: list[dict[str, Any]] = []
        for ref in refs:
            claim_id = clean_text(ref.get("claim_id"))
            claim = claims.get(claim_id)
            if not claim:
                continue
            rows.append(
                {
                    "ref": dict(ref),
                    "claim": claim,
                    "normalization": dict(normalizations.get(claim_id) or {}),
                    "anchors": list(anchors_by_claim.get(claim_id) or []),
                    "source_card": dict(source_card),
                }
            )
        return rows, normalizations, anchors_by_claim

    def _build_source_claim_card(self, *, source_kind: str, row: dict[str, Any]) -> dict[str, Any]:
        ref = dict(row.get("ref") or {})
        claim = dict(row.get("claim") or {})
        normalization = dict(row.get("normalization") or {})
        anchors = list(row.get("anchors") or [])
        source_card = dict(row.get("source_card") or {})
        claim_id = clean_text(claim.get("claim_id"))
        source_id = ""
        document_id = ""
        paper_id = ""
        if source_kind == "paper":
            source_id = clean_text(source_card.get("paper_id"))
            document_id = clean_text(source_card.get("paper_id")) and f"paper:{clean_text(source_card.get('paper_id'))}" or ""
            paper_id = clean_text(source_card.get("paper_id"))
        elif source_kind == "web":
            source_id = clean_text(source_card.get("canonical_url") or source_card.get("document_id"))
            document_id = clean_text(source_card.get("document_id"))
        else:
            source_id = clean_text(source_card.get("note_id"))
            document_id = clean_text(source_card.get("note_id"))
        claim_text = clean_text(claim.get("claim_text"))
        evidence_strength = clean_text(normalization.get("evidence_strength") or "weak")
        claim_type = _derive_claim_type(
            claim_text=claim_text,
            predicate=clean_text(claim.get("predicate")),
            role=clean_text(ref.get("role")),
            normalization=normalization,
        )
        confidence = max(_stable_float(ref.get("confidence")), _stable_float(claim.get("confidence")))
        status = _claim_status(confidence=confidence, evidence_strength=evidence_strength, anchor_count=len(anchors))
        limitation_text = clean_text(normalization.get("limitation_text"))
        canonical = self._canonical_fields(normalization)
        normalization_snapshot = clean_text(normalization.get("updated_at"))
        return {
            "claim_card_id": f"claim-card-v1:{source_kind}:{claim_id}",
            "claim_id": claim_id,
            "source_kind": source_kind,
            "source_id": source_id,
            "document_id": document_id,
            "paper_id": paper_id,
            "source_card_id": clean_text(source_card.get("card_id")),
            "claim_text": claim_text,
            "claim_type": claim_type,
            "status": status,
            "summary_text": _summary_text(
                claim_text=claim_text,
                normalization=normalization,
                anchors=anchors,
                source_title=clean_text(source_card.get("title")),
            ),
            "scope_text": clean_text(normalization.get("scope_text")),
            "condition_text": clean_text(normalization.get("condition_text")),
            "limitation_text": limitation_text,
            "negative_scope_text": clean_text(normalization.get("negative_scope_text")) or limitation_text,
            "task": clean_text(normalization.get("task")),
            "dataset": clean_text(normalization.get("dataset")),
            "metric": clean_text(normalization.get("metric")),
            "comparator": clean_text(normalization.get("comparator")),
            "result_direction": clean_text(normalization.get("result_direction") or "unknown"),
            "result_value_text": clean_text(normalization.get("result_value_text")),
            "result_value_numeric": normalization.get("result_value_numeric"),
            "evidence_strength": evidence_strength or "weak",
            "evidence_anchor_ids": _clean_list([anchor.get("anchor_id") for anchor in anchors]),
            "section_paths": _clean_list([anchor.get("section_path") for anchor in anchors]),
            "matched_entity_ids": _matched_entity_ids(claim),
            "search_text": _build_search_text(
                claim_text=claim_text,
                normalization=normalization,
                source_title=clean_text(source_card.get("title")),
                anchors=anchors,
            ),
            "quality_flag": _quality_flag(normalization=normalization, anchor_count=len(anchors), status=status),
            "confidence": confidence,
            "origin": "extracted",
            "trust_level": _trust_level(origin="extracted", status=status, evidence_strength=evidence_strength),
            "built_at": _snapshot_now(),
            "source_updated_at_snapshot": clean_text(source_card.get("updated_at")),
            "normalization_updated_at_snapshot": normalization_snapshot,
            **canonical,
            "updated_at": clean_text(source_card.get("updated_at")),
            "rank": int(ref.get("rank") or 0),
            "role": clean_text(ref.get("role")),
            "selection_reason": clean_text(ref.get("reason")),
            "anchors": anchors,
        }

    def _fallback_anchors_for_card(self, *, source_kind: str, source_card_id: str) -> list[dict[str, Any]]:
        if source_kind == "paper":
            return [dict(item) for item in self.sqlite_db.list_evidence_anchors_v2(card_id=source_card_id)]
        if source_kind == "web":
            return [dict(item) for item in self.sqlite_db.list_web_evidence_anchors_v2(card_id=source_card_id)]
        if source_kind == "vault":
            return [dict(item) for item in self.sqlite_db.list_vault_evidence_anchors_v2(card_id=source_card_id)]
        return []

    def _hydrate_stored_claim_cards(
        self,
        *,
        source_kind: str,
        source_card_id: str,
        cards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not cards:
            return []
        anchors = self._fallback_anchors_for_card(source_kind=source_kind, source_card_id=source_card_id)
        anchors_by_id = {
            clean_text(item.get("anchor_id")): dict(item)
            for item in anchors
            if clean_text(item.get("anchor_id"))
        }
        hydrated: list[dict[str, Any]] = []
        for item in cards:
            payload = dict(item)
            anchor_ids = [clean_text(value) for value in list(payload.get("evidence_anchor_ids") or []) if clean_text(value)]
            payload["anchors"] = [dict(anchors_by_id[anchor_id]) for anchor_id in anchor_ids if anchor_id in anchors_by_id]
            hydrated.append(payload)
        return hydrated

    def _build_fallback_claim_cards(self, *, source_kind: str, source_card: dict[str, Any]) -> list[dict[str, Any]]:
        anchors = self._fallback_anchors_for_card(source_kind=source_kind, source_card_id=clean_text(source_card.get("card_id")))
        if not anchors:
            return []
        type_map = {
            "method": "method",
            "result": "empirical",
            "supporting": "empirical",
            "version": "comparison",
            "limitation": "limitation",
            "dataset": "scope",
            "metric": "empirical",
            "decision": "definition",
            "action": "scope",
        }
        source_id = ""
        document_id = ""
        paper_id = ""
        if source_kind == "paper":
            source_id = clean_text(source_card.get("paper_id"))
            document_id = source_id and f"paper:{source_id}" or ""
            paper_id = source_id
        elif source_kind == "web":
            source_id = clean_text(source_card.get("canonical_url") or source_card.get("document_id"))
            document_id = clean_text(source_card.get("document_id"))
        else:
            source_id = clean_text(source_card.get("note_id"))
            document_id = source_id
        cards: list[dict[str, Any]] = []
        for rank, anchor in enumerate(anchors, 1):
            excerpt = clean_text(anchor.get("excerpt"))
            if not excerpt:
                continue
            evidence_role = clean_text(anchor.get("evidence_role"))
            claim_type = type_map.get(evidence_role, "unknown")
            cards.append(
                {
                    "claim_card_id": f"claim-card-v1:{source_kind}:fallback:{clean_text(source_card.get('card_id'))}:{rank}",
                    "claim_id": f"fallback-claim:{source_kind}:{clean_text(source_card.get('card_id'))}:{rank}",
                    "source_kind": source_kind,
                    "source_id": source_id,
                    "document_id": document_id,
                    "paper_id": paper_id,
                    "source_card_id": clean_text(source_card.get("card_id")),
                    "claim_text": excerpt,
                    "claim_type": claim_type,
                    "status": "weak",
                    "summary_text": trim_text(f"{clean_text(source_card.get('title'))} | {excerpt}", max_chars=260),
                    "scope_text": "",
                    "condition_text": "",
                    "limitation_text": excerpt if claim_type == "limitation" else "",
                    "negative_scope_text": excerpt if claim_type == "limitation" else "",
                    "task": "",
                    "dataset": "",
                    "metric": "",
                    "comparator": "",
                    "result_direction": "unknown",
                    "result_value_text": "",
                    "result_value_numeric": None,
                    "evidence_strength": "weak",
                    "evidence_anchor_ids": _clean_list([anchor.get("anchor_id")]),
                    "section_paths": _clean_list([anchor.get("section_path")]),
                    "matched_entity_ids": [],
                    "search_text": clean_text(f"{source_card.get('title')} {evidence_role} {excerpt}"),
                    "quality_flag": "needs_review",
                    "confidence": _stable_float(anchor.get("score")) or 0.5,
                    "origin": "synthetic_fallback",
                    "trust_level": "low",
                    "built_at": _snapshot_now(),
                    "source_updated_at_snapshot": clean_text(source_card.get("updated_at")),
                    "normalization_updated_at_snapshot": "",
                    "task_canonical": "",
                    "dataset_canonical": "",
                    "dataset_family": "",
                    "dataset_version": "",
                    "metric_canonical": "",
                    "comparator_canonical": "",
                    "updated_at": clean_text(source_card.get("updated_at")),
                    "rank": rank,
                    "role": evidence_role or "supporting",
                    "selection_reason": "anchor_fallback",
                    "anchors": [dict(anchor)],
                }
            )
        return cards

    def build_and_store_for_source_card(self, *, source_kind: str, source_card: dict[str, Any]) -> list[dict[str, Any]]:
        source_card_id = clean_text(source_card.get("card_id"))
        if source_kind not in {"paper", "web", "vault"} or not source_card_id:
            return []
        self._ensure_aliases_seeded()
        existing_refs = self.sqlite_db.list_claim_card_source_refs(source_card_id=source_card_id)
        existing_claim_card_ids = {
            clean_text(item.get("claim_card_id"))
            for item in existing_refs
            if clean_text(item.get("claim_card_id"))
        }
        rows, _normalizations, _anchors = self._claim_rows(source_kind=source_kind, source_card_id=source_card_id, source_card=source_card)
        claim_cards = [self._build_source_claim_card(source_kind=source_kind, row=row) for row in rows]
        if not claim_cards:
            claim_cards = self._build_fallback_claim_cards(source_kind=source_kind, source_card=source_card)
        for item in claim_cards:
            self.sqlite_db.upsert_claim_card(card=item)
        self.sqlite_db.replace_claim_card_source_refs(
            source_card_id=source_card_id,
            refs=[
                {
                    "claim_card_id": item.get("claim_card_id"),
                    "source_kind": source_kind,
                    "source_id": item.get("source_id"),
                    "document_id": item.get("document_id"),
                    "paper_id": item.get("paper_id"),
                    "role": item.get("role") or "source_card",
                    "rank": item.get("rank") or 0,
                }
                for item in claim_cards
            ],
        )
        current_claim_card_ids = {
            clean_text(item.get("claim_card_id"))
            for item in claim_cards
            if clean_text(item.get("claim_card_id"))
        }
        stale_claim_card_ids = sorted(existing_claim_card_ids - current_claim_card_ids)
        if stale_claim_card_ids:
            self.sqlite_db.delete_claim_cards(claim_card_ids=stale_claim_card_ids)
        for item in claim_cards:
            frame = self.alignment.frame_key(item)
            if not any(frame):
                self.sqlite_db.replace_claim_card_alignment_refs(claim_card_id=clean_text(item.get("claim_card_id")), refs=[])
                continue
            peers = self.sqlite_db.list_claim_cards(
                task_canonical=clean_text(item.get("task_canonical")) or None,
                dataset_canonical=clean_text(item.get("dataset_canonical")) or None,
                metric_canonical=clean_text(item.get("metric_canonical")) or None,
                comparator_canonical=clean_text(item.get("comparator_canonical")) or None,
                limit=100,
            )
            refs = self.alignment.build_alignment_refs(item, peers)
            self.sqlite_db.replace_claim_card_alignment_refs(claim_card_id=clean_text(item.get("claim_card_id")), refs=refs)
        return claim_cards

    def list_for_source_card(self, *, source_card_id: str) -> list[dict[str, Any]]:
        refs = self.sqlite_db.list_claim_card_source_refs(source_card_id=source_card_id)
        claim_card_ids = [clean_text(item.get("claim_card_id")) for item in refs if clean_text(item.get("claim_card_id"))]
        if not claim_card_ids:
            return []
        cards = self.sqlite_db.list_claim_cards(claim_card_ids=claim_card_ids, limit=max(10, len(claim_card_ids) * 2))
        by_id = {clean_text(item.get("claim_card_id")): dict(item) for item in cards}
        ordered: list[dict[str, Any]] = []
        source_kind = ""
        for ref in refs:
            item = by_id.get(clean_text(ref.get("claim_card_id")))
            if item:
                item["source_card_id"] = clean_text(ref.get("source_card_id"))
                source_kind = source_kind or clean_text(ref.get("source_kind"))
                ordered.append(item)
        return self._hydrate_stored_claim_cards(
            source_kind=source_kind,
            source_card_id=source_card_id,
            cards=ordered,
        )

    def load_or_build_for_source_card(self, *, source_kind: str, source_card: dict[str, Any]) -> list[dict[str, Any]]:
        source_card_id = clean_text(source_card.get("card_id"))
        if source_kind not in {"paper", "web", "vault"} or not source_card_id:
            return []
        rows, normalizations, _anchors = self._claim_rows(source_kind=source_kind, source_card_id=source_card_id, source_card=source_card)
        stored_cards = self.list_for_source_card(source_card_id=source_card_id)
        if not self._needs_rebuild_for_source_card(
            source_kind=source_kind,
            source_card=source_card,
            stored_cards=stored_cards,
            rows=rows,
            normalizations=normalizations,
        ):
            return stored_cards
        return self.build_and_store_for_source_card(source_kind=source_kind, source_card=source_card)

    def build_project_claim_cards(self, *, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return build_project_claim_cards(cards=cards)


def rank_claim_cards(*, query: str, claim_cards: list[dict[str, Any]], intent: str) -> list[dict[str, Any]]:
    type_bias = {
        "paper_summary": {"method": 1.2, "definition": 0.8, "empirical": 0.7},
        "paper_lookup": {"method": 1.1, "definition": 0.7, "empirical": 0.9},
        "comparison": {"comparison": 1.4, "empirical": 1.2},
        "relation": {"definition": 1.1, "method": 0.8, "scope": 0.8},
        "evaluation": {"empirical": 1.4, "comparison": 1.0, "limitation": 0.6},
        "implementation": {"method": 1.4, "scope": 0.7},
        "temporal": {"empirical": 0.8, "comparison": 0.8, "method": 0.5},
        "definition": {"definition": 1.5, "method": 0.6},
    }
    query_token_set = {term for term in query_terms(query) if term}
    ranked: list[dict[str, Any]] = []
    for item in claim_cards:
        normalized_overlap = 0.0
        ranking_reasons: list[str] = []
        for field in (
            "task",
            "dataset",
            "metric",
            "comparator",
            "task_canonical",
            "dataset_canonical",
            "metric_canonical",
            "comparator_canonical",
        ):
            token = clean_text(item.get(field))
            if token and any(term in token.casefold() or token.casefold() in term for term in query_token_set):
                normalized_overlap += 1.2
                ranking_reasons.append(f"{field}_match")
        base_overlap = text_overlap(
            query,
            item.get("claim_text"),
            item.get("summary_text"),
            item.get("search_text"),
            item.get("scope_text"),
            item.get("condition_text"),
            item.get("limitation_text"),
        )
        claim_type = clean_text(item.get("claim_type")).casefold() or "unknown"
        bias = type_bias.get(intent, {}).get(claim_type, 0.0)
        evidence_strength = {"strong": 1.2, "medium": 0.7, "weak": 0.2}.get(clean_text(item.get("evidence_strength")).casefold(), 0.0)
        source_card_score = _stable_float(item.get("source_card_score"))
        trust_bonus = {"high": 0.6, "medium": 0.15, "low": -0.4}.get(clean_text(item.get("trust_level")).casefold(), 0.0)
        penalty = 0.0
        status = clean_text(item.get("status")).casefold()
        if status == "contradicted":
            penalty += 2.5
            ranking_reasons.append("contradiction_penalty")
        elif status == "unsupported":
            penalty += 2.0
            ranking_reasons.append("unsupported_penalty")
        elif status == "weak":
            penalty += 0.9
            ranking_reasons.append("weak_penalty")
        score = round(
            base_overlap
            + normalized_overlap
            + bias
            + evidence_strength
            + trust_bonus
            + (source_card_score * 0.35)
            + _stable_float(item.get("confidence"))
            - penalty,
            4,
        )
        enriched = dict(item)
        enriched["selection_score"] = score
        reasons = ranking_reasons + ([f"type_bias:{claim_type}"] if bias else []) + ([f"evidence:{clean_text(item.get('evidence_strength')).casefold()}"] if evidence_strength else [])
        if normalized_overlap:
            reasons.append("normalized_overlap")
        if clean_text(item.get("trust_level")):
            reasons.append(f"trust:{clean_text(item.get('trust_level')).casefold()}")
        enriched["ranking_reasons"] = reasons
        ranked.append(enriched)
    ranked.sort(key=lambda item: (-_stable_float(item.get("selection_score")), -_stable_float(item.get("confidence")), clean_text(item.get("claim_card_id"))))
    return ranked


def claim_alignment(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return ClaimCardAlignmentService.group_claim_cards(cards)


__all__ = ["ClaimCardAlignmentService", "ClaimCardBuilder", "build_project_claim_cards", "claim_alignment", "rank_claim_cards"]

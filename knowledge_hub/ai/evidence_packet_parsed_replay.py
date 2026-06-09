from __future__ import annotations

from pathlib import Path
from typing import Final
import hashlib
import re

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap = dict[str, JsonValue]

MAX_SPAN_CHARS: Final = 700
SAFE_PAPER_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _as_maps(value: JsonValue | None) -> list[JsonMap]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _as_strings(value: JsonValue | None) -> list[str]:
    return [str(item) for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _clean_text(value: JsonValue | None, *, limit: int = MAX_SPAN_CHARS) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _source_id(value: str) -> str:
    text = value.strip()
    for prefix in ("paper:", "arxiv:"):
        if text.startswith(prefix):
            return text.removeprefix(prefix)
    return text


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _document_path(papers_dir: Path, paper_id: str) -> Path | None:
    if not SAFE_PAPER_ID.fullmatch(paper_id):
        return None
    return papers_dir / "parsed" / paper_id / "document.md"


def _first_body_line(document: str) -> str:
    for line in document.splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            return text
    return document.strip()


def _span_for_paper(*, papers_dir: Path, paper_id: str, citation_index: int) -> JsonMap | None:
    path = _document_path(papers_dir, paper_id)
    if path is None or not path.exists():
        return None
    document = path.read_text(encoding="utf-8")
    source_text = _first_body_line(document)
    excerpt = _clean_text(source_text)
    if not excerpt:
        return None
    start = max(document.find(source_text), 0)
    end = start + len(source_text)
    locator = f"chars:{start}-{end}"
    source_hash = _sha256_text(document)
    snippet_hash = _sha256_text(excerpt)
    return {
        "sourceId": paper_id,
        "source_id": f"paper:{paper_id}",
        "sourceType": "paper",
        "citationLabel": f"S{citation_index}",
        "citation_label": f"S{citation_index}",
        "locator": locator,
        "spanLocator": locator,
        "text": excerpt,
        "contentHash": source_hash,
        "content_hash": source_hash,
        "snippet_hash": snippet_hash,
        "source_ref": f"papers_dir/parsed/{paper_id}/document.md",
    }


def _raw_evidence_from_span(span: JsonMap) -> JsonMap:
    return {
        "source_id": span["source_id"],
        "source_ref": span["sourceId"],
        "source_type": "paper",
        "citation_target": span["sourceId"],
        "citation_label": span["citationLabel"],
        "excerpt": span["text"],
        "content_hash": span["contentHash"],
        "source_content_hash": span["contentHash"],
        "snippet_hash": span["snippet_hash"],
        "span_locator": span["spanLocator"],
        "sanitized_source_ref": span["source_ref"],
    }


def _payload_for_run(run: JsonMap, *, papers_dir: Path) -> JsonMap:
    expected_ids = [_source_id(value) for value in _as_strings(run.get("expectedIds"))]
    spans = [
        span
        for index, paper_id in enumerate(expected_ids, start=1)
        if (span := _span_for_paper(papers_dir=papers_dir, paper_id=paper_id, citation_index=index)) is not None
    ]
    evidence = [_raw_evidence_from_span(span) for span in spans]
    return {
        "status": "packet_only_replay",
        "query": _clean_text(run.get("query"), limit=900),
        "evidencePacketContract": {"spans": spans},
        "evidence": evidence,
        "sources": evidence,
        "citations": [
            {"target": f"paper:{span['sourceId']}", "label": span["citationLabel"]}
            for span in spans
        ],
        "replayDiagnostics": {
            "collector": "parsed_artifact_packet_only",
            "dbVectorRead": False,
            "dbVectorMutation": False,
            "vaultRead": False,
            "externalCall": False,
            "rawArtifactRefsSanitized": True,
        },
    }


def collect_parsed_replay_payloads(*, manifest: JsonMap, papers_dir: Path) -> dict[str, JsonMap]:
    payloads: dict[str, JsonMap] = {}
    for run in _as_maps(manifest.get("runs")):
        run_id = _clean_text(run.get("runId"), limit=160)
        if run_id:
            payloads[run_id] = _payload_for_run(run, papers_dir=papers_dir)
    return payloads


__all__ = ["collect_parsed_replay_payloads"]

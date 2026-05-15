from __future__ import annotations

import json
from typing import Any

from mcp.server.lowlevel.server import ReadResourceContents
from mcp.types import Resource, ResourceTemplate

from knowledge_hub.application.evidence_registry import resolve_registry_lookup
from knowledge_hub.application.evidence_substrate import build_inspect_payload, build_substrate_contract


RESOURCE_MIME_TYPE = "application/json"


def list_khub_resources() -> list[Resource]:
    return [
        Resource(
            name="Corpus Status",
            uri="khub://corpus/status",
            description="Current corpus/index status and evidence-substrate contract summary.",
            mimeType=RESOURCE_MIME_TYPE,
        ),
        Resource(
            name="Evidence Substrate Contract",
            uri="khub://corpus/contract",
            description="Canonical Source -> PreparedSource -> RetrievalUnit -> EvidencePacket contract.",
            mimeType=RESOURCE_MIME_TYPE,
        ),
    ]


def list_khub_resource_templates() -> list[ResourceTemplate]:
    return [
        ResourceTemplate(
            name="Source Lookup Template",
            uriTemplate="khub://source/{source_id}",
            description="Lookup source-level metadata from local lexical/vector metadata.",
            mimeType=RESOURCE_MIME_TYPE,
        ),
        ResourceTemplate(
            name="Chunk Lookup Template",
            uriTemplate="khub://chunk/{chunk_id}",
            description="Lookup chunk-level metadata and preview text from the lexical index.",
            mimeType=RESOURCE_MIME_TYPE,
        ),
        ResourceTemplate(
            name="Evidence Packet Lookup Template",
            uriTemplate="khub://packet/{packet_id}",
            description="Lookup a persisted evidence/compare packet registry record when one exists.",
            mimeType=RESOURCE_MIME_TYPE,
        ),
        ResourceTemplate(
            name="Context Pack Lookup Template",
            uriTemplate="khub://context/{context_pack_id}",
            description="Lookup a persisted context pack registry record when one exists.",
            mimeType=RESOURCE_MIME_TYPE,
        ),
    ]


def _payload_resource(kind: str, identifier: str, *, status: str = "not_found", reason: str = "") -> dict[str, Any]:
    return {
        "schema": "knowledge-hub.mcp.resource.result.v1",
        "status": status,
        "resourceKind": kind,
        "identifier": identifier,
        "reason": reason or f"{kind} registry record not found",
        "contract": build_substrate_contract(),
    }


def _registry_resource(sqlite_db: Any, kind: str, identifier: str) -> dict[str, Any]:
    lookup = resolve_registry_lookup(sqlite_db, kind, identifier) if sqlite_db is not None else {}
    status = str(lookup.get("status") or "not_found")
    return {
        "schema": "knowledge-hub.mcp.resource.result.v1",
        "status": status,
        "resourceKind": kind,
        "identifier": identifier,
        "reason": str(lookup.get("reason") or f"{kind} registry record not found"),
        "contract": build_substrate_contract(),
        "registryLookup": lookup,
        "payload": dict(lookup.get("payload") or {}),
        "lineage": dict(lookup.get("lineage") or {}),
        "authority": dict(lookup.get("authority") or {}),
        "warnings": list(lookup.get("warnings") or []),
    }


def read_khub_resource_payload(state: Any, uri: str) -> dict[str, Any]:
    uri_text = str(uri or "").strip()
    config = getattr(state, "config", None)
    sqlite_db = getattr(state, "sqlite_db", None)
    if config is None:
        return _payload_resource("unknown", uri_text, status="failed", reason="MCP core runtime is not initialized")

    if uri_text in {"khub://corpus/status", "khub://corpus"}:
        return build_inspect_payload(config=config, sqlite_db=sqlite_db, target="corpus")
    if uri_text == "khub://corpus/contract":
        return build_substrate_contract()
    if uri_text.startswith("khub://source/"):
        identifier = uri_text.removeprefix("khub://source/").strip("/")
        return build_inspect_payload(config=config, sqlite_db=sqlite_db, target="source", identifier=identifier)
    if uri_text.startswith("khub://chunk/"):
        identifier = uri_text.removeprefix("khub://chunk/").strip("/")
        return build_inspect_payload(config=config, sqlite_db=sqlite_db, target="chunk", identifier=identifier)
    if uri_text.startswith("khub://packet/"):
        return _registry_resource(sqlite_db, "packet", uri_text.removeprefix("khub://packet/").strip("/"))
    if uri_text.startswith("khub://context/"):
        return _registry_resource(sqlite_db, "context", uri_text.removeprefix("khub://context/").strip("/"))
    return _payload_resource("unknown", uri_text, status="failed", reason="unsupported khub resource URI")


def read_khub_resource(state: Any, uri: str) -> list[ReadResourceContents]:
    payload = read_khub_resource_payload(state, uri)
    return [
        ReadResourceContents(
            content=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            mime_type=RESOURCE_MIME_TYPE,
        )
    ]

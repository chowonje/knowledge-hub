"""Legacy claims command shim."""

from knowledge_hub.interfaces.cli.commands.claims_cmd import (
    WebOntologyExtractor,
    claims_group,
    extract_claim_candidates,
    get_llm_for_task,
    make_web_note_id,
    resolve_quality_mode_route,
    topic_matches_text,
)

__all__ = [
    "claims_group",
    "WebOntologyExtractor",
    "extract_claim_candidates",
    "get_llm_for_task",
    "make_web_note_id",
    "resolve_quality_mode_route",
    "topic_matches_text",
]

"""Legacy paper command shim."""

from knowledge_hub.interfaces.cli.commands.paper_cmd import (
    _concept_id,
    _extract_keywords_with_evidence,
    _regenerate_concept_index,
    _resolve_routed_llm,
    _update_note_concepts,
    _validate_arxiv_id,
    paper_group,
)
from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import (
    _update_obsidian_summary,
)

__all__ = [
    "_concept_id",
    "_extract_keywords_with_evidence",
    "_regenerate_concept_index",
    "_resolve_routed_llm",
    "_update_note_concepts",
    "_update_obsidian_summary",
    "_validate_arxiv_id",
    "paper_group",
]

from __future__ import annotations

from knowledge_hub.interfaces.cli.commands.paper_support import detect_synonym_groups, upsert_ai_concept


class _StubLLM:
    def __init__(self, raw: str):
        self.raw = raw
        self.prompt = ""
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        self.calls += 1
        return self.raw


def test_detect_synonym_groups_filters_noise_and_adds_local_groups():
    llm = _StubLLM("[]")

    groups = detect_synonym_groups(
        llm,
        ["AGENTS.md", "Across Diverse Tasks", "AI Agent", "AI Agents", "LLMs", "Large Language Models"],
    )

    assert {"canonical": "AI Agent", "aliases": ["AI Agents"]} in groups
    llm_group = next(group for group in groups if group["canonical"] == "Large Language Model")
    assert set(llm_group["aliases"]) == {"Large Language Models", "LLMs"}
    assert "AGENTS.md" not in llm.prompt
    assert "Across Diverse Tasks" not in llm.prompt


def test_detect_synonym_groups_salvages_fenced_object_payload():
    llm = _StubLLM(
        """
Here are the normalized groups.
```json
{"groups":[{"canonical":"Large Language Models","aliases":["Language Models","LLMs"]}]}
```
"""
    )

    groups = detect_synonym_groups(
        llm,
        ["Large Language Models", "Language Models", "LLMs"],
    )

    assert len(groups) == 2
    assert {"canonical": "Language Model", "aliases": ["Language Models"]} in groups
    llm_group = next(group for group in groups if group["canonical"] == "Large Language Model")
    assert set(llm_group["aliases"]) == {"Large Language Models", "LLMs"}


def test_detect_synonym_groups_blocks_loose_llm_merges():
    llm = _StubLLM(
        """
```json
[
  {"canonical":"interaction","aliases":["interactive"]},
  {"canonical":"motion","aliases":["movement"]},
  {"canonical":"Transformer","aliases":["Transformers"]}
]
```
"""
    )

    groups = detect_synonym_groups(
        llm,
        ["interaction", "interactive", "motion", "movement", "Transformer", "Transformers"],
    )

    assert {"canonical": "Transformer", "aliases": ["Transformers"]} in groups
    assert {"canonical": "interaction", "aliases": ["interactive"]} not in groups
    assert {"canonical": "motion", "aliases": ["movement"]} not in groups


def test_detect_synonym_groups_applies_curated_aliases_and_filters_generic_noise():
    llm = _StubLLM("[]")

    groups = detect_synonym_groups(
        llm,
        ["LLM Agents", "LLM-based agents", "Language Models", "prompt injection attacks", "Benchmark", "Survey"],
    )

    llm_agent_group = next(group for group in groups if group["canonical"] == "LLM Agent")
    assert set(llm_agent_group["aliases"]) == {"LLM Agents", "LLM-based agents"}
    assert {"canonical": "Language Model", "aliases": ["Language Models"]} in groups
    assert {"canonical": "Prompt Injection", "aliases": ["prompt injection attacks"]} in groups
    assert "Benchmark" not in llm.prompt
    assert "Survey" not in llm.prompt


def test_detect_synonym_groups_protects_benchmark_and_model_proper_nouns():
    llm = _StubLLM(
        """
```json
[
  {"canonical":"Benchmark","aliases":["AMA-Bench","FIRE-Bench"]},
  {"canonical":"Magma","aliases":["Gaia2"]}
]
```
"""
    )

    groups = detect_synonym_groups(
        llm,
        ["Benchmark", "AMA-Bench", "FIRE-Bench", "Magma", "Gaia2"],
    )

    assert groups == []


class _StubSQLite:
    def __init__(self):
        self.calls = []

    def resolve_entity(self, canonical_name: str, entity_type: str = "concept"):
        if canonical_name == "AI Agent" and entity_type == "concept":
            return {"entity_id": "existing-ai-agent", "canonical_name": canonical_name}
        return None

    def get_ontology_entity(self, entity_id: str):
        if entity_id == "existing-ai-agent":
            return {"entity_id": entity_id, "properties": {"category": "agent"}, "confidence": 0.9}
        return None

    def upsert_ontology_entity(self, **kwargs):
        self.calls.append(kwargs)


def test_upsert_ai_concept_reuses_existing_canonical_entity_id():
    sqlite_db = _StubSQLite()

    upsert_ai_concept(
        sqlite_db,
        entity_id="concept:ai-agent",
        canonical_name="AI Agent",
        source="paper_normalize_concepts",
    )

    assert len(sqlite_db.calls) == 1
    assert sqlite_db.calls[0]["entity_id"] == "existing-ai-agent"

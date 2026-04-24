from __future__ import annotations

import json
from pathlib import Path


INVENTORY_PATH = Path("docs/card_synthesis_hypothesis_inventory.json")


def _inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def test_card_synthesis_inventory_blocks_destructive_migration_by_default():
    payload = _inventory()

    assert payload["schema"] == "knowledge-hub.card-synthesis-hypothesis-inventory.v1"
    assert payload["migrationRequired"] is False
    assert payload["destructiveActions"] == []
    assert payload["recommendation"]["status"] == "hold_migration"


def test_card_synthesis_inventory_marks_all_card_surfaces_as_non_citation():
    payload = _inventory()
    card_surfaces = {item["surface"]: item for item in payload["cards"]}

    for surface in ("claim_cards_v1", "paper_section_cards_v1", "paper_cards_v2", "vault_cards_v2", "web_cards_v2"):
        assert surface in card_surfaces
        assert "citation" not in card_surfaces[surface]["defaultRuntimeRole"].replace("not_citation", "")
        assert card_surfaces[surface]["authority"] == "derivative"


def test_hypothesis_surface_is_recorded_as_not_implemented():
    payload = _inventory()

    assert payload["hypothesis"]["firstClassStore"] is False
    assert payload["hypothesis"]["status"] == "not_implemented"
    assert payload["proposals"]["evidenceRole"] == "not_citation_endpoint"

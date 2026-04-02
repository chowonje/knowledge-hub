"""Ontology-focused database interface.

Currently this class is a thin compatibility layer over SQLiteDatabase,
which already contains ontology entity/claim/relation/event/conflict methods.
"""

from __future__ import annotations

from knowledge_hub.infrastructure.persistence import SQLiteDatabase


class OntologyDatabase(SQLiteDatabase):
    """SQLite-backed ontology store interface."""


__all__ = ["OntologyDatabase"]

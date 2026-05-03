"""Backward-compatible database import surface.

Use module-specific imports for new code:
- knowledge_hub.infrastructure.persistence.VectorDatabase
- knowledge_hub.infrastructure.persistence.SQLiteDatabase
- knowledge_hub.core.ontology_db.OntologyDatabase
"""

from __future__ import annotations

from knowledge_hub.infrastructure.persistence import SQLiteDatabase, VectorDatabase
from knowledge_hub.core.ontology_db import OntologyDatabase

__all__ = ["VectorDatabase", "SQLiteDatabase", "OntologyDatabase"]

"""Ontology graph export + validation helper using optional KG stack."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_\-가-힣]+", "_", (value or "").strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "anonymous"


def _parse_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _library_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


RELATION_ENUM = {
    "causes",
    "enables",
    "part_of",
    "contrasts",
    "example_of",
    "requires",
    "improves",
    "related_to",
    "unknown_relation",
    "mentions",
}


@dataclass
class OntologyGraphResult:
    status: str
    run_id: str
    concept_count: int
    relation_count: int
    turtle_path: str | None
    validation: dict[str, Any]
    libraries: dict[str, bool]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "runId": self.run_id,
            "conceptCount": self.concept_count,
            "relationCount": self.relation_count,
            "turtlePath": self.turtle_path,
            "validation": self.validation,
            "libraries": self.libraries,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


class OntologyGraphService:
    def __init__(self, db: SQLiteDatabase, config: Config | None = None):
        self.db = db
        self.config = config

    def _libraries(self) -> dict[str, bool]:
        return {
            "rdflib": _library_available("rdflib"),
            "pyshacl": _library_available("pyshacl"),
            "owlready2": _library_available("owlready2"),
            "oxigraph": _library_available("oxigraph") or _library_available("pyoxigraph"),
        }

    def _collect_from_summary(self, ontology_summary: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        concepts = {}
        relations: list[dict[str, Any]] = []
        run_id = str(ontology_summary.get("runId", "")).strip() or "web_ontology"

        for item in ontology_summary.get("acceptedConcepts", []) or []:
            if not isinstance(item, dict):
                continue
            concept_id = str(item.get("canonical_id", "")).strip()
            if not concept_id:
                continue
            concepts[concept_id] = float(item.get("confidence", 0.0))

        for item in ontology_summary.get("acceptedRelations", []) or []:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_canonical_id", "")).strip()
            target_id = str(item.get("target_canonical_id", "")).strip()
            if not source_id or not target_id:
                continue
            relation = str(item.get("relation_norm", "")).strip()
            if relation not in RELATION_ENUM:
                relation = "unknown_relation"
            relations.append(
                {
                    "run_id": run_id,
                    "source_canonical_id": source_id,
                    "target_canonical_id": target_id,
                    "relation_norm": relation,
                    "confidence": float(item.get("confidence", 0.0)),
                    "source": "web",
                    "source_url": "",
                }
            )
            concepts.setdefault(source_id, max(concepts.get(source_id, 0.0), float(item.get("confidence", 0.0))))
            concepts.setdefault(target_id, max(concepts.get(target_id, 0.0), float(item.get("confidence", 0.0))))

        return concepts, relations

    def _collect_from_db(self, run_id: str | None = None, source: str | None = "web") -> tuple[dict[str, float], list[dict[str, Any]]]:
        relations_rows = self.db.list_relations(
            source_type="concept",
            target_type="concept",
            limit=2000,
        )

        concepts: dict[str, float] = {}
        relations: list[dict[str, Any]] = []
        for row in relations_rows:
            evidence = row.get("evidence_json") or {}
            source_type = str(evidence.get("source", "") or row.get("source", "")).strip()
            if source and source_type and source_type != source:
                continue
            reason = evidence.get("reason") if isinstance(evidence.get("reason"), dict) else {}
            record_run_id = str(reason.get("run_id", "") or evidence.get("run_id", "")).strip()
            if run_id and record_run_id and record_run_id != run_id:
                continue
            source_id = str(row.get("source_id", "")).strip()
            target_id = str(row.get("target_id", "")).strip()
            if not source_id or not target_id:
                continue
            relation_norm = str(evidence.get("relation_norm", "related_to"))
            if relation_norm not in RELATION_ENUM:
                relation_norm = "unknown_relation"
            confidence = float(row.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            relation = {
                "run_id": record_run_id or str(run_id or ""),
                "source_canonical_id": source_id,
                "target_canonical_id": target_id,
                "relation_norm": relation_norm,
                "confidence": confidence,
                "source": source_type,
                "source_url": str(evidence.get("source_url", "")),
            }
            relations.append(relation)
            concepts[source_id] = max(float(concepts.get(source_id, 0.0)), confidence)
            concepts[target_id] = max(float(concepts.get(target_id, 0.0)), confidence)

        return concepts, relations

    def _serialize_graph_turtle(self, concepts: dict[str, float], relations: list[dict[str, Any]], output_path: Path, run_id: str) -> None:
        from rdflib import Graph, Literal, Namespace
        from rdflib.namespace import RDF, RDFS

        ns = Namespace("https://knowledge-hub.local/ontology/")
        rel = Namespace("https://knowledge-hub.local/ontology/relation/")
        prov = Namespace("https://knowledge-hub.local/ontology/provenance/")

        graph = Graph()
        graph.bind("kh", ns)
        graph.bind("rrel", rel)
        graph.bind("prov", prov)
        graph.bind("rdfs", RDFS)

        for concept_id, confidence in concepts.items():
            c_uri = ns[f"concept/{_safe_token(concept_id)}"]
            graph.add((c_uri, RDF.type, ns.Concept))
            graph.add((c_uri, RDFS.label, Literal(concept_id)))
            raw_confidence = float(confidence)
            concept_confidence = round(max(0.0, min(1.0, raw_confidence)), 4)
            graph.add((c_uri, ns.confidence, Literal(concept_confidence)))

        for idx, relation in enumerate(relations):
            source_id = relation.get("source_canonical_id", "")
            target_id = relation.get("target_canonical_id", "")
            if not source_id or not target_id:
                continue
            relation_norm = str(relation.get("relation_norm", "unknown_relation"))
            source_uri = ns[f"concept/{_safe_token(source_id)}"]
            target_uri = ns[f"concept/{_safe_token(target_id)}"]
            predicate = rel[_safe_token(relation_norm)]
            graph.add((source_uri, predicate, target_uri))

            edge_uri = prov[f"edge/{_safe_token(run_id)}/{idx}"]
            graph.add((edge_uri, prov.sourceCanonicalId, Literal(source_id)))
            graph.add((edge_uri, prov.targetCanonicalId, Literal(target_id)))
            graph.add((edge_uri, prov.relationNorm, Literal(relation_norm)))
            raw_relation_confidence = float(relation.get("confidence", 0.0))
            relation_confidence = round(max(0.0, min(1.0, raw_relation_confidence)), 4)
            graph.add((edge_uri, prov.confidence, Literal(relation_confidence)))
            relation_run_id = str(relation.get("run_id", run_id))
            graph.add((edge_uri, prov.runId, Literal(relation_run_id)))
            source_url = str(relation.get("source_url", "")).strip()
            if source_url:
                graph.add((edge_uri, prov.sourceUrl, Literal(source_url)))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(graph.serialize(format="turtle"), encoding="utf-8")

    def _validate_graph(self, turtle_path: Path, run_id: str) -> dict[str, Any]:
        result = {
            "enabled": bool(_library_available("pyshacl")),
            "status": "skipped",
            "errors": [],
            "warnings": [],
            "reportPath": None,
            "passed": None,
        }

        if not _library_available("pyshacl") or not _library_available("rdflib"):
            result["reason"] = "pyshacl/rdflib not installed"
            return result

        try:
            from rdflib import Graph
            from rdflib.namespace import Namespace
            from pyshacl import validate

            graph = Graph()
            graph.parse(turtle_path.as_posix(), format="turtle")

            shapes = Graph()
            shapes.parse(data=f"""
                @prefix sh: <http://www.w3.org/ns/shacl#> .
                @prefix kh: <https://knowledge-hub.local/ontology/> .
                @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
                @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
                @prefix shsh: <https://knowledge-hub.local/ontology/shacl/> .
                kh:ConceptShape a sh:NodeShape ;
                  sh:targetClass kh:Concept ;
                  sh:property [
                    sh:path rdfs:label ;
                    sh:datatype xsd:string ;
                    sh:minCount 1
                  ] ;
                  sh:closed false .
                kh:OntologyGraph a sh:NodeShape ;
                  sh:closed false .
            """, format="turtle")

            valid, _, report_text = validate(graph, shacl_graph=shapes, inference="rdfs", abort_on_error=False, allow_warnings=True)
            result["status"] = "passed" if valid else "failed"
            result["passed"] = bool(valid)
            result["report"] = str(report_text)[:12000] if report_text else ""
        except Exception as error:
            result["status"] = "failed"
            result["passed"] = False
            result["errors"].append(str(error))

        report_path = turtle_path.with_suffix(".validation.ttl")
        if result.get("report"):
            report_path.write_text(str(result["report"]), encoding="utf-8")
            result["reportPath"] = str(report_path)
        return result

    def export_from_summary(
        self,
        ontology_summary: dict[str, Any],
        run_id: str | None = None,
        output_path: str | None = None,
        validate: bool = True,
    ) -> OntologyGraphResult:
        topic = str((ontology_summary or {}).get("topic", "")).strip()
        run_id = str(run_id or ontology_summary.get("runId", "")).strip() or "web_ontology"
        if output_path:
            output = Path(output_path).expanduser().resolve()
        else:
            base = Path(self.config.sqlite_path).expanduser().resolve().parent if self.config else Path.cwd()
            if topic:
                from knowledge_hub.learning.obsidian_writeback import build_paths
                paths = build_paths(self.config.vault_path if self.config else str(Path.home()), topic)
                output = paths.web_concepts_file.with_name(f"{paths.web_concepts_file.stem}.ttl")
            else:
                output = base / "web_ontology_graph.ttl"

        concepts, relations = self._collect_from_summary(ontology_summary or {})

        libraries = self._libraries()
        if not libraries["rdflib"]:
            return OntologyGraphResult(
                status="skipped",
                run_id=run_id,
                concept_count=len(concepts),
                relation_count=len(relations),
                turtle_path=None,
                validation={"enabled": False, "status": "skipped", "reason": "rdflib not installed"},
                libraries=libraries,
                created_at=_now_iso(),
                updated_at=_now_iso(),
            )

        try:
            self._serialize_graph_turtle(concepts, relations, output, run_id=run_id)
        except Exception as error:
            return OntologyGraphResult(
                status="error",
                run_id=run_id,
                concept_count=len(concepts),
                relation_count=len(relations),
                turtle_path=str(output),
                validation={"status": "error", "errors": [str(error)]},
                libraries=libraries,
                created_at=_now_iso(),
                updated_at=_now_iso(),
            )

        validation = {"status": "skipped", "enabled": bool(libraries["pyshacl"]), "errors": []}
        if validate:
            validation = self._validate_graph(output, run_id)

        return OntologyGraphResult(
            status="ok",
            run_id=run_id,
            concept_count=len(concepts),
            relation_count=len(relations),
            turtle_path=str(output),
            validation=validation,
            libraries=libraries,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )

    def export_from_db(
        self,
        run_id: str | None = None,
        source: str = "web",
        output_path: str | None = None,
        validate: bool = True,
    ) -> OntologyGraphResult:
        if not run_id:
            run_id = "web_ontology"
        if output_path:
            output = Path(output_path).expanduser().resolve()
        else:
            output = Path(self.config.sqlite_path).expanduser().resolve().parent / f"{run_id}_ontology_graph.ttl"

        concepts, relations = self._collect_from_db(run_id=run_id, source=source)
        libraries = self._libraries()
        if not libraries["rdflib"]:
            return OntologyGraphResult(
                status="skipped",
                run_id=run_id,
                concept_count=len(concepts),
                relation_count=len(relations),
                turtle_path=None,
                validation={"enabled": False, "status": "skipped", "reason": "rdflib not installed"},
                libraries=libraries,
                created_at=_now_iso(),
                updated_at=_now_iso(),
            )
        try:
            self._serialize_graph_turtle(concepts, relations, output, run_id=run_id)
        except Exception as error:
            return OntologyGraphResult(
                status="error",
                run_id=run_id,
                concept_count=len(concepts),
                relation_count=len(relations),
                turtle_path=str(output),
                validation={"status": "error", "errors": [str(error)]},
                libraries=libraries,
                created_at=_now_iso(),
                updated_at=_now_iso(),
            )

        validation = {"status": "skipped", "enabled": bool(libraries["pyshacl"]), "errors": []}
        if validate:
            validation = self._validate_graph(output, run_id)

        return OntologyGraphResult(
            status="ok",
            run_id=run_id,
            concept_count=len(concepts),
            relation_count=len(relations),
            turtle_path=str(output),
            validation=validation,
            libraries=libraries,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )

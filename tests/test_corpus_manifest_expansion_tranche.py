from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from knowledge_hub.core.schema_validator import validate_payload


class _ConfigWithPapersDir:
    def __init__(self, papers_dir: Path):
        self.papers_dir = str(papers_dir)

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        if tuple(args) == ("storage", "papers_dir"):
            return self.papers_dir
        return default


def _load_builder() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "eval/knowledgeos/scripts/build_corpus_manifest_expansion_tranche.py"
    spec = importlib.util.spec_from_file_location("corpus_manifest_expansion_tranche_builder", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _allowlist_payload(source_id: str, content: bytes) -> dict:
    return {
        "schema": "knowledge-hub.priority-corpus-manifest-expansion-allowlist.v1",
        "generated_at": "2026-05-21T00:00:00Z",
        "scope_note": "test fixture",
        "source_report": "report.json",
        "counts": {
            "allowlist_rows": 1,
            "first_tranche_recommended_rows": 1,
            "by_candidate_tier": {"recent_ai": 1},
            "already_in_manifest_excluded": 0,
        },
        "first_tranche_size_target": "30-50",
        "allowlist": [
            {
                "source_id": source_id,
                "title": "Recovered Source Paper",
                "year": 2026,
                "candidate_tier": "recent_ai",
                "join_method": "direct_source_id",
                "observed_sha256": _hash(content),
                "byte_length": len(content),
                "canonical_filename": f"{source_id}.pdf",
                "corpus_location_ref": f"papers_dir/recovered_sources/arxiv/{source_id}.pdf",
                "provenance_url": f"https://arxiv.org/abs/{source_id}",
                "parsed_status": "parsed_present",
            }
        ],
        "first_tranche_recommendation": [],
    }


def test_expansion_tranche_plan_reverifies_recovered_source_without_manifest_write(tmp_path: Path) -> None:
    builder = _load_builder()
    papers_dir = tmp_path / "papers"
    recovered_dir = papers_dir / "recovered_sources" / "arxiv"
    recovered_dir.mkdir(parents=True)
    source_id = "2501.00001"
    content = b"%PDF-1.4 recovered source"
    (recovered_dir / f"{source_id}.pdf").write_bytes(content)
    manifest_path = _write_json(
        tmp_path / "manifest.json",
        {"schema": "knowledge-hub.corpus-manifest.v1", "artifacts": []},
    )
    allowlist_path = _write_json(tmp_path / "allowlist.json", _allowlist_payload(source_id, content))

    payload = builder.build_expansion_tranche_plan(
        allowlist_path=allowlist_path,
        manifest_path=manifest_path,
        papers_dir=papers_dir,
        target_count=1,
        batch_size=1,
        config=_ConfigWithPapersDir(papers_dir),
    )

    assert payload["status"] == "ready"
    assert validate_payload(payload, builder.SCHEMA_ID, strict=True).errors == []
    assert payload["counts"]["ready_rows"] == 1
    entry = payload["proposed_manifest_entries"][0]
    assert entry["sourceIds"] == [source_id]
    assert entry["expectedSourceContentHash"] == _hash(content)
    assert entry["byteLength"] == len(content)
    assert entry["provenanceUrl"] == f"https://arxiv.org/pdf/{source_id}"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"] == []


def test_expansion_tranche_apply_appends_only_ready_rows(tmp_path: Path) -> None:
    builder = _load_builder()
    papers_dir = tmp_path / "papers"
    recovered_dir = papers_dir / "recovered_sources" / "arxiv"
    recovered_dir.mkdir(parents=True)
    source_id = "2501.00002"
    content = b"%PDF-1.4 apply source"
    (recovered_dir / f"{source_id}.pdf").write_bytes(content)
    manifest_path = _write_json(
        tmp_path / "manifest.json",
        {"schema": "knowledge-hub.corpus-manifest.v1", "artifacts": []},
    )
    allowlist_path = _write_json(tmp_path / "allowlist.json", _allowlist_payload(source_id, content))

    payload = builder.build_expansion_tranche_plan(
        allowlist_path=allowlist_path,
        manifest_path=manifest_path,
        papers_dir=papers_dir,
        target_count=1,
        batch_size=1,
        apply=True,
        config=_ConfigWithPapersDir(papers_dir),
    )

    assert payload["status"] == "applied"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [entry["sourceIds"][0] for entry in manifest["artifacts"]] == [source_id]

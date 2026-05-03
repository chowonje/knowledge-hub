#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from knowledge_hub.application.context import AppContextFactory
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_extraction import PaperMemorySchemaExtractor
from knowledge_hub.papers.memory_quality import evaluate_paper_memory_quality, summarize_quality_reports
from knowledge_hub.providers.registry import get_llm


def _load_ids(path_value: str) -> list[str]:
    rows = Path(path_value).expanduser().read_text(encoding="utf-8").splitlines()
    return [row.strip() for row in rows if row.strip() and not row.strip().startswith("#")]


def _build_builder(factory: AppContextFactory, *, provider: str, model: str, timeout: float) -> PaperMemoryBuilder:
    sqlite_db = factory.get_sqlite_db()
    cfg = dict(factory.config.get_provider_config(provider) or {})
    cfg["timeout"] = float(timeout)
    cfg.setdefault("temperature", 0.0)
    llm = get_llm(provider, model=model, **cfg)
    extractor = PaperMemorySchemaExtractor(llm, model=model)
    return PaperMemoryBuilder(sqlite_db, schema_extractor=extractor, extraction_mode="shadow")


def _run_model(factory: AppContextFactory, paper_ids: list[str], *, provider: str, model: str, timeout: float) -> dict[str, Any]:
    builder = _build_builder(factory, provider=provider, model=model, timeout=timeout)
    rows: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        item = builder.build_card(paper_id=paper_id).to_record()
        diagnostics = builder.get_last_extraction_diagnostics(paper_id)
        rows.append(
            {
                "paperId": paper_id,
                "title": str(item.get("title") or ""),
                **diagnostics,
                **evaluate_paper_memory_quality(
                    title=item.get("title"),
                    paper_core=item.get("paper_core"),
                    method_core=item.get("method_core"),
                    evidence_core=item.get("evidence_core"),
                    limitations=item.get("limitations"),
                    diagnostics=diagnostics,
                ),
            }
        )
    return {
        "provider": provider,
        "model": model,
        "summary": {
            **summarize_quality_reports(rows),
            "attempted": len(rows),
            "avgLatencyMs": round(
                sum(float(row.get("latencyMs") or 0.0) for row in rows) / max(1, len(rows)),
                3,
            ),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local paper-memory extractors on a bounded paper-id subset.")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument("--paper-id-file", required=True, help="Newline-delimited paper ids to compare on")
    parser.add_argument("--provider", default="ollama", help="Provider name for all compared models")
    parser.add_argument("--models", default="exaone3.5:7.8b,qwen3:14b", help="Comma-separated local models")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-call timeout in seconds")
    args = parser.parse_args()

    paper_ids = _load_ids(args.paper_id_file)
    factory = AppContextFactory(config_path=args.config)
    payload = {
        "status": "ok",
        "paperIds": paper_ids,
        "comparisons": [
            _run_model(factory, paper_ids, provider=str(args.provider), model=model.strip(), timeout=float(args.timeout))
            for model in str(args.models or "").split(",")
            if model.strip()
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

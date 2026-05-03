#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from knowledge_hub.application.context import AppContextFactory
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_extraction import PaperMemoryExtractionV1, PaperMemorySchemaExtractor
from knowledge_hub.providers.registry import get_llm


def _auto_load_dotenv(repo_root: Path) -> None:
    for candidate in (Path.cwd() / ".env", repo_root / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
        break


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def _builder(factory: AppContextFactory, *, provider: str, model: str, timeout_sec: int) -> PaperMemoryBuilder:
    cfg = dict(factory.config.get_provider_config(provider) or {})
    cfg["timeout"] = float(timeout_sec)
    cfg["request_timeout"] = float(timeout_sec)
    cfg.setdefault("temperature", 0.0)
    llm = get_llm(provider, model=model, **cfg)
    extractor = PaperMemorySchemaExtractor(llm, model=model)
    return PaperMemoryBuilder(factory.create_sqlite_db(), schema_extractor=extractor, extraction_mode="schema")


def _extractor(factory: AppContextFactory, *, provider: str, model: str, timeout_sec: int) -> PaperMemorySchemaExtractor:
    cfg = dict(factory.config.get_provider_config(provider) or {})
    cfg["timeout"] = float(timeout_sec)
    cfg["request_timeout"] = float(timeout_sec)
    cfg.setdefault("temperature", 0.0)
    llm = get_llm(provider, model=model, **cfg)
    return PaperMemorySchemaExtractor(llm, model=model)


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _clip_text(value: str, limit: int) -> str:
    token = str(value or "").strip()
    if len(token) <= limit:
        return token
    return token[:limit].rstrip()


def _compact_input_from_row(row: dict[str, str]) -> dict[str, Any]:
    source_excerpt = _clean_text(row.get("source_excerpt", ""))
    baseline_paper_core = _clean_text(row.get("paper_core", ""))
    baseline_method = _clean_text(row.get("method_core", ""))
    baseline_evidence = _clean_text(row.get("evidence_core", ""))
    baseline_limitations = _clean_text(row.get("limitations", ""))
    limitations_excerpt = ""
    lowered_excerpt = source_excerpt.casefold()
    if any(token in lowered_excerpt for token in ("limitation", "limitations", "future work", "limited", "한계")):
        limitations_excerpt = source_excerpt
    return {
        "paperId": _clean_text(row.get("paper_id", "")),
        "title": _clean_text(row.get("title", "")),
        "field": _clean_text(row.get("field", "")),
        "year": _clean_text(row.get("year", "")),
        "summaryExcerpt": _clip_text(source_excerpt or baseline_paper_core, 800),
        "methodExcerpt": _clip_text(" ".join(part for part in (source_excerpt, baseline_method) if part), 800),
        "findingsExcerpt": _clip_text(" ".join(part for part in (source_excerpt, baseline_evidence) if part), 800),
        "limitationsExcerpt": _clip_text(limitations_excerpt, 400),
        "topConceptCandidates": [],
        "claimTexts": [],
        "textSanitation": {
            "preferredSource": str(row.get("excerpt_source", "") or "sample_excerpt"),
            "preferredLength": len(source_excerpt),
            "weakContent": not bool(source_excerpt),
            "warnings": [],
        },
        "deterministicBaseline": {
            "paperCore": baseline_paper_core,
            "methodCore": baseline_method,
            "evidenceCore": baseline_evidence,
            "limitations": baseline_limitations,
            "conceptLinks": [],
            "qualityFlag": _clean_text(row.get("quality_flag", "")) or "unscored",
        },
        "limitationsPolicy": {
            "explicitSupportPresent": bool(limitations_excerpt),
            "fallbackText": "limitations not explicit in visible excerpt",
        },
    }


def _build_variant(
    builder: PaperMemoryBuilder | None,
    *,
    paper_id: str,
    baseline_row: dict[str, str],
    variant_name: str,
) -> dict[str, Any]:
    if builder is None:
        return {
            f"{variant_name}_status": "baseline",
            f"{variant_name}_paper_core": baseline_row.get("paper_core", ""),
            f"{variant_name}_method_core": baseline_row.get("method_core", ""),
            f"{variant_name}_evidence_core": baseline_row.get("evidence_core", ""),
            f"{variant_name}_limitations": baseline_row.get("limitations", ""),
            f"{variant_name}_diagnostics_json": "{}",
            f"{variant_name}_elapsed_sec": "0.00",
        }
    started = time.perf_counter()
    try:
        card = builder.build_card(paper_id=paper_id)
        elapsed = time.perf_counter() - started
        diagnostics = builder.get_last_extraction_diagnostics(paper_id)
        return {
            f"{variant_name}_status": "ok",
            f"{variant_name}_paper_core": card.paper_core,
            f"{variant_name}_method_core": card.method_core,
            f"{variant_name}_evidence_core": card.evidence_core,
            f"{variant_name}_limitations": card.limitations,
            f"{variant_name}_diagnostics_json": json.dumps(diagnostics, ensure_ascii=False),
            f"{variant_name}_elapsed_sec": f"{elapsed:.2f}",
        }
    except Exception as exc:  # pragma: no cover - runtime comparison path
        elapsed = time.perf_counter() - started
        return {
            f"{variant_name}_status": f"error:{type(exc).__name__}",
            f"{variant_name}_paper_core": "",
            f"{variant_name}_method_core": "",
            f"{variant_name}_evidence_core": "",
            f"{variant_name}_limitations": "",
            f"{variant_name}_diagnostics_json": json.dumps({"error": str(exc)}, ensure_ascii=False),
            f"{variant_name}_elapsed_sec": f"{elapsed:.2f}",
        }


def _build_compact_variant(
    extractor: PaperMemorySchemaExtractor,
    *,
    baseline_row: dict[str, str],
    variant_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    compact_input = _compact_input_from_row(baseline_row)
    try:
        raw_payload, metadata = extractor.extract_with_metadata(paper=compact_input)
        extraction = PaperMemoryExtractionV1.from_dict(raw_payload, default_model=extractor.model)
        elapsed = time.perf_counter() - started
        if extraction is None:
            return {
                f"{variant_name}_status": "fallback:empty_payload",
                f"{variant_name}_paper_core": baseline_row.get("paper_core", ""),
                f"{variant_name}_method_core": baseline_row.get("method_core", ""),
                f"{variant_name}_evidence_core": baseline_row.get("evidence_core", ""),
                f"{variant_name}_limitations": baseline_row.get("limitations", ""),
                f"{variant_name}_diagnostics_json": json.dumps(
                    {
                        "mode": "compact",
                        "rawPayloadBytes": int(metadata.get("rawPayloadBytes") or 0),
                        "parsedFields": list(metadata.get("parsedFields") or []),
                        "fallbackUsed": True,
                    },
                    ensure_ascii=False,
                ),
                f"{variant_name}_elapsed_sec": f"{elapsed:.2f}",
            }
        return {
            f"{variant_name}_status": "ok",
            f"{variant_name}_paper_core": extraction.thesis or baseline_row.get("paper_core", ""),
            f"{variant_name}_method_core": extraction.method_core or baseline_row.get("method_core", ""),
            f"{variant_name}_evidence_core": extraction.evidence_core or baseline_row.get("evidence_core", ""),
            f"{variant_name}_limitations": extraction.limitations or baseline_row.get("limitations", ""),
            f"{variant_name}_diagnostics_json": json.dumps(
                {
                    "mode": "compact",
                    "rawPayloadBytes": int(metadata.get("rawPayloadBytes") or 0),
                    "parsedFields": list(metadata.get("parsedFields") or []),
                    "extractorModel": extraction.extractor_model,
                    "warnings": list(extraction.warnings or []),
                    "fieldConfidence": dict(extraction.field_confidence),
                    "coverageByField": dict(extraction.coverage_status_by_field),
                    "compactInput": compact_input,
                },
                ensure_ascii=False,
            ),
            f"{variant_name}_elapsed_sec": f"{elapsed:.2f}",
        }
    except Exception as exc:  # pragma: no cover - runtime comparison path
        elapsed = time.perf_counter() - started
        return {
            f"{variant_name}_status": f"error:{type(exc).__name__}",
            f"{variant_name}_paper_core": "",
            f"{variant_name}_method_core": "",
            f"{variant_name}_evidence_core": "",
            f"{variant_name}_limitations": "",
            f"{variant_name}_diagnostics_json": json.dumps(
                {"mode": "compact", "error": str(exc), "compactInput": compact_input},
                ensure_ascii=False,
            ),
            f"{variant_name}_elapsed_sec": f"{elapsed:.2f}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build baseline/exaone/openai paper-memory comparison rows for a sample set.")
    parser.add_argument(
        "--sample-csv",
        default="eval/knowledgeos/review/knowledgeos_paper_memory_vs_gptpro_sample_12_v1.csv",
        help="Input sample CSV containing baseline card fields and source excerpts",
    )
    parser.add_argument(
        "--config",
        default="exports/config_bge_m3_local.yaml",
        help="Knowledge Hub config path",
    )
    parser.add_argument(
        "--exaone-provider",
        default="ollama",
        help="Provider for EXAONE rebuild variant",
    )
    parser.add_argument(
        "--exaone-model",
        default="exaone3.5:7.8b",
        help="Model for EXAONE rebuild variant",
    )
    parser.add_argument(
        "--openai-provider",
        default="openai",
        help="Provider for OpenAI rebuild variant",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-5.4",
        help="Model for OpenAI rebuild variant",
    )
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument(
        "--variant-source",
        choices=("builder", "compact"),
        default="builder",
        help="How to build non-baseline variants: full builder from DB or compact packet from sample excerpt",
    )
    parser.add_argument(
        "--output-prefix",
        default="eval/knowledgeos/runs/knowledgeos_paper_memory_triplet_compare_12_v1",
        help="Output prefix without extension",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    _auto_load_dotenv(repo_root)
    sample_csv = (repo_root / args.sample_csv).resolve()
    config_path = (repo_root / args.config).resolve()
    output_prefix = (repo_root / args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(sample_csv)
    if not rows:
        raise SystemExit(f"no rows found in {sample_csv}")

    factory = AppContextFactory(config_path=str(config_path), project_root=repo_root)
    exaone_builder: PaperMemoryBuilder | None = None
    openai_builder: PaperMemoryBuilder | None = None
    exaone_extractor: PaperMemorySchemaExtractor | None = None
    openai_extractor: PaperMemorySchemaExtractor | None = None
    if str(args.variant_source) == "builder":
        exaone_builder = _builder(
            factory,
            provider=str(args.exaone_provider),
            model=str(args.exaone_model),
            timeout_sec=int(args.timeout_sec),
        )
        openai_builder = _builder(
            factory,
            provider=str(args.openai_provider),
            model=str(args.openai_model),
            timeout_sec=int(args.timeout_sec),
        )
    else:
        exaone_extractor = _extractor(
            factory,
            provider=str(args.exaone_provider),
            model=str(args.exaone_model),
            timeout_sec=int(args.timeout_sec),
        )
        openai_extractor = _extractor(
            factory,
            provider=str(args.openai_provider),
            model=str(args.openai_model),
            timeout_sec=int(args.timeout_sec),
        )

    out_rows: list[dict[str, str]] = []
    jsonl_rows: list[dict[str, Any]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "").strip()
        if not paper_id:
            continue
        record: dict[str, Any] = {
            "paper_id": paper_id,
            "title": row.get("title", ""),
            "year": row.get("year", ""),
            "field": row.get("field", ""),
            "source_excerpt": row.get("source_excerpt", ""),
            "excerpt_source": row.get("excerpt_source", ""),
            "quality_flag": row.get("quality_flag", ""),
            "issue_score": row.get("issue_score", ""),
            "latex_core": row.get("latex_core", ""),
            "text_starts_latex": row.get("text_starts_latex", ""),
            "generic_limitation": row.get("generic_limitation", ""),
        }
        record.update(_build_variant(None, paper_id=paper_id, baseline_row=row, variant_name="baseline"))
        if str(args.variant_source) == "builder":
            record.update(_build_variant(exaone_builder, paper_id=paper_id, baseline_row=row, variant_name="exaone"))
            record.update(_build_variant(openai_builder, paper_id=paper_id, baseline_row=row, variant_name="openai"))
        else:
            assert exaone_extractor is not None
            assert openai_extractor is not None
            record.update(_build_compact_variant(exaone_extractor, baseline_row=row, variant_name="exaone"))
            record.update(_build_compact_variant(openai_extractor, baseline_row=row, variant_name="openai"))
        out_rows.append({str(k): str(v) for k, v in record.items()})
        jsonl_rows.append(record)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    csv_path = output_prefix.with_suffix(".csv")
    jsonl_path = output_prefix.with_suffix(".jsonl")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in jsonl_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "sample_csv": str(sample_csv),
        "config": str(config_path),
        "row_count": len(out_rows),
        "exaone_provider": str(args.exaone_provider),
        "exaone_model": str(args.exaone_model),
        "openai_provider": str(args.openai_provider),
        "openai_model": str(args.openai_model),
        "variant_source": str(args.variant_source),
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

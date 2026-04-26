"""CLI surface for shared eval gate workflows."""

from __future__ import annotations

from copy import copy
import csv
import json
from pathlib import Path

import click
from rich.console import Console

from knowledge_hub.application.eval_gate import (
    build_eval_gate_result,
    export_claim_synthesis_eval_template,
    export_document_memory_eval_template,
    export_paper_summary_eval_template,
)
from knowledge_hub.application.eval_center import build_eval_center_summary
from knowledge_hub.application.answer_loop import (
    ANSWER_BACKEND_NAMES,
    CollectRequest,
    autofix_answer_loop,
    collect_answer_loop,
    judge_answer_loop,
    optimize_answer_loop,
    run_answer_loop,
    summarize_answer_loop,
)
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.infrastructure.persistence import SQLiteDatabase

console = Console()
_LOCAL_PROVIDER_NAMES = {"ollama", "pplx-local", "pplx-st", "pplx_st"}


def _validate_cli_payload(config, payload: dict, schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


def _require_existing_path(path: str, *, label: str) -> Path:
    target = Path(str(path)).expanduser()
    if not target.exists():
        raise click.ClickException(f"{label} not found: {target}")
    return target


def _allow_external_default(khub, searcher) -> bool:
    khub_config = getattr(khub, "config", None)
    searcher_config = getattr(searcher, "config", None)
    provider = str(getattr(khub_config, "summarization_provider", "") or "").strip().lower()
    if not provider and khub_config is not None and hasattr(khub_config, "get_nested"):
        provider = str(khub_config.get_nested("summarization", "provider", default="") or "").strip().lower()
    if not provider:
        provider = str(getattr(searcher_config, "summarization_provider", "") or "").strip().lower()
    if not provider and searcher_config is not None and hasattr(searcher_config, "get_nested"):
        provider = str(searcher_config.get_nested("summarization", "provider", default="") or "").strip().lower()
    if not provider:
        return False
    return provider not in _LOCAL_PROVIDER_NAMES


def _read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line_text = str(line or "").strip()
        if not line_text:
            continue
        items.append(json.loads(line_text))
    return items


def _parse_backend_models(model_overrides: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for model_override in model_overrides:
        raw = str(model_override or "").strip()
        if not raw:
            continue
        if "=" not in raw:
            raise click.ClickException(f"backend model override must be backend=model, got: {raw}")
        backend, _, model = raw.partition("=")
        backend = str(backend).strip()
        model = str(model).strip()
        if backend not in ANSWER_BACKEND_NAMES:
            raise click.ClickException(f"unsupported backend model override: {backend}")
        if not model:
            raise click.ClickException(f"backend model override missing model for {backend}")
        parsed[backend] = model
    return parsed


def _resolve_answer_backends(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return tuple(ANSWER_BACKEND_NAMES)
    out: list[str] = []
    for item in values:
        backend_name = str(item or "").strip()
        if backend_name not in ANSWER_BACKEND_NAMES:
            raise click.ClickException(f"unsupported answer backend: {backend_name}")
        if backend_name not in out:
            out.append(backend_name)
    return tuple(out)


def _default_answer_loop_out_dir() -> str:
    stamp = Path.cwd().name
    return str(Path("eval/knowledgeos/runs/answer_loop") / _safe_output_stamp(f"{stamp}"))


def _safe_output_stamp(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value or "").strip())
    return "-".join(part for part in token.split("-") if part) or "latest"


def _resolve_collect_request(
    *,
    queries: str,
    out_dir: str,
    top_k: int,
    mode: str,
    alpha: float,
    answer_backends: tuple[str, ...],
    repo_path: str,
    backend_model_tokens: tuple[str, ...],
) -> CollectRequest:
    return CollectRequest(
        queries_path=str(Path(queries).expanduser()),
        out_dir=str(Path(out_dir).expanduser()),
        top_k=max(1, int(top_k)),
        retrieval_mode=str(mode),
        alpha=float(alpha),
        answer_backends=_resolve_answer_backends(answer_backends),
        repo_path=str(Path(repo_path).expanduser()),
        backend_models=_parse_backend_models(backend_model_tokens),
    )


def _section_eval_row(answer: dict, *, paper_id: str, bucket: str, question: dict, mode: str) -> dict[str, str]:
    v2 = dict(answer.get("v2") or {})
    routing = dict(v2.get("routing") or {})
    runtime = dict(v2.get("runtimeExecution") or {})
    provenance = dict(answer.get("answerProvenance") or v2.get("answerProvenance") or {})
    section_coverage = dict(v2.get("sectionCoverage") or {})
    quality_gate = dict(v2.get("sectionQualityGate") or {})
    return {
        "paper_id": paper_id,
        "bucket": bucket,
        "question_id": str(question.get("question_id") or ""),
        "intent": str(question.get("intent") or ""),
        "question": str(question.get("question") or ""),
        "mode": mode,
        "runtime_used": str(runtime.get("used") or ""),
        "section_decision": str(runtime.get("sectionDecision") or ""),
        "section_block_reason": str(runtime.get("sectionBlockReason") or ""),
        "memory_form": str(routing.get("memoryForm") or ""),
        "answer_provenance_mode": str(provenance.get("mode") or ""),
        "section_coverage_status": str(section_coverage.get("status") or ""),
        "quality_gate_allowed": str(bool(quality_gate.get("allowed"))).lower() if quality_gate else "",
        "answer_text": str(answer.get("answer") or ""),
        "problem_fidelity": "",
        "method_fidelity": "",
        "evidence_fidelity": "",
        "limitation_fidelity": "",
        "scope_accuracy": "",
        "source_traceability": "",
        "abstention_behavior": "",
        "reviewer_notes": "",
    }


def _section_eval_error_row(*, paper_id: str, bucket: str, question: dict, mode: str, error: Exception) -> dict[str, str]:
    row = _section_eval_row({"answer": f"[[ERROR]] {error}"}, paper_id=paper_id, bucket=bucket, question=question, mode=mode)
    row["reviewer_notes"] = f"runtime_error: {error}"
    return row


def _write_section_eval_csv(path: Path, *, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "paper_id",
        "bucket",
        "question_id",
        "intent",
        "question",
        "mode",
        "runtime_used",
        "section_decision",
        "section_block_reason",
        "memory_form",
        "answer_provenance_mode",
        "section_coverage_status",
        "quality_gate_allowed",
        "answer_text",
        "problem_fidelity",
        "method_fidelity",
        "evidence_fidelity",
        "limitation_fidelity",
        "scope_accuracy",
        "source_traceability",
        "abstention_behavior",
        "reviewer_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@click.group("eval")
def eval_group():
    """공통 eval gate 준비/실행"""


@eval_group.command("sectioncards")
@click.option("--paper-subset", required=True, help="JSON file describing the 6-paper eval subset")
@click.option("--questions", required=True, help="JSONL file with sectioncard eval questions")
@click.option("--out-dir", required=True, help="output directory for answer bundle and scored CSV template")
@click.option("--top-k", type=int, default=8, show_default=True)
@click.option("--allow-external/--no-allow-external", default=None, help="answer generation external LLM usage; defaults to configured summarization provider")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def run_sectioncard_eval(ctx, paper_subset, questions, out_dir, top_k, allow_external, as_json):
    """SectionCard-first vs claim-first vs legacy answer bundle 생성"""
    khub = ctx.obj["khub"]
    subset_path = _require_existing_path(paper_subset, label="paper subset")
    questions_path = _require_existing_path(questions, label="questions file")
    output_dir = Path(str(out_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    if not isinstance(subset, list) or not subset:
        raise click.ClickException("paper subset must be a non-empty JSON array")
    questions_rows = _read_jsonl(questions_path)
    if not questions_rows:
        raise click.ClickException("questions file must contain at least one JSONL row")

    paper_meta = {
        str(item.get("paper_id") or "").strip(): dict(item)
        for item in subset
        if str(item.get("paper_id") or "").strip()
    }
    missing = sorted({str(item.get("paper_id") or "").strip() for item in questions_rows if str(item.get("paper_id") or "").strip() not in paper_meta})
    if missing:
        raise click.ClickException(f"questions reference unknown paper ids: {', '.join(missing[:5])}")

    searcher = khub.searcher()
    allow_external_effective = _allow_external_default(khub, searcher) if allow_external is None else bool(allow_external)

    bundle_items: list[dict] = []
    csv_rows: list[dict[str, str]] = []
    mode_map = {
        "section_first": "section_first",
        "claim_first": "claim_first",
        "legacy": "legacy",
    }
    for question in questions_rows:
        paper_id = str(question.get("paper_id") or "").strip()
        meta = paper_meta[paper_id]
        item_payload = {
            "paperId": paper_id,
            "bucket": str(meta.get("bucket") or ""),
            "questionId": str(question.get("question_id") or ""),
            "intent": str(question.get("intent") or ""),
            "question": str(question.get("question") or ""),
            "expectedFocus": str(question.get("expected_focus") or ""),
            "answers": {},
        }
        for mode, ask_v2_mode in mode_map.items():
            try:
                answer = searcher.generate_answer(
                    str(question.get("question") or ""),
                    top_k=max(1, int(top_k)),
                    source_type="paper",
                    retrieval_mode="hybrid",
                    alpha=0.7,
                    allow_external=allow_external_effective,
                    metadata_filter={"paper_id": paper_id},
                    ask_v2_mode=ask_v2_mode,
                    answer_route_override="api" if allow_external_effective else None,
                )
                v2 = dict(answer.get("v2") or {})
                runtime = dict(v2.get("runtimeExecution") or {})
                routing = dict(v2.get("routing") or {})
                provenance = dict(answer.get("answerProvenance") or v2.get("answerProvenance") or {})
                item_payload["answers"][mode] = {
                    "answer": str(answer.get("answer") or ""),
                    "warnings": list(answer.get("warnings") or []),
                    "runtime": runtime,
                    "memoryForm": str(routing.get("memoryForm") or ""),
                    "answerProvenanceMode": str(provenance.get("mode") or ""),
                    "sectionCoverage": dict(v2.get("sectionCoverage") or {}),
                    "qualityGate": dict(v2.get("sectionQualityGate") or {}),
                }
                csv_rows.append(
                    _section_eval_row(
                        answer,
                        paper_id=paper_id,
                        bucket=str(meta.get("bucket") or ""),
                        question=question,
                        mode=mode,
                    )
                )
            except Exception as error:
                item_payload["answers"][mode] = {
                    "answer": "",
                    "warnings": [f"runtime_error: {error}"],
                    "runtime": {"used": "error", "sectionDecision": "", "sectionBlockReason": "", "fallbackReason": ""},
                    "memoryForm": "",
                    "answerProvenanceMode": "",
                    "sectionCoverage": {},
                    "qualityGate": {},
                }
                csv_rows.append(
                    _section_eval_error_row(
                        paper_id=paper_id,
                        bucket=str(meta.get("bucket") or ""),
                        question=question,
                        mode=mode,
                        error=error,
                    )
                )
        bundle_items.append(item_payload)

    bundle_payload = {
        "schema": "knowledge-hub.sectioncard.eval.bundle.v1",
        "status": "ok",
        "paperSubsetPath": str(subset_path),
        "questionsPath": str(questions_path),
        "outDir": str(output_dir),
        "allowExternal": allow_external_effective,
        "questionCount": len(questions_rows),
        "answerCount": len(csv_rows),
        "items": bundle_items,
    }
    bundle_path = output_dir / "sectioncard_answer_bundle.json"
    sheet_path = output_dir / "sectioncard_eval_sheet.csv"
    bundle_path.write_text(json.dumps(bundle_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_section_eval_csv(sheet_path, rows=csv_rows)
    bundle_payload["artifactPaths"] = {
        "answerBundlePath": str(bundle_path),
        "evalSheetPath": str(sheet_path),
    }
    if as_json:
        console.print_json(data=bundle_payload)
        return
    console.print(
        f"[bold]sectioncards eval[/bold] questions={bundle_payload['questionCount']} "
        f"answers={bundle_payload['answerCount']} out={output_dir}"
    )
    console.print(f"[dim]allow_external={allow_external_effective}[/dim]")
    console.print(f"[dim]answer bundle: {bundle_path}[/dim]")
    console.print(f"[dim]eval sheet: {sheet_path}[/dim]")


@eval_group.command("prepare-document-memory")
@click.option("--db", "db_path", default="data/knowledge.db", show_default=True, help="SQLite database path")
@click.option(
    "--queries",
    default="docs/research/document-memory-eval-queries-v1.txt",
    show_default=True,
    help="Text file with one query per line",
)
@click.option(
    "--out",
    default="docs/experiments/document_memory_eval_template.csv",
    show_default=True,
    help="Output CSV path",
)
@click.option("--top-k", type=int, default=3, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def prepare_document_memory(ctx, db_path, queries, out, top_k, as_json):
    """document-memory manual eval CSV template 생성"""
    khub = ctx.obj["khub"]
    db_file = _require_existing_path(db_path, label="db path")
    queries_file = _require_existing_path(queries, label="queries file")
    payload = export_document_memory_eval_template(
        SQLiteDatabase(str(db_file)),
        db_path=str(db_file),
        queries_path=str(queries_file),
        out_path=out,
        top_k=max(1, int(top_k)),
    )
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]document-memory eval prepare[/bold] queries={payload.get('queryCount')} "
        f"rows={payload.get('rowCount')} out={payload.get('outPath')}"
    )


@eval_group.command("prepare-claim-synthesis")
@click.option("--db", "db_path", default="data/knowledge.db", show_default=True, help="SQLite database path")
@click.option("--claim-id", "claim_ids", multiple=True, help="target claim id")
@click.option("--paper-id", "paper_ids", multiple=True, help="target paper id / arXiv id")
@click.option("--task", default="", help="normalized task filter")
@click.option("--dataset", default="", help="normalized dataset filter")
@click.option("--metric", default="", help="normalized metric filter")
@click.option(
    "--out",
    default="docs/experiments/claim_synthesis_eval_template.csv",
    show_default=True,
    help="Output CSV path",
)
@click.option("--limit", type=int, default=200, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def prepare_claim_synthesis(ctx, db_path, claim_ids, paper_ids, task, dataset, metric, out, limit, as_json):
    """claim-synthesis manual eval CSV template 생성"""
    khub = ctx.obj["khub"]
    db_file = _require_existing_path(db_path, label="db path")
    if not claim_ids and not paper_ids:
        raise click.ClickException("하나의 대상만 지정해야 합니다: --claim-id 또는 --paper-id")
    payload = export_claim_synthesis_eval_template(
        SQLiteDatabase(str(db_file)),
        khub.config,
        out_path=out,
        claim_ids=list(claim_ids),
        paper_ids=list(paper_ids),
        task=str(task or "").strip(),
        dataset=str(dataset or "").strip(),
        metric=str(metric or "").strip(),
        limit=max(1, int(limit)),
    )
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]claim-synthesis eval prepare[/bold] rows={payload.get('rowCount')} out={payload.get('outPath')}"
    )


@eval_group.command("prepare-paper-summary")
@click.option("--db", "db_path", default="data/knowledge.db", show_default=True, help="SQLite database path")
@click.option("--paper-id", "paper_ids", multiple=True, help="target paper id / arXiv id")
@click.option("--paper-id-file", default="", help="text file with one paper id per line")
@click.option(
    "--out",
    default="docs/experiments/paper_summary_eval_template.csv",
    show_default=True,
    help="Output CSV path",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def prepare_paper_summary(ctx, db_path, paper_ids, paper_id_file, out, as_json):
    """paper-summary manual eval CSV template 생성"""
    khub = ctx.obj["khub"]
    db_file = _require_existing_path(db_path, label="db path")
    paper_ids_file = _require_existing_path(paper_id_file, label="paper id file") if str(paper_id_file or "").strip() else None
    if not paper_ids and paper_ids_file is None:
        raise click.ClickException("하나의 대상은 필요합니다: --paper-id 또는 --paper-id-file")
    payload = export_paper_summary_eval_template(
        SQLiteDatabase(str(db_file)),
        khub.config,
        out_path=out,
        paper_ids=list(paper_ids),
        paper_ids_path=str(paper_ids_file) if paper_ids_file is not None else None,
    )
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return
    console.print(
        f"[bold]paper-summary eval prepare[/bold] rows={payload.get('rowCount')} out={payload.get('outPath')}"
    )


@eval_group.command("run")
@click.option(
    "--profile",
    type=click.Choice(["retrieval-core", "memory-promotion", "memory-router-v1", "ask-v2", "all"], case_sensitive=False),
    default="all",
    show_default=True,
)
@click.option("--db", "db_path", default="data/knowledge.db", show_default=True, help="SQLite database path")
@click.option("--retrieval-csv", default="", help="Labeled retrieval CSV for retrieval-core eval")
@click.option(
    "--document-memory-csv",
    default="docs/experiments/document_memory_eval_template.csv",
    show_default=True,
)
@click.option(
    "--paper-memory-cases",
    default="tests/fixtures/paper_memory_eval/cases.json",
    show_default=True,
)
@click.option("--memory-router-csv", default="", help="Labeled candidate CSV for memory-router-v1 eval")
@click.option("--memory-router-baseline-csv", default="", help="Labeled baseline CSV for memory-router-v1 eval")
@click.option("--memory-router-label-col", default="label", show_default=True)
@click.option("--memory-router-no-result-col", default="no_result", show_default=True)
@click.option("--memory-router-temporal-col", default="temporal_query", show_default=True)
@click.option("--memory-router-wrong-era-col", default="wrong_era", show_default=True)
@click.option("--ask-v2-csv", default="", help="Labeled ask-v2 CSV for manual gate")
@click.option("--ask-v2-label-col", default="label", show_default=True)
@click.option("--ask-v2-wrong-source-col", default="wrong_source", show_default=True)
@click.option("--ask-v2-no-result-col", default="no_result", show_default=True)
@click.option("--ask-v2-should-abstain-col", default="should_abstain", show_default=True)
@click.option("--ask-v2-wrong-era-col", default="wrong_era", show_default=True)
@click.option("--claim-synthesis-csv", default="", help="Labeled claim-synthesis CSV for optional memory-promotion eval")
@click.option("--retrieval-label-col", default="label_context", show_default=True)
@click.option("--retrieval-source-col", default="context_source", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def run_eval_gate(
    ctx,
    profile,
    db_path,
    retrieval_csv,
    document_memory_csv,
    paper_memory_cases,
    memory_router_csv,
    memory_router_baseline_csv,
    memory_router_label_col,
    memory_router_no_result_col,
    memory_router_temporal_col,
    memory_router_wrong_era_col,
    ask_v2_csv,
    ask_v2_label_col,
    ask_v2_wrong_source_col,
    ask_v2_no_result_col,
    ask_v2_should_abstain_col,
    ask_v2_wrong_era_col,
    claim_synthesis_csv,
    retrieval_label_col,
    retrieval_source_col,
    as_json,
):
    """retrieval/memory promotion 공통 eval gate 실행"""
    khub = ctx.obj["khub"]
    profile_name = str(profile).strip().lower()
    db_file = _require_existing_path(db_path, label="db path")

    if profile_name in {"retrieval-core", "memory-router-v1", "all"}:
        retrieval_file = _require_existing_path(retrieval_csv, label="retrieval csv")
    else:
        retrieval_file = None

    if profile_name in {"memory-promotion", "memory-router-v1", "all"}:
        document_memory_file = _require_existing_path(document_memory_csv, label="document-memory csv")
        paper_memory_cases_file = _require_existing_path(paper_memory_cases, label="paper-memory cases file")
    else:
        document_memory_file = None
        paper_memory_cases_file = None

    if profile_name == "memory-router-v1":
        memory_router_file = _require_existing_path(memory_router_csv, label="memory-router csv")
        memory_router_baseline_file = _require_existing_path(
            memory_router_baseline_csv,
            label="memory-router baseline csv",
        )
    else:
        memory_router_file = None
        memory_router_baseline_file = None

    if profile_name == "ask-v2":
        ask_v2_file = _require_existing_path(ask_v2_csv, label="ask-v2 csv")
    elif profile_name == "all" and str(ask_v2_csv or "").strip():
        ask_v2_file = _require_existing_path(ask_v2_csv, label="ask-v2 csv")
    else:
        ask_v2_file = None

    searcher = None
    searcher_error = ""
    if profile_name in {"retrieval-core", "memory-router-v1", "all"}:
        try:
            searcher = khub.searcher()
        except Exception as error:  # pragma: no cover - exercised through gate payload tests
            searcher_error = str(error)

    payload = build_eval_gate_result(
        config=khub.config,
        sqlite_db=SQLiteDatabase(str(db_file)),
        profile=profile_name,
        retrieval_csv=str(retrieval_file) if retrieval_file is not None else None,
        retrieval_label_col=str(retrieval_label_col),
        retrieval_source_col=str(retrieval_source_col),
        document_memory_csv=str(document_memory_file) if document_memory_file is not None else str(document_memory_csv),
        paper_memory_cases=str(paper_memory_cases_file) if paper_memory_cases_file is not None else str(paper_memory_cases),
        claim_synthesis_csv=str(claim_synthesis_csv or "").strip() or None,
        memory_router_csv=str(memory_router_file) if memory_router_file is not None else None,
        memory_router_baseline_csv=str(memory_router_baseline_file) if memory_router_baseline_file is not None else None,
        memory_router_label_col=str(memory_router_label_col),
        memory_router_no_result_col=str(memory_router_no_result_col),
        memory_router_temporal_col=str(memory_router_temporal_col),
        memory_router_wrong_era_col=str(memory_router_wrong_era_col),
        ask_v2_csv=str(ask_v2_file) if ask_v2_file is not None else None,
        ask_v2_label_col=str(ask_v2_label_col),
        ask_v2_wrong_source_col=str(ask_v2_wrong_source_col),
        ask_v2_no_result_col=str(ask_v2_no_result_col),
        ask_v2_should_abstain_col=str(ask_v2_should_abstain_col),
        ask_v2_wrong_era_col=str(ask_v2_wrong_era_col),
        searcher=searcher,
        searcher_error=searcher_error,
    )
    _validate_cli_payload(khub.config, payload, payload["schema"])

    if as_json:
        console.print_json(data=payload)
        if payload.get("status") == "fail":
            raise click.exceptions.Exit(1)
        return

    gate = payload.get("gate") or {}
    console.print(
        f"[bold]eval gate[/bold] profile={payload.get('profile')} status={payload.get('status')} "
        f"{gate.get('summary', '')}"
    )
    for check in list(gate.get("checks") or [])[:20]:
        console.print(
            f"- [{check.get('status')}] {check.get('name')}: {check.get('summary')} "
            f"(observed={check.get('observed')} threshold={check.get('threshold')})"
        )
    if gate.get("recommendation"):
        console.print(f"[dim]{gate.get('recommendation')}[/dim]")
    for warning in list(payload.get("warnings") or [])[:10]:
        console.print(f"[yellow]- {warning}[/yellow]")
    if payload.get("status") == "fail":
        raise click.exceptions.Exit(1)


@eval_group.command("center")
@click.option(
    "--runs-root",
    default="eval/knowledgeos/runs",
    show_default=True,
    help="Eval run artifact root, often a symlink to ~/.khub/eval/knowledgeos/runs.",
)
@click.option(
    "--queries-dir",
    default="eval/knowledgeos/queries",
    show_default=True,
    help="Eval query CSV directory.",
)
@click.option(
    "--failure-bank-path",
    default="~/.khub/eval/knowledgeos/failures/failure_bank.jsonl",
    show_default=True,
    help="Failure Bank JSONL path; read-only in this summary surface.",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def eval_center(ctx, runs_root, queries_dir, failure_bank_path, as_json):
    """현재 eval 자산과 최신 결과를 읽기 전용으로 요약합니다."""
    khub = ctx.obj["khub"]
    payload = build_eval_center_summary(
        runs_root=runs_root,
        queries_dir=queries_dir,
        failure_bank_path=failure_bank_path,
        repo_root=Path.cwd(),
    )
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return

    source = payload.get("sourceQuality") or {}
    base = source.get("baseObservation") or {}
    detail = source.get("detailObservation") or {}
    answer = payload.get("answerLoop") or {}
    summary = answer.get("summary") or {}
    inventory = payload.get("queryInventory") or {}
    console.print(
        f"[bold]eval center[/bold] status={payload.get('status')} "
        f"warnings={len(payload.get('warnings') or [])}"
    )
    console.print(
        f"- source_quality: base={base.get('decision') or base.get('status') or 'unknown'} "
        f"detail={detail.get('decision') or detail.get('status') or 'unknown'}"
    )
    console.print(
        f"- answer_loop: rows={summary.get('rowCount', 0)} "
        f"status={summary.get('status') or 'missing'} run={answer.get('latestRunDir') or 'none'}"
    )
    console.print(f"- query_sets: {inventory.get('count', 0)} dir={payload.get('queriesDir')}")
    for warning in list(payload.get("warnings") or [])[:10]:
        console.print(f"[yellow]- {warning}[/yellow]")


@eval_group.group("answer-loop")
def answer_loop_group():
    """Frozen-packet answer evaluation and Codex autofix loop."""


@answer_loop_group.command("collect")
@click.option(
    "--queries",
    default="eval/knowledgeos/queries/user_answer_eval_queries_v1.csv",
    show_default=True,
)
@click.option(
    "--out-dir",
    default="eval/knowledgeos/runs/answer_loop/latest",
    show_default=True,
)
@click.option("--top-k", type=int, default=8, show_default=True)
@click.option("--mode", default="hybrid", type=click.Choice(["semantic", "keyword", "hybrid"]), show_default=True)
@click.option("--alpha", type=float, default=0.7, show_default=True)
@click.option("--answer-backend", "answer_backends", multiple=True, type=click.Choice(ANSWER_BACKEND_NAMES), help="Repeat to limit backends; default runs all three")
@click.option("--repo-path", default=".", show_default=True)
@click.option("--backend-model", "backend_model_tokens", multiple=True, help="backend=model override, repeatable")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def collect_answer_loop_cmd(
    ctx,
    queries,
    out_dir,
    top_k,
    mode,
    alpha,
    answer_backends,
    repo_path,
    backend_model_tokens,
    as_json,
):
    """Collect frozen packets and backend answers."""
    khub = ctx.obj["khub"]
    request = _resolve_collect_request(
        queries=queries,
        out_dir=out_dir,
        top_k=top_k,
        mode=mode,
        alpha=alpha,
        answer_backends=tuple(answer_backends),
        repo_path=repo_path,
        backend_model_tokens=tuple(backend_model_tokens),
    )
    payload = collect_answer_loop(factory=khub, request=request)
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return
    artifacts = payload.get("artifactPaths") or {}
    console.print(
        f"[bold]answer-loop collect[/bold] rows={payload.get('rowCount')} packets={payload.get('packetCount')} "
        f"out={request.out_dir}"
    )
    console.print(f"[dim]csv: {artifacts.get('csvPath')}[/dim]")
    console.print(f"[dim]records: {artifacts.get('recordsPath')}[/dim]")


@answer_loop_group.command("judge")
@click.option("--collect-manifest", required=True, help="Path to answer_loop_collect_manifest.json")
@click.option("--judge-model", default="gpt-5", show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def judge_answer_loop_cmd(ctx, collect_manifest, judge_model, as_json):
    """Judge collected backend answers with GPT-family evaluator."""
    khub = ctx.obj["khub"]
    payload = judge_answer_loop(
        factory=khub,
        collect_manifest_path=str(Path(collect_manifest).expanduser()),
        judge_model=str(judge_model),
    )
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return
    artifacts = payload.get("artifactPaths") or {}
    console.print(
        f"[bold]answer-loop judge[/bold] rows={payload.get('rowCount')} "
        f"model={payload.get('judgeModel')}"
    )
    console.print(f"[dim]judged csv: {artifacts.get('judgedCsvPath')}[/dim]")
    console.print(f"[dim]judge artifacts: {artifacts.get('judgeArtifactsPath')}[/dim]")


@answer_loop_group.command("summarize")
@click.option("--judge-manifest", required=True, help="Path to answer_loop_judge_manifest.json")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def summarize_answer_loop_cmd(ctx, judge_manifest, as_json):
    """Summarize judged rows into scores and failure buckets."""
    khub = ctx.obj["khub"]
    payload = summarize_answer_loop(judge_manifest_path=str(Path(judge_manifest).expanduser()))
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        return
    overall = payload.get("overall") or {}
    console.print(
        f"[bold]answer-loop summary[/bold] rows={payload.get('rowCount')} "
        f"label={overall.get('predLabelScore')} groundedness={overall.get('predGroundednessScore')} "
        f"source_accuracy={overall.get('predSourceAccuracyScore')}"
    )
    for bucket, count in sorted((payload.get("failureBucketCounts") or {}).items()):
        console.print(f"- {bucket}: {count}")


@answer_loop_group.command("autofix")
@click.option("--judge-manifest", required=True, help="Path to answer_loop_judge_manifest.json")
@click.option("--repo-path", default=".", show_default=True)
@click.option("--allow-dirty/--no-allow-dirty", default=False, show_default=True)
@click.option("--patch-model", default="", help="Optional Codex patch model override")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def autofix_answer_loop_cmd(ctx, judge_manifest, repo_path, allow_dirty, patch_model, as_json):
    """Build failure cards and run Codex patch mode once."""
    khub = ctx.obj["khub"]
    payload = autofix_answer_loop(
        factory=khub,
        judge_manifest_path=str(Path(judge_manifest).expanduser()),
        repo_path=str(Path(repo_path).expanduser()),
        allow_dirty=bool(allow_dirty),
        patch_model=str(patch_model or ""),
    )
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        if payload.get("status") not in {"ok", "blocked"}:
            raise click.exceptions.Exit(1)
        return
    console.print(
        f"[bold]answer-loop autofix[/bold] status={payload.get('status')} repo={payload.get('repoPath')}"
    )
    for warning in list(payload.get("warnings") or []):
        console.print(f"[yellow]- {warning}[/yellow]")
    for path in list(payload.get("changedFiles") or []):
        console.print(f"[dim]changed: {path}[/dim]")
    if payload.get("status") not in {"ok", "blocked"}:
        raise click.exceptions.Exit(1)


@answer_loop_group.command("run")
@click.option(
    "--queries",
    default="eval/knowledgeos/queries/user_answer_eval_queries_v1.csv",
    show_default=True,
)
@click.option(
    "--out-dir",
    default="eval/knowledgeos/runs/answer_loop/latest",
    show_default=True,
)
@click.option("--top-k", type=int, default=8, show_default=True)
@click.option("--mode", default="hybrid", type=click.Choice(["semantic", "keyword", "hybrid"]), show_default=True)
@click.option("--alpha", type=float, default=0.7, show_default=True)
@click.option("--answer-backend", "answer_backends", multiple=True, type=click.Choice(ANSWER_BACKEND_NAMES), help="Repeat to limit backends; default runs all three")
@click.option("--judge-model", default="gpt-5", show_default=True)
@click.option("--max-attempts", type=int, default=3, show_default=True)
@click.option("--repo-path", default=".", show_default=True)
@click.option("--allow-dirty/--no-allow-dirty", default=False, show_default=True)
@click.option("--backend-model", "backend_model_tokens", multiple=True, help="backend=model override, repeatable")
@click.option("--patch-model", default="", help="Optional Codex patch model override")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def run_answer_loop_cmd(
    ctx,
    queries,
    out_dir,
    top_k,
    mode,
    alpha,
    answer_backends,
    judge_model,
    max_attempts,
    repo_path,
    allow_dirty,
    backend_model_tokens,
    patch_model,
    as_json,
):
    """Run collect -> judge -> summarize -> autofix until stop conditions fire."""
    khub = ctx.obj["khub"]
    request = _resolve_collect_request(
        queries=queries,
        out_dir=out_dir,
        top_k=top_k,
        mode=mode,
        alpha=alpha,
        answer_backends=tuple(answer_backends),
        repo_path=repo_path,
        backend_model_tokens=tuple(backend_model_tokens),
    )
    payload = run_answer_loop(
        factory=khub,
        request=request,
        judge_model=str(judge_model),
        max_attempts=max(1, int(max_attempts)),
        repo_path=str(Path(repo_path).expanduser()),
        allow_dirty=bool(allow_dirty),
        patch_model=str(patch_model or ""),
    )
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        if payload.get("status") != "ok":
            raise click.exceptions.Exit(1)
        return
    console.print(
        f"[bold]answer-loop run[/bold] attempts={payload.get('attemptCount')} "
        f"best={payload.get('bestAttempt')} stopped={payload.get('stoppedReason')}"
    )
    overall = payload.get("overall") or {}
    if overall:
        console.print(
            f"[dim]label={overall.get('predLabelScore')} "
            f"groundedness={overall.get('predGroundednessScore')} "
            f"source_accuracy={overall.get('predSourceAccuracyScore')}[/dim]"
        )


@answer_loop_group.command("optimize")
@click.option(
    "--queries",
    default="eval/knowledgeos/queries/user_answer_eval_queries_v1.csv",
    show_default=True,
)
@click.option(
    "--out-dir",
    default="eval/knowledgeos/runs/answer_loop/latest",
    show_default=True,
)
@click.option("--top-k", type=int, default=8, show_default=True)
@click.option("--mode", default="hybrid", type=click.Choice(["semantic", "keyword", "hybrid"]), show_default=True)
@click.option("--alpha", type=float, default=0.7, show_default=True)
@click.option("--repo-path", default=".", show_default=True)
@click.option("--candidate-count", type=int, default=2, show_default=True)
@click.option("--max-rounds", type=int, default=3, show_default=True)
@click.option("--daily-token-budget-estimate", type=int, default=None)
@click.option("--judge-budget-ratio", type=float, default=0.10, show_default=True)
@click.option("--max-total-candidates", type=int, default=60, show_default=True)
@click.option("--time-limit-minutes", type=int, default=0, show_default=True)
@click.option("--improvement-epsilon", type=float, default=0.01, show_default=True)
@click.option("--generator-model", default="", help="Optional Codex generator model override")
@click.option("--judge-model", default="", help="Optional Codex judge model override")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def optimize_answer_loop_cmd(
    ctx,
    queries,
    out_dir,
    top_k,
    mode,
    alpha,
    repo_path,
    candidate_count,
    max_rounds,
    daily_token_budget_estimate,
    judge_budget_ratio,
    max_total_candidates,
    time_limit_minutes,
    improvement_epsilon,
    generator_model,
    judge_model,
    as_json,
):
    """Run a Codex-only non-mutating overnight answer optimizer."""
    khub = ctx.obj["khub"]
    request = CollectRequest(
        queries_path=str(Path(queries).expanduser()),
        out_dir=str(Path(out_dir).expanduser()),
        top_k=max(1, int(top_k)),
        retrieval_mode=str(mode),
        alpha=float(alpha),
        answer_backends=("codex_mcp",),
        repo_path=str(Path(repo_path).expanduser()),
        backend_models={},
    )
    payload = optimize_answer_loop(
        factory=khub,
        request=request,
        repo_path=str(Path(repo_path).expanduser()),
        generator_model=str(generator_model or ""),
        judge_model=str(judge_model or ""),
        candidate_count=max(1, int(candidate_count)),
        max_rounds=max(1, int(max_rounds)),
        daily_token_budget_estimate=daily_token_budget_estimate,
        judge_budget_ratio=float(judge_budget_ratio),
        max_total_candidates=max(1, int(max_total_candidates)),
        time_limit_minutes=max(0, int(time_limit_minutes)),
        improvement_epsilon=max(0.0, float(improvement_epsilon)),
    )
    _validate_cli_payload(khub.config, payload, payload["schema"])
    if as_json:
        console.print_json(data=payload)
        if payload.get("status") != "ok":
            raise click.exceptions.Exit(1)
        return
    console.print(
        f"[bold]answer-loop optimize[/bold] rounds={payload.get('roundCount')} stopped={payload.get('stopReason')}"
    )
    overall = payload.get("overall") or {}
    budget = payload.get("budget") or {}
    console.print(
        f"[dim]label={overall.get('predLabelScore')} groundedness={overall.get('predGroundednessScore')} "
        f"source_accuracy={overall.get('predSourceAccuracyScore')} judge_cap={budget.get('judgeBudgetCap')}[/dim]"
    )


eval_compat_group = copy(eval_group)
eval_compat_group.help = "Compatibility alias for `khub labs eval`."
eval_compat_group.short_help = "Compatibility alias for `khub labs eval`."

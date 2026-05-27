#!/usr/bin/env python3
"""Build whole-local-corpus visual annotation expansion batches.

This report-only/operator-pack builder scans already-local paper PDFs, selects a
category-balanced batch set for manual web GPT/Pro visual annotation, renders
context crops, and packages Finder-friendly upload folders. It does not call
models, download papers, scan the vault, write vector indexes, mutate databases,
or promote derived visual text to evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_annotation_expansion_attachment_pack import (
    VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
    build_visual_annotation_expansion_attachment_pack,
    default_papers_root,
    sanitized_report_ref,
    write_visual_annotation_expansion_attachment_pack,
)
from knowledge_hub.papers.visual_annotation_expansion_manual_output_capture import (
    VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BATCH_TEMPLATE_SCHEMA_ID,
    VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID,
    build_visual_annotation_expansion_manual_run_packet,
    build_visual_annotation_expansion_operator_handoff,
    build_visual_annotation_expansion_web_output_template,
    build_visual_annotation_expansion_web_run_batch_template,
    build_visual_annotation_expansion_web_run_bundle,
    load_json,
    write_visual_annotation_expansion_manual_run_packet,
    write_visual_annotation_expansion_operator_handoff,
    write_visual_annotation_expansion_web_output_template,
    write_visual_annotation_expansion_web_run_bundle,
)
from knowledge_hub.papers.visual_annotation_expansion_pack_design import (
    VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID,
    VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
    build_visual_annotation_expansion_pack_design,
    write_visual_annotation_expansion_pack_design,
)
from knowledge_hub.papers.visual_annotation_expansion_upload_bundle import (
    VISUAL_ANNOTATION_EXPANSION_WEB_UPLOAD_BUNDLE_SCHEMA_ID,
    build_visual_annotation_expansion_web_upload_bundle,
    write_visual_annotation_expansion_web_upload_bundle,
)
from knowledge_hub.papers.visual_layout_candidate_list_report import (
    VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID,
    build_visual_layout_candidate_list_report,
    discover_local_paper_specs,
)


CORPUS_PACK_ID = "visual_annotation_corpus_expansion_pack_005"
DEFAULT_MAX_CANDIDATES = 40
DEFAULT_BATCH_SIZE = 8
DEFAULT_TYPE_QUOTAS = {
    "table_region": 8,
    "figure_caption_region": 8,
    "equation_region": 8,
    "layout_region": 8,
    "image_region": 8,
}
REPORTS_ROOT = PROJECT_ROOT / "eval/knowledgeos/reports"
DEFAULT_WEB_PACK_PATH = REPORTS_ROOT / "visual_annotation_web_pack_001.v1.json"
DEFAULT_SOURCE_DRY_RUN_PATHS = (
    REPORTS_ROOT / "visual_retrieval_hint_candidate_store_dry_run.v1.json",
    REPORTS_ROOT / "visual_retrieval_hint_candidate_store_expansion_dry_run.v1.json",
    REPORTS_ROOT / "visual_retrieval_hint_candidate_store_expansion_dry_run_003.v1.json",
    REPORTS_ROOT / "visual_retrieval_hint_candidate_store_expansion_dry_run_004.v1.json",
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-root", type=Path, default=default_papers_root())
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--refresh-candidate-report",
        action="store_true",
        help="Rescan local PDFs even if the full local candidate report already exists.",
    )
    parser.add_argument("--source-web-pack", type=Path, default=DEFAULT_WEB_PACK_PATH)
    parser.add_argument(
        "--source-dry-run",
        type=Path,
        action="append",
        default=None,
        help="Prior candidate-store dry-run report to exclude. Repeat to supply multiple reports.",
    )
    parser.add_argument("--pack-id", default=CORPUS_PACK_ID)
    parser.add_argument("--json", action="store_true", help="Print full summary JSON to stdout.")
    return parser.parse_args(argv)


def _combined_previous_dry_run_report(dry_run_reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for report in dry_run_reports:
        for row in list(report.get("dryRunRowsDetail") or []):
            if not isinstance(row, dict):
                continue
            source_candidate_id = str(row.get("sourceCandidateId") or "").strip()
            if not source_candidate_id or source_candidate_id in seen:
                continue
            rows.append({"sourceCandidateId": source_candidate_id})
            seen.add(source_candidate_id)
    return {
        "schema": VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_EXPANSION_DRY_RUN_SCHEMA_ID,
        "status": "ready",
        "dryRunRowsDetail": rows,
    }


def _combined_ref(paths: list[Path]) -> str:
    return " + ".join(sanitized_report_ref(path, project_root=PROJECT_ROOT) for path in paths)


def _candidate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in list(report.get("candidateRowsDetail") or []) if isinstance(row, dict)]


def _prior_candidate_ids(web_pack: dict[str, Any], dry_run_report: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for row in list(web_pack.get("packRowsDetail") or []):
        if isinstance(row, dict) and row.get("sourceCandidateId"):
            ids.add(str(row.get("sourceCandidateId")))
    for row in list(dry_run_report.get("dryRunRowsDetail") or []):
        if isinstance(row, dict) and row.get("sourceCandidateId"):
            ids.add(str(row.get("sourceCandidateId")))
    return ids


def _preferred_paper_ids(
    candidate_report: dict[str, Any],
    *,
    previous_candidate_ids: set[str],
    type_quotas: dict[str, int],
) -> list[str]:
    allowed_types = set(type_quotas)
    totals: Counter[str] = Counter()
    type_sets: dict[str, set[str]] = defaultdict(set)
    previous_papers: set[str] = set()
    for row in _candidate_rows(candidate_report):
        candidate_id = str(row.get("candidateId") or "")
        paper_id = str(row.get("paperId") or "")
        candidate_type = str(row.get("candidateType") or "")
        if candidate_id in previous_candidate_ids:
            previous_papers.add(paper_id)
            continue
        if row.get("blockerReason") or candidate_type not in allowed_types or not paper_id:
            continue
        totals[paper_id] += 1
        type_sets[paper_id].add(candidate_type)
    return sorted(
        totals,
        key=lambda paper_id: (
            paper_id in previous_papers,
            -len(type_sets[paper_id]),
            -totals[paper_id],
            paper_id,
        ),
    )


def _type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("candidateType") or "") for row in rows if not row.get("blockerReason"))
    return {
        "image_region": counts.get("image_region", 0),
        "figure_caption_region": counts.get("figure_caption_region", 0),
        "table_region": counts.get("table_region", 0),
        "equation_region": counts.get("equation_region", 0),
        "layout_region": counts.get("layout_region", 0),
    }


def _render_corpus_candidate_markdown(report: dict[str, Any], *, top_n: int = 60) -> str:
    rows = _candidate_rows(report)
    counts = dict(report.get("counts") or {})
    by_paper: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("blockerReason"):
            by_paper[str(row.get("paperId") or "")]["blocked"] += 1
            continue
        by_paper[str(row.get("paperId") or "")][str(row.get("candidateType") or "")] += 1
    paper_rows = []
    for paper_id, counter in by_paper.items():
        total = sum(counter.values())
        paper_rows.append((total, paper_id, counter))
    paper_rows.sort(key=lambda item: (-item[0], item[1]))
    lines = [
        "# Visual Layout Corpus Candidate List Report",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- inputPaperRows: `{counts.get('inputPaperRows')}`",
        f"- candidateRows: `{counts.get('candidateRows')}`",
        f"- figureCandidateRows: `{counts.get('figureCandidateRows')}`",
        f"- tableCandidateRows: `{counts.get('tableCandidateRows')}`",
        f"- equationCandidateRows: `{counts.get('equationCandidateRows')}`",
        f"- layoutCandidateRows: `{counts.get('layoutCandidateRows')}`",
        f"- imageCandidateRows: `{counts.get('imageCandidateRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        "",
        "## Top Papers By Candidate Count",
        "",
        "| # | paperId | total | table | figure | equation | layout | image | blocked |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, (total, paper_id, counter) in enumerate(paper_rows[:top_n], start=1):
        lines.append(
            "| {index} | `{paper}` | {total} | {table} | {figure} | {equation} | {layout} | {image} | {blocked} |".format(
                index=index,
                paper=paper_id,
                total=total,
                table=counter.get("table_region", 0),
                figure=counter.get("figure_caption_region", 0),
                equation=counter.get("equation_region", 0),
                layout=counter.get("layout_region", 0),
                image=counter.get("image_region", 0),
                blocked=counter.get("blocked", 0),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _write_candidate_report(report: dict[str, Any], *, report_json: Path, report_md: Path) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(_render_corpus_candidate_markdown(report), encoding="utf-8")
    return {"json": str(report_json), "markdown": str(report_md)}


def _validate_or_raise(payload: dict[str, Any], schema_id: str, label: str) -> None:
    validation = validate_payload(payload, schema_id, strict=True)
    if not validation.ok:
        raise ValueError(f"{label} schema validation failed: " + "; ".join(str(error) for error in validation.errors))


def _remove_generated_dirs(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            shutil.rmtree(path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    papers_root = args.papers_root.expanduser()
    dry_run_paths = [path.expanduser() for path in (args.source_dry_run or DEFAULT_SOURCE_DRY_RUN_PATHS)]
    web_pack_path = args.source_web_pack.expanduser()

    candidate_json = REPORTS_ROOT / "visual_layout_candidate_list_report_corpus.full.local.json"
    candidate_md = REPORTS_ROOT / "visual_layout_candidate_list_report_corpus.v1.md"
    pack_json = REPORTS_ROOT / "visual_annotation_expansion_pack_design_005.v1.json"
    pack_md = REPORTS_ROOT / "visual_annotation_expansion_pack_design_005.v1.md"
    attachment_dir = REPORTS_ROOT / "visual_annotation_expansion_attachment_pack_005"
    attachment_json = REPORTS_ROOT / "visual_annotation_expansion_attachment_pack_005.v1.json"
    attachment_md = REPORTS_ROOT / "visual_annotation_expansion_attachment_pack_005.v1.md"
    manual_json = REPORTS_ROOT / "visual_annotation_expansion_manual_run_packet_005.v1.json"
    manual_md = REPORTS_ROOT / "visual_annotation_expansion_manual_run_packet_005.v1.md"
    handoff_json = REPORTS_ROOT / "visual_annotation_expansion_operator_handoff_005.v1.json"
    handoff_md = REPORTS_ROOT / "visual_annotation_expansion_operator_handoff_005.v1.md"
    template_json = REPORTS_ROOT / "visual_annotation_expansion_web_output_template_005.v1.json"
    template_md = REPORTS_ROOT / "visual_annotation_expansion_web_output_template_005.v1.md"
    web_bundle_json = REPORTS_ROOT / "visual_annotation_expansion_web_run_bundle_005.v1.json"
    web_bundle_md = REPORTS_ROOT / "visual_annotation_expansion_web_run_bundle_005.v1.md"
    web_bundle_dir = REPORTS_ROOT / "visual_annotation_expansion_web_run_bundle_005"
    upload_json = REPORTS_ROOT / "visual_annotation_expansion_web_upload_bundle_corpus_005.v1.json"
    upload_md = REPORTS_ROOT / "visual_annotation_expansion_web_upload_bundle_corpus_005.v1.md"
    upload_dir = REPORTS_ROOT / "visual_annotation_expansion_web_upload_bundle_corpus_005"
    target_output_ref = "eval/knowledgeos/reports/visual_annotation_expansion_web_output_005.manual.json"
    validation_output_ref = (
        "eval/knowledgeos/reports/visual_annotation_expansion_web_output_005.validation.v1.json"
    )
    validation_md_ref = (
        "eval/knowledgeos/reports/visual_annotation_expansion_web_output_005.validation.v1.md"
    )
    validation_command = (
        "PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py "
        f"--output {target_output_ref} "
        f"--source-expansion-pack {sanitized_report_ref(pack_json, project_root=PROJECT_ROOT)} "
        f"--source-attachment-pack {sanitized_report_ref(attachment_json, project_root=PROJECT_ROOT)} "
        f"--validation-json {validation_output_ref} "
        f"--validation-md {validation_md_ref}"
    )

    _remove_generated_dirs([attachment_dir, web_bundle_dir, upload_dir])

    if candidate_json.is_file() and not args.refresh_candidate_report and args.max_papers is None:
        candidate_report = load_json(candidate_json)
        paper_specs = discover_local_paper_specs(papers_root)
    else:
        paper_specs = discover_local_paper_specs(papers_root, limit=args.max_papers)
        candidate_report = build_visual_layout_candidate_list_report(
            papers_root=papers_root,
            paper_specs=paper_specs,
        )
    _validate_or_raise(candidate_report, VISUAL_LAYOUT_CANDIDATE_LIST_REPORT_SCHEMA_ID, "corpus candidate report")
    candidate_paths = _write_candidate_report(
        candidate_report,
        report_json=candidate_json,
        report_md=candidate_md,
    )

    web_pack = load_json(web_pack_path)
    dry_run_report = _combined_previous_dry_run_report([load_json(path) for path in dry_run_paths])
    previous_ids = _prior_candidate_ids(web_pack, dry_run_report)
    preferred_paper_ids = _preferred_paper_ids(
        candidate_report,
        previous_candidate_ids=previous_ids,
        type_quotas=DEFAULT_TYPE_QUOTAS,
    )
    pack_report = build_visual_annotation_expansion_pack_design(
        candidate_report,
        web_pack,
        dry_run_report,
        pack_id=args.pack_id,
        source_candidate_report_ref=sanitized_report_ref(candidate_json, project_root=PROJECT_ROOT),
        source_web_pack_ref=sanitized_report_ref(web_pack_path, project_root=PROJECT_ROOT),
        source_dry_run_report_ref=_combined_ref(dry_run_paths),
        max_candidates=args.max_candidates,
        type_quotas=DEFAULT_TYPE_QUOTAS,
        preferred_paper_ids=preferred_paper_ids,
        max_per_paper_type_page=1,
    )
    _validate_or_raise(pack_report, VISUAL_ANNOTATION_EXPANSION_PACK_DESIGN_SCHEMA_ID, "corpus pack design")
    pack_paths = write_visual_annotation_expansion_pack_design(
        pack_report,
        report_json=pack_json,
        report_md=pack_md,
    )

    attachment_report = build_visual_annotation_expansion_attachment_pack(
        pack_report,
        papers_root=papers_root,
        output_dir=attachment_dir,
        output_dir_ref=sanitized_report_ref(attachment_dir, project_root=PROJECT_ROOT),
        attachment_pack_id="visual_annotation_expansion_attachment_pack_005",
        source_expansion_pack_ref=sanitized_report_ref(pack_json, project_root=PROJECT_ROOT),
    )
    _validate_or_raise(
        attachment_report,
        VISUAL_ANNOTATION_EXPANSION_ATTACHMENT_PACK_SCHEMA_ID,
        "corpus attachment pack",
    )
    attachment_paths = write_visual_annotation_expansion_attachment_pack(
        attachment_report,
        report_json=attachment_json,
        report_md=attachment_md,
    )

    manual_packet = build_visual_annotation_expansion_manual_run_packet(
        pack_report,
        attachment_report,
        packet_id="visual_annotation_expansion_manual_run_packet_005",
        source_expansion_pack_ref=sanitized_report_ref(pack_json, project_root=PROJECT_ROOT),
        source_attachment_pack_ref=sanitized_report_ref(attachment_json, project_root=PROJECT_ROOT),
        batch_size=args.batch_size,
    )
    _validate_or_raise(
        manual_packet,
        VISUAL_ANNOTATION_EXPANSION_MANUAL_RUN_PACKET_SCHEMA_ID,
        "corpus manual run packet",
    )
    manual_paths = write_visual_annotation_expansion_manual_run_packet(
        manual_packet,
        report_json=manual_json,
        report_md=manual_md,
    )

    handoff_report = build_visual_annotation_expansion_operator_handoff(
        manual_packet,
        handoff_id="visual_annotation_expansion_operator_handoff_005",
        source_manual_run_packet_ref=sanitized_report_ref(manual_json, project_root=PROJECT_ROOT),
        expected_output_ref=target_output_ref,
        validation_command=validation_command,
    )
    _validate_or_raise(
        handoff_report,
        VISUAL_ANNOTATION_EXPANSION_OPERATOR_HANDOFF_SCHEMA_ID,
        "corpus operator handoff",
    )
    handoff_paths = write_visual_annotation_expansion_operator_handoff(
        handoff_report,
        report_json=handoff_json,
        report_md=handoff_md,
    )

    template_report = build_visual_annotation_expansion_web_output_template(
        handoff_report,
        template_id="visual_annotation_expansion_web_output_template_005",
        source_operator_handoff_ref=sanitized_report_ref(handoff_json, project_root=PROJECT_ROOT),
        target_output_ref=target_output_ref,
        validation_command=validation_command,
    )
    _validate_or_raise(
        template_report,
        VISUAL_ANNOTATION_EXPANSION_WEB_OUTPUT_TEMPLATE_SCHEMA_ID,
        "corpus web output template",
    )
    template_paths = write_visual_annotation_expansion_web_output_template(
        template_report,
        report_json=template_json,
        report_md=template_md,
    )

    web_bundle_report = build_visual_annotation_expansion_web_run_bundle(
        template_report,
        bundle_id="visual_annotation_expansion_web_run_bundle_005",
        source_web_output_template_ref=sanitized_report_ref(template_json, project_root=PROJECT_ROOT),
        bundle_dir_ref=sanitized_report_ref(web_bundle_dir, project_root=PROJECT_ROOT),
        target_output_ref=target_output_ref,
        validation_command=validation_command,
    )
    _validate_or_raise(
        web_bundle_report,
        VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BUNDLE_SCHEMA_ID,
        "corpus web run bundle",
    )
    for batch in list(web_bundle_report.get("batchBundles") or []):
        _validate_or_raise(
            build_visual_annotation_expansion_web_run_batch_template(batch),
            VISUAL_ANNOTATION_EXPANSION_WEB_RUN_BATCH_TEMPLATE_SCHEMA_ID,
            "corpus web run batch template",
        )
    web_bundle_paths = write_visual_annotation_expansion_web_run_bundle(
        web_bundle_report,
        report_json=web_bundle_json,
        report_md=web_bundle_md,
        bundle_dir=web_bundle_dir,
    )

    upload_report = build_visual_annotation_expansion_web_upload_bundle(
        web_bundle_report,
        attachment_report,
        project_root=PROJECT_ROOT,
        output_dir=upload_dir,
        output_dir_ref=sanitized_report_ref(upload_dir, project_root=PROJECT_ROOT),
        upload_bundle_id="visual_annotation_expansion_web_upload_bundle_corpus_005",
        source_web_run_bundle_ref=sanitized_report_ref(web_bundle_json, project_root=PROJECT_ROOT),
        source_attachment_pack_ref=sanitized_report_ref(attachment_json, project_root=PROJECT_ROOT),
    )
    _validate_or_raise(
        upload_report,
        VISUAL_ANNOTATION_EXPANSION_WEB_UPLOAD_BUNDLE_SCHEMA_ID,
        "corpus web upload bundle",
    )
    upload_paths = write_visual_annotation_expansion_web_upload_bundle(
        upload_report,
        report_json=upload_json,
        report_md=upload_md,
        output_dir=upload_dir,
    )

    statuses = [
        candidate_report.get("status"),
        pack_report.get("status"),
        attachment_report.get("status"),
        manual_packet.get("status"),
        handoff_report.get("status"),
        template_report.get("status"),
        web_bundle_report.get("status"),
        upload_report.get("status"),
    ]
    selected_rows = list(pack_report.get("packRowsDetail") or [])
    summary = {
        "status": "ready" if all(status == "ready" for status in statuses) else "blocked",
        "packId": args.pack_id,
        "counts": {
            "localPdfRows": len(paper_specs),
            "corpusCandidateRows": candidate_report.get("counts", {}).get("candidateRows"),
            "selectedExpansionRows": pack_report.get("counts", {}).get("selectedExpansionRows"),
            "selectedTypeCounts": _type_counts(selected_rows),
            "batchRows": web_bundle_report.get("counts", {}).get("batchRows"),
            "uploadCopiedAttachmentRows": upload_report.get("counts", {}).get("copiedAttachmentRows"),
            "privatePathLeakRows": max(
                int(candidate_report.get("counts", {}).get("privatePathLeakRows", 0) or 0),
                int(pack_report.get("counts", {}).get("privatePathLeakRows", 0) or 0),
                int(attachment_report.get("counts", {}).get("privatePathLeakRows", 0) or 0),
                int(upload_report.get("counts", {}).get("privatePathLeakRows", 0) or 0),
            ),
        },
        "paths": {
            "candidate": candidate_paths,
            "pack": pack_paths,
            "attachment": attachment_paths,
            "manualRunPacket": manual_paths,
            "operatorHandoff": handoff_paths,
            "webOutputTemplate": template_paths,
            "webRunBundle": web_bundle_paths,
            "webUploadBundle": upload_paths,
        },
    }
    print(json.dumps(summary if not args.json else {**summary, "uploadReport": upload_report}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Contract-only write target definition for parsed-artifact SourceSpan records.

This module defines the SourceSpan store contract used after promotion executor
dry-run planning. It does not implement an apply executor and performs no
filesystem, database, index, parser-routing, or answer-runtime mutation.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-store-contract.v1"
)
PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-record.v1"
)

PARSED_ARTIFACT_SOURCE_SPAN_STORE = "parsed_artifact_source_span_store"

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-store-contract"
    / "01-parsed-artifact-source-span-store-contract"
)

NO_MUTATION_POLICY = {
    "contractOnly": True,
    "executorImplemented": False,
    "sourceSpanCreated": False,
    "strictEvidenceCreated": False,
    "citationGradeEvidenceCreated": False,
    "runtimeEvidenceCreated": False,
    "parserRoutingChanged": False,
    "answerIntegrationChanged": False,
    "databaseMutation": False,
    "vaultScan": False,
    "reindexOrReembed": False,
    "canonicalParsedArtifactsWritten": False,
}

REQUIRED_RECORD_FIELDS = [
    "sourceSpanId",
    "candidateRecordId",
    "runId",
    "paperId",
    "artifactType",
    "sourceCandidateId",
    "sourceContentHash",
    "sourceFile",
    "locator",
    "idempotencyKey",
    "evidenceTier",
    "strictEligible",
    "citationGrade",
    "runtimeEvidence",
    "writePolicy",
]

IDEMPOTENCY_KEY_FIELDS = [
    "plannedWriteTarget",
    "paperId",
    "artifactType",
    "sourceCandidateId",
    "sourceContentHash",
    "sourceFile",
    "locator",
    "idempotencyKey",
    "candidateRecordId",
]

SOURCE_SPAN_STORE_CONTRACT: dict[str, Any] = {
    "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_STORE,
    "contractReference": PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID,
    "sourceSpanRecordSchema": PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
    "storeKind": "local_papers_dir_jsonl_source_span_store",
    "storeRootTemplate": "{papers_dir}/structured_evidence/source_span",
    "recordPathTemplate": "{papers_dir}/structured_evidence/source_span/{paper_id}.jsonl",
    "runManifestPathTemplate": "{papers_dir}/structured_evidence/runs/{run_id}.json",
    "allowedArtifactTypes": ["section", "table", "figure"],
    "requiredRecordFields": REQUIRED_RECORD_FIELDS,
    "idempotencyKeyFields": IDEMPOTENCY_KEY_FIELDS,
    "writeSemantics": "explicit_apply_executor_appends_or_replaces_same_idempotency_key",
    "readbackChecks": [
        "record_schema_validates",
        "idempotency_key_stable",
        "sourceContentHash_preserved",
        "locator_preserved",
        "candidateRecordId_linked",
        "record_remains_non_strict_and_non_runtime",
    ],
    "rollbackStrategy": (
        "delete records written by the explicit run_id only while no downstream "
        "strict or runtime evidence record references them"
    ),
    "executorImplemented": False,
    "runtimeUseAllowed": False,
    "strictEvidenceAllowed": False,
    "parserRoutingAllowed": False,
    "answerIntegrationAllowed": False,
    "databaseMutationAllowed": False,
}

KNOWN_WRITE_TARGET_CONTRACTS: dict[str, str] = {
    PARSED_ARTIFACT_SOURCE_SPAN_STORE: PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_parsed_artifact_source_span_store_contract() -> dict[str, Any]:
    write_target = dict(SOURCE_SPAN_STORE_CONTRACT)
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID,
        "status": "ok",
        "generatedAt": _now_iso(),
        "counts": {
            "writeTargetContracts": 1,
            "sourceSpanStoreContracts": 1,
            "recordSchemas": 1,
            "executorImplementedRows": 0,
            "sourceSpanCreatedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
        },
        "gate": {
            "writeTargetContractsDefined": True,
            "sourceSpanStoreContractDefined": True,
            "executorReady": False,
            "runtimeMutationAllowed": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": "parsed_artifact_source_span_store_contract_ready",
            "recommendedNextTranche": "parsed_artifact_source_span_promotion_executor_apply",
        },
        "policy": dict(NO_MUTATION_POLICY),
        "contractPrinciples": [
            "source_span_store_records_are_not_strict_evidence",
            "source_span_store_records_are_not_runtime_citations",
            "explicit_apply_executor_required_before_any_write",
            "readback_validation_required_before_strict_evidence_promotion",
            "strict_evidence_requires_later_explicit_gate",
            "parser_routing_and_answer_integration_require_later_explicit_tranches",
            "promotion_executor_dry_run_must_reference_this_contract_before_apply",
        ],
        "writeTargets": [write_target],
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "counts",
            "gate",
            "policy",
            "contractPrinciples",
            "writeTargets",
        )
        if key in report
    }


def render_parsed_artifact_source_span_store_contract_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact SourceSpan Store Contract",
        "",
        f"- status: {report.get('status', '')}",
        f"- contract-only: {json.dumps(report.get('policy', {}).get('contractOnly'))}",
        f"- executor implemented: {json.dumps(report.get('policy', {}).get('executorImplemented'))}",
        f"- write target contracts: {int(counts.get('writeTargetContracts') or 0)}",
        f"- source spans created: {int(counts.get('sourceSpanCreatedRows') or 0)}",
        f"- database mutations: {int(counts.get('databaseMutationRows') or 0)}",
        "",
        "## Write Targets",
    ]
    for target in list(report.get("writeTargets") or []):
        lines.extend(
            [
                f"- {target.get('plannedWriteTarget', '')}",
                f"  - record schema: {target.get('sourceSpanRecordSchema', '')}",
                f"  - store root: {target.get('storeRootTemplate', '')}",
                f"  - record path: {target.get('recordPathTemplate', '')}",
                f"  - allowed artifacts: {', '.join(list(target.get('allowedArtifactTypes') or []))}",
                f"  - executor implemented: {json.dumps(target.get('executorImplemented'))}",
            ]
        )
    return "\n".join(lines)


def write_parsed_artifact_source_span_store_contract_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-store-contract.json"
    summary_path = root / "parsed-artifact-source-span-store-contract-summary.json"
    markdown_path = root / "parsed-artifact-source-span-store-contract.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_store_contract_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description="Emit the contract-only parsed-artifact SourceSpan store payload."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_parsed_artifact_source_span_store_contract()
    if args.output_dir:
        paths = write_parsed_artifact_source_span_store_contract_reports(
            payload,
            args.output_dir,
        )
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(payload), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "KNOWN_WRITE_TARGET_CONTRACTS",
    "NO_MUTATION_POLICY",
    "PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID",
    "PARSED_ARTIFACT_SOURCE_SPAN_STORE",
    "PARSED_ARTIFACT_SOURCE_SPAN_STORE_CONTRACT_SCHEMA_ID",
    "SOURCE_SPAN_STORE_CONTRACT",
    "build_parsed_artifact_source_span_store_contract",
    "render_parsed_artifact_source_span_store_contract_markdown",
    "write_parsed_artifact_source_span_store_contract_reports",
]

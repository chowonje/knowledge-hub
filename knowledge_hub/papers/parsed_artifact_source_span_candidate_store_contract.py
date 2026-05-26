"""Contract-only write target definitions for parsed-artifact source-span candidates.

This module defines the candidate store contracts used by parsed-artifact
structured-evidence planning. It does not implement an executor and performs no
filesystem, database, index, parser-routing, or answer-runtime mutation.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-candidate-store-contract.v1"
)
PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-candidate-record.v1"
)
STRUCTURED_EVIDENCE_CANDIDATE_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.structured-evidence-candidate-record.v1"
)

PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE = "parsed_artifact_source_span_candidate_store"
STRUCTURED_EVIDENCE_CANDIDATE_STORE = "structured_evidence_candidate_store"

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-candidate-store-contract"
    / "01-parsed-artifact-source-span-candidate-store-contract"
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

COMMON_REQUIRED_PROVENANCE_FIELDS = [
    "paper_id",
    "artifact_type",
    "source_candidate_id",
    "sourceContentHash",
    "source_file",
    "page",
    "bbox",
    "blockIndexes",
    "planned_operation",
    "planned_write_target",
]

COMMON_IDEMPOTENCY_FIELDS = [
    "planned_write_target",
    "paper_id",
    "artifact_type",
    "source_candidate_id",
    "sourceContentHash",
    "source_file",
    "page",
    "bbox",
    "blockIndexes",
]

WRITE_TARGET_CONTRACTS: dict[str, dict[str, Any]] = {
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE: {
        "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
        "contractReference": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID,
        "candidateRecordSchema": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
        "storeKind": "local_papers_dir_jsonl_candidate_store",
        "storeRootTemplate": "{papers_dir}/structured_evidence_candidates/source_span",
        "recordPathTemplate": (
            "{papers_dir}/structured_evidence_candidates/source_span/{paper_id}.jsonl"
        ),
        "runManifestPathTemplate": (
            "{papers_dir}/structured_evidence_candidates/runs/{run_id}.json"
        ),
        "allowedArtifactTypes": ["section", "table", "figure"],
        "requiredProvenanceFields": COMMON_REQUIRED_PROVENANCE_FIELDS,
        "idempotencyKeyFields": COMMON_IDEMPOTENCY_FIELDS,
        "writeSemantics": "explicit_apply_executor_appends_or_replaces_same_idempotency_key",
        "readbackChecks": [
            "record_schema_validates",
            "idempotency_key_matches_input_row",
            "sourceContentHash_matches_input_row",
            "location_fields_match_input_row",
            "record_remains_candidate_only",
        ],
        "rollbackStrategy": (
            "delete records written by the explicit run_id while no downstream promotion "
            "record references them"
        ),
        "executorImplemented": False,
        "runtimeUseAllowed": False,
        "strictEvidenceAllowed": False,
        "parserRoutingAllowed": False,
        "answerIntegrationAllowed": False,
        "databaseMutationAllowed": False,
    },
    STRUCTURED_EVIDENCE_CANDIDATE_STORE: {
        "plannedWriteTarget": STRUCTURED_EVIDENCE_CANDIDATE_STORE,
        "contractReference": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID,
        "candidateRecordSchema": STRUCTURED_EVIDENCE_CANDIDATE_RECORD_SCHEMA_ID,
        "storeKind": "local_papers_dir_jsonl_candidate_store",
        "storeRootTemplate": "{papers_dir}/structured_evidence_candidates/structured",
        "recordPathTemplate": (
            "{papers_dir}/structured_evidence_candidates/structured/{paper_id}.jsonl"
        ),
        "runManifestPathTemplate": (
            "{papers_dir}/structured_evidence_candidates/runs/{run_id}.json"
        ),
        "allowedArtifactTypes": ["equation"],
        "requiredProvenanceFields": [
            *COMMON_REQUIRED_PROVENANCE_FIELDS,
            "source_span_candidate_record_id",
        ],
        "idempotencyKeyFields": [
            *COMMON_IDEMPOTENCY_FIELDS,
            "source_span_candidate_record_id",
        ],
        "writeSemantics": "explicit_apply_executor_appends_or_replaces_same_idempotency_key",
        "readbackChecks": [
            "record_schema_validates",
            "idempotency_key_matches_input_row",
            "source_span_candidate_record_id_resolves_when_present",
            "sourceContentHash_matches_input_row",
            "record_remains_candidate_only",
        ],
        "rollbackStrategy": (
            "delete records written by the explicit run_id while no strict evidence "
            "or runtime evidence record references them"
        ),
        "executorImplemented": False,
        "runtimeUseAllowed": False,
        "strictEvidenceAllowed": False,
        "parserRoutingAllowed": False,
        "answerIntegrationAllowed": False,
        "databaseMutationAllowed": False,
    },
}

KNOWN_WRITE_TARGET_CONTRACTS: dict[str, str] = {
    target: str(contract["contractReference"])
    for target, contract in WRITE_TARGET_CONTRACTS.items()
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_parsed_artifact_source_span_candidate_store_contract() -> dict[str, Any]:
    write_targets = [
        dict(contract)
        for _, contract in sorted(WRITE_TARGET_CONTRACTS.items())
    ]
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID,
        "status": "ok",
        "generatedAt": _now_iso(),
        "counts": {
            "writeTargetContracts": len(write_targets),
            "sourceSpanCandidateStoreContracts": sum(
                1
                for item in write_targets
                if item["plannedWriteTarget"] == PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE
            ),
            "structuredEvidenceCandidateStoreContracts": sum(
                1
                for item in write_targets
                if item["plannedWriteTarget"] == STRUCTURED_EVIDENCE_CANDIDATE_STORE
            ),
            "recordSchemas": len(
                {str(item.get("candidateRecordSchema") or "") for item in write_targets}
            ),
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
            "sourceSpanCandidateStoreContractDefined": True,
            "structuredEvidenceCandidateStoreContractDefined": True,
            "executorReady": False,
            "runtimeMutationAllowed": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": "parsed_artifact_source_span_candidate_store_contract_ready",
            "recommendedNextTranche": (
                "parsed_artifact_structured_evidence_write_target_contract_audit_rerun"
            ),
        },
        "policy": dict(NO_MUTATION_POLICY),
        "contractPrinciples": [
            "candidate_store_records_are_not_strict_evidence",
            "candidate_store_records_are_not_runtime_citations",
            "explicit_apply_executor_required_before_any_write",
            "readback_validation_required_before_promotion",
            "strict_evidence_requires_later_source_span_promotion_gate",
            "parser_routing_and_answer_integration_require_later_explicit_tranches",
        ],
        "writeTargets": write_targets,
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


def render_parsed_artifact_source_span_candidate_store_contract_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact SourceSpan Candidate Store Contract",
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
                f"  - record schema: {target.get('candidateRecordSchema', '')}",
                f"  - store root: {target.get('storeRootTemplate', '')}",
                f"  - allowed artifacts: {', '.join(list(target.get('allowedArtifactTypes') or []))}",
                f"  - executor implemented: {json.dumps(target.get('executorImplemented'))}",
            ]
        )
    return "\n".join(lines)


def write_parsed_artifact_source_span_candidate_store_contract_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-candidate-store-contract.json"
    summary_path = root / "parsed-artifact-source-span-candidate-store-contract-summary.json"
    markdown_path = root / "parsed-artifact-source-span-candidate-store-contract.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_candidate_store_contract_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description="Emit the contract-only parsed-artifact source-span candidate store payload."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_parsed_artifact_source_span_candidate_store_contract()
    if args.output_dir:
        paths = write_parsed_artifact_source_span_candidate_store_contract_reports(
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
    "PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID",
    "PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE",
    "PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE_CONTRACT_SCHEMA_ID",
    "STRUCTURED_EVIDENCE_CANDIDATE_RECORD_SCHEMA_ID",
    "STRUCTURED_EVIDENCE_CANDIDATE_STORE",
    "WRITE_TARGET_CONTRACTS",
    "build_parsed_artifact_source_span_candidate_store_contract",
    "render_parsed_artifact_source_span_candidate_store_contract_markdown",
    "write_parsed_artifact_source_span_candidate_store_contract_reports",
]

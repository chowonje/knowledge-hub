"""Report-only command packet for candidate-layer blocker manual decisions.

This helper emits copy/paste commands for a human/operator decision step. It
does not edit decision files, record decisions, create evidence, change parser
routing, write canonical parsed artifacts, mutate SQLite, or change answer
behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
from typing import Any


CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_COMMAND_PACKET_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-manual-decision-command-packet.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-input-pack.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-template.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_FILE_DRAFT_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-file-draft.v1"
)
CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID = "knowledge-hub.paper.candidate-layer-blocker-backlog.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _schema_violations(
    *,
    input_pack: dict[str, Any],
    template: dict[str, Any],
    draft: dict[str, Any],
    backlog: dict[str, Any],
    decision_file_draft_path: str,
) -> list[str]:
    violations: list[str] = []
    if input_pack.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_INPUT_PACK_SCHEMA_ID:
        violations.append("candidate_layer_blocker_decision_input_pack_schema_mismatch")
    if input_pack and input_pack.get("status") != "decision_input_pack_ready":
        violations.append("candidate_layer_blocker_decision_input_pack_not_ready")
    if template.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_TEMPLATE_SCHEMA_ID:
        violations.append("candidate_layer_blocker_decision_template_schema_mismatch")
    if template and template.get("status") != "decision_template_ready":
        violations.append("candidate_layer_blocker_decision_template_not_ready")
    if draft.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_FILE_DRAFT_SCHEMA_ID:
        violations.append("candidate_layer_blocker_decision_file_draft_schema_mismatch")
    if draft and draft.get("status") != "decision_file_draft_ready":
        violations.append("candidate_layer_blocker_decision_file_draft_not_ready")
    if not decision_file_draft_path:
        violations.append("candidate_layer_blocker_decision_file_draft_path_missing")
    if backlog.get("schema") != CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID:
        violations.append("candidate_layer_blocker_backlog_schema_mismatch")
    if backlog and backlog.get("status") != "ok":
        violations.append("candidate_layer_blocker_backlog_not_ok")
    return list(dict.fromkeys(violations))


def _command(
    *,
    step: int,
    name: str,
    purpose: str,
    command: str,
    reads: list[str] | None = None,
    writes: list[str] | None = None,
    requires: list[str] | None = None,
    does_not_do: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "step": step,
        "name": name,
        "purpose": purpose,
        "command": command,
        "reads": reads or [],
        "writes": writes or [],
        "requires": requires or [],
        "doesNotDo": does_not_do
        or [
            "no DB mutation",
            "no parser routing",
            "no strict evidence promotion",
            "no runtime evidence creation",
            "no canonical parsed artifact write",
        ],
    }


def _build_commands(
    *,
    worktree: Path,
    output_dir: Path,
    input_pack_report: Path,
    template_report: Path,
    draft_report: Path,
    draft_decision_file: Path,
    backlog_report: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    review_file = output_dir / "candidate-layer-blocker-decisions.review.json"
    validation_dir = output_dir.parent / "candidate-layer-blocker-decision-file-validation-after-manual-edit"
    record_dir = output_dir.parent / "candidate-layer-blocker-decision-record-after-manual-edit"
    preview_dir = output_dir.parent / "candidate-layer-blocker-resolution-preview-after-manual-edit"
    record_report = record_dir / "candidate-layer-blocker-decision-record.json"

    commands = [
        _command(
            step=1,
            name="create_review_copy",
            purpose="Copy the generated needs-review draft to an editable review file.",
            command=f"cp {_q(draft_decision_file)} {_q(review_file)}",
            reads=[str(draft_decision_file)],
            writes=[str(review_file)],
            requires=["edit the review copy, not the generated draft"],
        ),
        _command(
            step=2,
            name="manual_edit_review_copy",
            purpose="Human/operator edits decision, reviewer, and notes fields in the review copy.",
            command=f"${{EDITOR:-vi}} {_q(review_file)}",
            reads=[str(review_file)],
            writes=[str(review_file)],
            requires=[
                "do not change allowed_decisions",
                "do not add strict/runtime evidence fields",
                "leave rows as needs_review when no actual review happened",
            ],
        ),
        _command(
            step=3,
            name="validate_decision_file",
            purpose="Validate the edited decision file before any decision record is generated.",
            command=(
                f"cd {_q(worktree)} && "
                "python -m knowledge_hub.papers.candidate_layer_blocker_decision_file_validation "
                f"--candidate-layer-blocker-decision-input-pack-report {_q(input_pack_report)} "
                f"--candidate-layer-blocker-decisions-file {_q(review_file)} "
                f"--output-dir {_q(validation_dir)} --json"
            ),
            reads=[str(input_pack_report), str(review_file)],
            writes=[str(validation_dir)],
            requires=["invalidRows must equal 0", "missingRows must equal 0"],
        ),
        _command(
            step=4,
            name="record_decisions_report_only",
            purpose="Convert the validated decision file into a report-only decision record.",
            command=(
                f"cd {_q(worktree)} && "
                "python -m knowledge_hub.papers.candidate_layer_blocker_decision_record "
                f"--candidate-layer-blocker-decision-template-report {_q(template_report)} "
                f"--blocker-decisions-report {_q(review_file)} "
                f"--output-dir {_q(record_dir)} --json"
            ),
            reads=[str(template_report), str(review_file), str(validation_dir)],
            writes=[str(record_dir)],
            requires=[
                "decision record remains report-only",
                "strictEligibleRows must remain 0",
                "runtimeEvidenceRows must remain 0",
            ],
        ),
        _command(
            step=5,
            name="preview_resolution_report_only",
            purpose="Preview decision-record impact against the current blocker backlog without mutating backlog state.",
            command=(
                f"cd {_q(worktree)} && "
                "python -m knowledge_hub.papers.candidate_layer_blocker_resolution_preview "
                f"--candidate-layer-blocker-decision-record-report {_q(record_report)} "
                f"--candidate-layer-blocker-backlog-report {_q(backlog_report)} "
                f"--output-dir {_q(preview_dir)} --json"
            ),
            reads=[str(record_report), str(backlog_report)],
            writes=[str(preview_dir)],
            requires=["preview is report-only", "backlog state is not mutated"],
        ),
    ]
    derived_paths = {
        "reviewFile": str(review_file),
        "validationOutputDir": str(validation_dir),
        "recordOutputDir": str(record_dir),
        "recordReport": str(record_report),
        "previewOutputDir": str(preview_dir),
    }
    return commands, derived_paths


def build_candidate_layer_blocker_manual_decision_command_packet(
    *,
    candidate_layer_blocker_decision_input_pack_report: str | Path,
    candidate_layer_blocker_decision_template_report: str | Path,
    candidate_layer_blocker_decision_file_draft_report: str | Path,
    candidate_layer_blocker_backlog_report: str | Path,
    worktree: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a report-only command packet for manual blocker decisions."""

    input_pack_path = Path(str(candidate_layer_blocker_decision_input_pack_report)).expanduser()
    template_path = Path(str(candidate_layer_blocker_decision_template_report)).expanduser()
    draft_path = Path(str(candidate_layer_blocker_decision_file_draft_report)).expanduser()
    backlog_path = Path(str(candidate_layer_blocker_backlog_report)).expanduser()
    output_root = Path(str(output_dir)).expanduser()
    worktree_path = Path(str(worktree)).expanduser()

    input_pack = _read_json(input_pack_path)
    template = _read_json(template_path)
    draft = _read_json(draft_path)
    backlog = _read_json(backlog_path)
    draft_decision_file = str((draft.get("reportPaths") or {}).get("decisionFileDraft") or "")
    if not draft_decision_file:
        sibling_draft = draft_path.parent / "candidate-layer-blocker-decisions.draft.json"
        if sibling_draft.exists():
            draft_decision_file = str(sibling_draft)

    violations = _schema_violations(
        input_pack=input_pack,
        template=template,
        draft=draft,
        backlog=backlog,
        decision_file_draft_path=draft_decision_file,
    )
    commands: list[dict[str, Any]] = []
    derived_paths: dict[str, str] = {}
    if not violations:
        commands, derived_paths = _build_commands(
            worktree=worktree_path,
            output_dir=output_root,
            input_pack_report=input_pack_path,
            template_report=template_path,
            draft_report=draft_path,
            draft_decision_file=Path(draft_decision_file),
            backlog_report=backlog_path,
        )

    input_counts = dict(input_pack.get("counts") or {})
    counts = {
        "commandCount": len(commands),
        "inputRows": _safe_int(input_counts.get("inputRows")),
        "manualDecisionInputRows": _safe_int(input_counts.get("manualDecisionInputRows")),
        "technicalDecisionInputRows": _safe_int(input_counts.get("technicalDecisionInputRows")),
        "policyDecisionInputRows": _safe_int(input_counts.get("policyDecisionInputRows")),
        "repoFilesChanged": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(violations),
    }
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_COMMAND_PACKET_SCHEMA_ID,
        "status": "command_packet_ready" if commands and not violations else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerBlockerDecisionInputPackReport": str(input_pack_path),
            "candidateLayerBlockerDecisionInputPackSchema": str(input_pack.get("schema") or ""),
            "candidateLayerBlockerDecisionTemplateReport": str(template_path),
            "candidateLayerBlockerDecisionTemplateSchema": str(template.get("schema") or ""),
            "candidateLayerBlockerDecisionFileDraftReport": str(draft_path),
            "candidateLayerBlockerDecisionFileDraftSchema": str(draft.get("schema") or ""),
            "candidateLayerBlockerBacklogReport": str(backlog_path),
            "candidateLayerBlockerBacklogSchema": str(backlog.get("schema") or ""),
            "worktree": str(worktree_path),
            "outputDir": str(output_root),
        },
        "derivedPaths": derived_paths,
        "counts": counts,
        "gate": {
            "commandPacketReady": bool(commands and not violations),
            "manualEditRequired": counts["manualDecisionInputRows"] > 0,
            "commandsExecuted": False,
            "decisionsRecorded": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "manual_commands_ready" if commands and not violations else "blocked",
            "schemaViolations": violations,
            "recommendedNextTranche": "execute_manual_decision_commands_outside_codex_auto_apply",
        },
        "policy": {
            "commandPacketOnly": True,
            "commandsExecuted": False,
            "decisionFileModified": False,
            "decisionRecordCreated": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "command_packet_does_not_execute_commands",
            "command_packet_does_not_record_decisions",
            "manual_review_copy_must_be_edited_by_a_human_or_operator",
            "strict_or_runtime_promotion_requires_a_separate_explicit_tranche",
        ],
        "commands": commands,
    }


def render_candidate_layer_blocker_manual_decision_command_packet_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Blocker Manual Decision Command Packet",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Commands: `{int(counts.get('commandCount') or 0)}`",
        f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
        f"- Manual decision rows: `{int(counts.get('manualDecisionInputRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        f"- Runtime evidence rows: `{int(counts.get('runtimeEvidenceRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This packet is report-only. It does not execute commands, modify decision files, record decisions, create strict evidence, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Commands",
        "",
    ]
    for command in list(report.get("commands") or []):
        lines.extend(
            [
                f"### {int(command.get('step') or 0)}. `{command.get('name', '')}`",
                "",
                str(command.get("purpose") or ""),
                "",
                "```bash",
                str(command.get("command") or ""),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def write_candidate_layer_blocker_manual_decision_command_packet_reports(
    report: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    packet_path = root / "candidate-layer-blocker-manual-decision-command-packet.json"
    summary_path = root / "candidate-layer-blocker-manual-decision-command-summary.json"
    markdown_path = root / "candidate-layer-blocker-manual-decision-command-packet.md"
    packet_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_payload = {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "derivedPaths", "counts", "gate", "policy", "warnings")
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_candidate_layer_blocker_manual_decision_command_packet_markdown(report),
        encoding="utf-8",
    )
    return {
        "packet": str(packet_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer blocker manual decision command packet.")
    parser.add_argument("--candidate-layer-blocker-decision-input-pack-report", required=True)
    parser.add_argument("--candidate-layer-blocker-decision-template-report", required=True)
    parser.add_argument("--candidate-layer-blocker-decision-file-draft-report", required=True)
    parser.add_argument("--candidate-layer-blocker-backlog-report", required=True)
    parser.add_argument("--worktree", default=str(Path.cwd()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_manual_decision_command_packet(
        candidate_layer_blocker_decision_input_pack_report=args.candidate_layer_blocker_decision_input_pack_report,
        candidate_layer_blocker_decision_template_report=args.candidate_layer_blocker_decision_template_report,
        candidate_layer_blocker_decision_file_draft_report=args.candidate_layer_blocker_decision_file_draft_report,
        candidate_layer_blocker_backlog_report=args.candidate_layer_blocker_backlog_report,
        worktree=args.worktree,
        output_dir=args.output_dir,
    )
    paths = write_candidate_layer_blocker_manual_decision_command_packet_reports(report, args.output_dir)
    report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_MANUAL_DECISION_COMMAND_PACKET_SCHEMA_ID",
    "build_candidate_layer_blocker_manual_decision_command_packet",
    "render_candidate_layer_blocker_manual_decision_command_packet_markdown",
    "write_candidate_layer_blocker_manual_decision_command_packet_reports",
]

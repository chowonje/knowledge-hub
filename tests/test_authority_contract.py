from __future__ import annotations

import json
import knowledge_hub.application.agent.foundry_bridge as foundry_bridge
from pathlib import Path
import re
import subprocess
import time

from click.testing import CliRunner
import pytest
from jsonschema import Draft202012Validator

from knowledge_hub.application.agent.foundry_bridge import (
    FOUNDRY_PROJECT_DIST_SCRIPT,
    FOUNDRY_PROJECT_SCRIPT,
    PROJECT_ROOT,
    _bridge_candidates,
    coerce_json_output,
)
from knowledge_hub.application.mcp.agent_payloads import _normalize_playbook
from knowledge_hub.application.mcp.responses import evaluate_policy_gate
from knowledge_hub.core.config import Config
from knowledge_hub.core.sanitizer import P0_PATTERN_CONFIG_PATH, get_p0_detection_config
from knowledge_hub.core.schema_validator import SCHEMA_NAME_BY_ID, validate_payload
from knowledge_hub.interfaces.cli.commands.agent_cmd import _normalize_run_payload

FIXTURE_ROOT = PROJECT_ROOT / "docs" / "schemas" / "fixtures"
POLICY_CORPUS_PATH = PROJECT_ROOT / "docs" / "policy" / "p0-sample-corpus.json"
POLICY_CONFORMANCE_PATH = PROJECT_ROOT / "docs" / "policy" / "policy-conformance-cases.json"
AGENT_RUN_SCHEMA_PATH = PROJECT_ROOT / "docs" / "schemas" / "agent-run-result.v1.json"
AUTHORITY_ENVELOPE_SCHEMA_PATH = PROJECT_ROOT / "docs" / "schemas" / "authority-result-envelope.v1.json"
TS_RUNTIME_PATH = PROJECT_ROOT / "foundry-core" / "src" / "runtime.ts"


class _AuthorityStubSQLite:
    def list_ops_actions(self, status=None, scope=None, limit=100):  # noqa: ANN001
        return []


class _AuthorityStubKhub:
    def __init__(self, config: Config):
        self.config = config
        self._sqlite = _AuthorityStubSQLite()

    def sqlite_db(self):
        return self._sqlite


def _make_capture_runtime_config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("validation", "schema", "strict", True)
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("obsidian", "vault_path", str(tmp_path / "vault"))
    config.set_nested("obsidian", "write_backend", "filesystem")
    config.set_nested("obsidian", "cli_binary", "obsidian")
    return config


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_playbook_subschema() -> dict[str, object]:
    schema = _load_json(AGENT_RUN_SCHEMA_PATH)
    return schema["properties"]["playbook"]  # type: ignore[index]


def _load_authority_envelope_schema() -> dict[str, object]:
    return _load_json(AUTHORITY_ENVELOPE_SCHEMA_PATH)


def _schema_errors(schema_path: Path, payload: dict[str, object]) -> list[str]:
    validator = Draft202012Validator(_load_json(schema_path))
    return [error.message for error in validator.iter_errors(payload)]


def _pointer_from_traceability(traceability: dict[str, object]) -> str | None:
    packet_path = traceability.get("packetPath")
    if isinstance(packet_path, str) and packet_path:
        return packet_path
    runtime_pointer = traceability.get("runtimePointer")
    if isinstance(runtime_pointer, str) and runtime_pointer:
        return runtime_pointer
    return None


def _extract_ts_runtime_schema_ids() -> set[str]:
    source = TS_RUNTIME_PATH.read_text(encoding="utf-8")
    marker = "const SCHEMA_NAME_BY_ID"
    start = source.index(marker)
    brace_start = source.index("{", start)

    depth = 0
    brace_end = brace_start
    for index in range(brace_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                brace_end = index
                break

    block = source[brace_start + 1:brace_end]
    return set(re.findall(r'"([^"]+)"\s*:\s*"[^"]+"', block))


def _project_cli_candidate_label(candidate: list[str]) -> str:
    binary_name = Path(candidate[0]).name if candidate else ""
    if candidate[:2] == ["npx", "tsx"]:
        return "npx-tsx-source"
    if binary_name == "tsx":
        return "tsx-source"
    if binary_name == "ts-node":
        return "ts-node-source"
    if len(candidate) >= 2 and candidate[0] == "node" and candidate[1].endswith(".ts"):
        return "node-ts-source"
    if len(candidate) >= 2 and candidate[0] == "node" and candidate[1].endswith(".js"):
        return "node-dist"
    return candidate[0] if candidate else "unknown"


def _project_cli_output_excerpt(raw: str | bytes | None, limit: int = 240) -> str:
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw or "")
    collapsed = " | ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def _project_cli_error_summary(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    combined = "\n".join(
        part for part in [_project_cli_output_excerpt(stderr), _project_cli_output_excerpt(stdout)] if part
    )
    if not combined:
        return ""
    for pattern in ("Cannot find module", "ModuleNotFoundError", "ERR_UNKNOWN_FILE_EXTENSION", "Error", "ERR_"):
        match = re.search(rf"([^\n]*{re.escape(pattern)}[^\n]*)", combined)
        if match:
            return match.group(1)
    return combined


def _format_project_cli_failures(command_args: list[str], attempts: list[dict[str, object]]) -> str:
    lines = [
        "project-cli returned no valid JSON payload",
        f"command_args={command_args!r}",
    ]
    for attempt in attempts:
        lines.append(
            " | ".join(
                [
                    f"candidate={attempt['candidate_label']}",
                    f"argv={attempt['argv']!r}",
                    f"failure={attempt['failure']}",
                    f"returncode={attempt['returncode']}",
                    f"elapsed_sec={attempt['elapsed_sec']}",
                    f"stdout_schema={attempt['stdout_schema']!r}",
                    f"error={attempt['error_summary']!r}",
                    f"stderr={attempt['stderr_excerpt']!r}",
                    f"stdout={attempt['stdout_excerpt']!r}",
                ]
            )
        )
    return "\n".join(lines)


def _run_ts_project_cli(project_root: Path, command_args: list[str]) -> dict[str, object]:
    script_args = [str(project_root), "python", *command_args]
    candidates = _bridge_candidates(
        script_args,
        script_path=FOUNDRY_PROJECT_SCRIPT,
        dist_path=FOUNDRY_PROJECT_DIST_SCRIPT,
    )
    if not candidates:
        pytest.skip("foundry-core project bridge candidates are unavailable")

    attempts: list[dict[str, object]] = []
    for candidate in candidates:
        started = time.perf_counter()
        candidate_label = _project_cli_candidate_label(candidate)
        try:
            result = subprocess.run(
                candidate,
                check=False,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=90,
            )
        except FileNotFoundError:
            attempts.append(
                {
                    "candidate_label": candidate_label,
                    "argv": candidate,
                    "failure": "missing-binary",
                    "returncode": "missing-binary",
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "stdout_schema": None,
                    "error_summary": "binary not found",
                    "stderr_excerpt": "",
                    "stdout_excerpt": "",
                }
            )
            continue
        except subprocess.TimeoutExpired as error:
            attempts.append(
                {
                    "candidate_label": candidate_label,
                    "argv": candidate,
                    "failure": "timeout",
                    "returncode": "timeout",
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "stdout_schema": None,
                    "error_summary": _project_cli_error_summary(error.stdout, error.stderr) or "timed out",
                    "stderr_excerpt": _project_cli_output_excerpt(error.stderr),
                    "stdout_excerpt": _project_cli_output_excerpt(error.stdout),
                }
            )
            continue

        payload = coerce_json_output(result.stdout or "")
        if result.returncode == 0 and payload is not None:
            return payload

        attempts.append(
            {
                "candidate_label": candidate_label,
                "argv": candidate,
                "failure": "nonzero-exit" if result.returncode != 0 else "no-json-payload",
                "returncode": result.returncode,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "stdout_schema": payload.get("schema") if isinstance(payload, dict) else None,
                "error_summary": _project_cli_error_summary(result.stdout, result.stderr)
                or f"command failed with code {result.returncode}",
                "stderr_excerpt": _project_cli_output_excerpt(result.stderr),
                "stdout_excerpt": _project_cli_output_excerpt(result.stdout),
            }
        )

    pytest.fail(_format_project_cli_failures(command_args, attempts))


def test_ts_runtime_schema_registry_is_subset_of_python_registry():
    ts_schema_ids = _extract_ts_runtime_schema_ids()
    missing = sorted(schema_id for schema_id in ts_schema_ids if schema_id not in SCHEMA_NAME_BY_ID)
    assert missing == []


def test_bridge_candidates_prefer_repo_local_tsx_before_npx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script_path = tmp_path / "project-cli.ts"
    dist_path = tmp_path / "project-cli.js"
    local_tsx = tmp_path / "tsx"
    script_path.write_text("console.log('ts')", encoding="utf-8")
    dist_path.write_text("console.log('js')", encoding="utf-8")
    local_tsx.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    local_tsx.chmod(0o755)

    monkeypatch.setattr(foundry_bridge, "FOUNDRY_LOCAL_TSX_BIN", local_tsx)
    monkeypatch.setattr(
        foundry_bridge.shutil,
        "which",
        lambda binary: {"npx": "/usr/bin/npx"}.get(binary),
    )

    candidates = foundry_bridge._bridge_candidates(
        ["repo-root", "python", "project", "status"],
        script_path=script_path,
        dist_path=dist_path,
    )
    labels = [_project_cli_candidate_label(candidate) for candidate in candidates]

    assert "tsx-source" in labels
    assert "npx-tsx-source" in labels
    assert labels.index("tsx-source") < labels.index("npx-tsx-source")


@pytest.mark.parametrize(
    ("fixture_name", "schema_id"),
    [
        ("agent-run-result.v1.fixture.json", "knowledge-hub.foundry.agent.run.result.v1"),
        ("connector-sync-result.v2.fixture.json", "knowledge-hub.foundry.connector.sync.result.v2"),
        ("os-project-create-result.v1.fixture.json", "knowledge-hub.os.project.create.result.v1"),
        ("os-project-update-result.v1.fixture.json", "knowledge-hub.os.project.update.result.v1"),
        ("os-project-show-result.v1.fixture.json", "knowledge-hub.os.project.show.result.v1"),
        ("os-project-evidence-result.v1.fixture.json", "knowledge-hub.os.project.evidence.result.v1"),
        ("os-evidence-review-result.v1.fixture.json", "knowledge-hub.os.evidence.review.result.v1"),
        ("dinger-ingest-result.v1.fixture.json", "knowledge-hub.dinger.ingest.result.v1"),
        ("dinger-ask-result.v1.fixture.json", "knowledge-hub.dinger.ask.result.v1"),
        ("dinger-capture-result.v1.fixture.json", "knowledge-hub.dinger.capture.result.v1"),
        ("dinger-capture-cleanup-result.v1.fixture.json", "knowledge-hub.dinger.capture-cleanup.result.v1"),
        ("dinger-file-result.v1.fixture.json", "knowledge-hub.dinger.file.result.v1"),
        ("dinger-recent-result.v1.fixture.json", "knowledge-hub.dinger.recent.result.v1"),
        ("dinger-lint-result.v1.fixture.json", "knowledge-hub.dinger.lint.result.v1"),
        ("os-project-export-obsidian-result.v1.fixture.json", "knowledge-hub.os.project.export.obsidian.result.v1"),
        ("os-capture-result.v1.fixture.json", "knowledge-hub.os.capture.result.v1"),
        ("os-goal-update-result.v1.fixture.json", "knowledge-hub.os.goal.update.result.v1"),
        ("os-task-update-result.v1.fixture.json", "knowledge-hub.os.task.update.result.v1"),
        ("os-task-start-result.v1.fixture.json", "knowledge-hub.os.task.start.result.v1"),
        ("os-task-block-result.v1.fixture.json", "knowledge-hub.os.task.block.result.v1"),
        ("os-task-complete-result.v1.fixture.json", "knowledge-hub.os.task.complete.result.v1"),
        ("os-task-cancel-result.v1.fixture.json", "knowledge-hub.os.task.cancel.result.v1"),
        ("os-inbox-triage-result.v1.fixture.json", "knowledge-hub.os.inbox.triage.result.v1"),
        ("os-decide-result.v1.fixture.json", "knowledge-hub.os.decide.result.v1"),
        ("os-decision-add-result.v1.fixture.json", "knowledge-hub.os.decision.add.result.v1"),
        ("os-decision-list-result.v1.fixture.json", "knowledge-hub.os.decision.list.result.v1"),
        ("os-next-result.v1.fixture.json", "knowledge-hub.os.next.result.v1"),
    ],
)
def test_bridge_fixtures_validate_against_python_authority_schemas(fixture_name: str, schema_id: str):
    fixture = _load_json(FIXTURE_ROOT / fixture_name)
    result = validate_payload(fixture, schema_id, strict=True)
    assert result.ok, result.errors


def test_agent_run_playbook_fixture_validates_against_authority_subschema():
    fixture = _load_json(FIXTURE_ROOT / "agent-run-playbook.v1.fixture.json")
    validator = Draft202012Validator(_load_playbook_subschema())
    errors = [error.message for error in validator.iter_errors(fixture)]
    assert errors == []


@pytest.mark.parametrize(
    "fixture_name",
    [
        "authority-result-envelope.v1.fixture.json",
        "dinger-capture-result.v1.fixture.json",
        "dinger-file-result.v1.fixture.json",
        "os-capture-result.v1.fixture.json",
    ],
)
def test_capture_flow_fixtures_validate_against_docs_helper_envelope(fixture_name: str):
    fixture = _load_json(FIXTURE_ROOT / fixture_name)
    errors = _schema_errors(AUTHORITY_ENVELOPE_SCHEMA_PATH, fixture)
    assert errors == []


def test_capture_flow_docs_helper_pins_stage_policy_and_traceability_progression():
    capture_fixture = _load_json(FIXTURE_ROOT / "dinger-capture-result.v1.fixture.json")
    filed_fixture = _load_json(FIXTURE_ROOT / "dinger-file-result.v1.fixture.json")
    os_fixture = _load_json(FIXTURE_ROOT / "os-capture-result.v1.fixture.json")

    expected_policy = {
        "capturePacket": "input",
        "dingerFiling": "projection_only",
        "osBridge": "inbox_evidence_candidate_only",
        "canonicalStore": "no_new_store",
    }

    assert capture_fixture["stage"] == "captured"
    assert filed_fixture["stage"] == "filed"
    assert os_fixture["stage"] == "linked_to_os"
    assert capture_fixture["flowSemantics"] == expected_policy
    assert filed_fixture["flowSemantics"] == expected_policy
    assert os_fixture["flowSemantics"] == expected_policy

    capture_trace = capture_fixture["traceability"]
    filed_trace = filed_fixture["traceability"]
    os_trace = os_fixture["traceability"]
    assert capture_trace["captureId"] == filed_trace["captureId"] == os_trace["captureId"]
    assert capture_trace["packetPath"] == filed_trace["packetPath"] == os_trace["packetPath"]
    assert "filingOutputPointer" not in capture_trace
    assert filed_trace["filingOutputPointer"] == "KnowledgeOS/Dinger/Pages/rag-capture.md"
    assert os_trace["filingOutputPointer"] == filed_trace["filingOutputPointer"]
    assert os_trace["osBridgeTrace"]["bridge"] == "os_capture"
    assert os_trace["osBridgeTrace"]["schema"] == "knowledge-hub.os.capture.result.v1"
    assert os_trace["osBridgeTrace"]["itemId"] == "inbox_123"
    assert os_trace["osBridgeTrace"]["projectId"] == "proj_123"
    assert os_trace["osBridgeTrace"]["captureTraceBridge"] == "dinger"
    assert os_trace["osBridgeTrace"]["relativePath"] == filed_trace["filingOutputPointer"]


def test_authority_timeout_fixture_stays_classification_only_and_non_canonical():
    fixture = _load_json(FIXTURE_ROOT / "authority-result-envelope.v1.fixture.json")

    assert fixture["stage"] == "failed"
    assert fixture["authorityTimeout"]["handledAs"] == "classification_only"
    assert fixture["flowSemantics"]["capturePacket"] == "input"
    assert fixture["flowSemantics"]["dingerFiling"] == "projection_only"
    assert fixture["flowSemantics"]["osBridge"] == "inbox_evidence_candidate_only"
    assert fixture["flowSemantics"]["canonicalStore"] == "no_new_store"


def test_authority_envelope_fixture_validates_via_python_schema_registry():
    fixture = _load_json(FIXTURE_ROOT / "authority-result-envelope.v1.fixture.json")
    result = validate_payload(fixture, "knowledge-hub.authority.result-envelope.v1", strict=True)
    assert result.ok, result.errors


def test_authority_envelope_fixture_requires_error_for_failed_status():
    fixture = _load_json(FIXTURE_ROOT / "authority-result-envelope.v1.fixture.json")
    fixture.pop("error", None)
    errors = _schema_errors(AUTHORITY_ENVELOPE_SCHEMA_PATH, fixture)
    assert any("error" in message for message in errors)


@pytest.mark.parametrize(
    ("fixture_name", "expected_stage", "expects_projection_pointer", "expects_os_bridge_trace"),
    [
        ("authority-result-envelope.v1.fixture.json", "failed", True, False),
        ("dinger-capture-result.v1.fixture.json", "captured", False, False),
        ("dinger-file-result.v1.fixture.json", "filed", True, False),
        ("os-capture-result.v1.fixture.json", "linked_to_os", True, True),
    ],
)
def test_capture_flow_fixtures_pin_flow_semantics_and_traceability(
    fixture_name: str,
    expected_stage: str,
    expects_projection_pointer: bool,
    expects_os_bridge_trace: bool,
):
    fixture = _load_json(FIXTURE_ROOT / fixture_name)
    assert fixture["stage"] == expected_stage
    assert fixture["flowSemantics"] == {
        "capturePacket": "input",
        "dingerFiling": "projection_only",
        "osBridge": "inbox_evidence_candidate_only",
        "canonicalStore": "no_new_store",
    }

    traceability = fixture["traceability"]
    assert traceability["captureId"]
    assert _pointer_from_traceability(traceability) is not None

    if expects_projection_pointer:
        assert traceability["filingOutputPointer"] == fixture["projectionRelativePath"]
    else:
        assert "filingOutputPointer" not in traceability

    if expects_os_bridge_trace:
        bridge_trace = traceability["osBridgeTrace"]
        assert bridge_trace["bridge"] == "os_capture"
        assert bridge_trace["schema"] == "knowledge-hub.os.capture.result.v1"
        assert bridge_trace["captureTraceBridge"] == "dinger"
        assert bridge_trace["itemId"] == fixture["item"]["id"]
        assert bridge_trace["projectId"] == fixture["item"]["projectId"]
        assert bridge_trace["relativePath"] == fixture["projectionRelativePath"]
    else:
        assert "osBridgeTrace" not in traceability

    if fixture_name == "authority-result-envelope.v1.fixture.json":
        assert fixture["authorityTimeout"]["handledAs"] == "classification_only"


def test_runtime_statuses_stay_command_specific_while_docs_stage_pins_flow_position():
    dinger_file_fixture = _load_json(FIXTURE_ROOT / "dinger-file-result.v1.fixture.json")
    os_capture_fixture = _load_json(FIXTURE_ROOT / "os-capture-result.v1.fixture.json")
    assert dinger_file_fixture["status"] == "ok"
    assert dinger_file_fixture["stage"] == "filed"
    assert os_capture_fixture["status"] == "ok"
    assert os_capture_fixture["stage"] == "linked_to_os"


def test_os_evidence_review_fixture_pins_operator_receipt_without_new_store():
    fixture = _load_json(FIXTURE_ROOT / "os-evidence-review-result.v1.fixture.json")
    assert fixture["action"] == "approve"
    assert fixture["item"]["state"] == "resolved"
    assert fixture["review"]["afterState"] == "resolved"
    assert fixture["review"]["receipt"] == {
        "chosenAction": "approve",
        "semanticMeaning": "reviewed_for_manual_promotion",
        "operatorMeaning": "Resolve this inbox item as reviewed while keeping later task or decision promotion explicit.",
        "followUpExpectation": "No task or decision is created here; use explicit promotion later if this evidence should drive work.",
        "resultingInboxState": "resolved",
    }
    assert fixture["review"]["preservesFlow"]["promotion"] == "manual_only"


def test_os_capture_fixture_pins_note_first_dedupe_and_replay_policy():
    fixture = _load_json(FIXTURE_ROOT / "os-capture-result.v1.fixture.json")
    assert fixture["dedupeKey"] == {
        "kind": "dinger_file",
        "strategy": "vault_note",
        "markers": [{"sourceType": "vault", "primary": "KnowledgeOS/Dinger/Pages/rag-capture.md"}],
        "fingerprint": "dinger_file|vault_note|vault:KnowledgeOS/Dinger/Pages/rag-capture.md",
    }
    assert fixture["replay"] == {
        "policy": {"open": "reuse", "resolved": "create_new", "triaged": "create_new"},
        "dedupeKey": fixture["dedupeKey"],
        "action": "created_new_without_prior_match",
        "matchedItemIds": [],
        "matchedStates": [],
    }
    assert fixture["linkAction"] == "created"
    assert fixture["duplicateSourceRefsSkipped"] == 1
    assert fixture["reason"] == {
        "dedupeKeySummary": {
            "kind": "dinger_file",
            "strategy": "vault_note",
            "markerCount": 1,
            "markers": [{"sourceType": "vault", "primary": "KnowledgeOS/Dinger/Pages/rag-capture.md"}],
            "fingerprint": "dinger_file|vault_note|vault:KnowledgeOS/Dinger/Pages/rag-capture.md",
        },
        "replayAction": "created_new_without_prior_match",
        "matchedOpenItems": [],
        "matchedResolvedItems": [],
        "matchedOtherItems": [],
        "bridgeTraceSummary": {
            "bridge": "dinger",
            "sourceSchema": "knowledge-hub.dinger.file.result.v1",
            "kind": "web_capture",
            "relativePath": "KnowledgeOS/Dinger/Pages/rag-capture.md",
            "title": "RAG Capture",
            "captureUrl": "https://example.com/rag",
            "captureId": "cap_123456789abc",
        },
    }
    assert "created_new_without_prior_match" in fixture["explanation"]
    assert "KnowledgeOS/Dinger/Pages/rag-capture.md" in fixture["explanation"]


def test_os_project_evidence_fixture_pins_candidate_reason_and_explanation():
    fixture = _load_json(FIXTURE_ROOT / "os-project-evidence-result.v1.fixture.json")
    candidates = fixture["projectEvidence"]["evidenceCandidates"]
    assert len(candidates) == 1
    assert candidates[0]["reason"] == {
        "kind": "dinger_linked_inbox_candidate",
        "summary": "Open inbox item inbox_001 is linked to filed Dinger page KnowledgeOS/Dinger/Pages/decision-os.md and keeps 1 supporting source ref attached for later manual promotion. rerun reuse=inbox_001(open):Blocked task: Review projection output; replay resolved=none; replay other=none.",
        "inboxState": "open",
        "bridge": "dinger",
        "inboxSummary": "Blocked task: Review projection output",
        "relativePath": "KnowledgeOS/Dinger/Pages/decision-os.md",
        "supportingSourceRefCount": 1,
        "sourceTypes": ["vault", "web"],
        "upstreamSourceRefs": [
            {
                "sourceType": "web",
                "url": "https://example.com/decision-os",
                "title": "Design note",
            }
        ],
        "replayAction": "reused_existing_open_item",
        "replayPolicy": {"open": "reuse", "resolved": "create_new", "triaged": "create_new"},
        "matchedOpenItems": [
            {
                "id": "inbox_001",
                "state": "open",
                "summary": "Blocked task: Review projection output",
            }
        ],
        "matchedResolvedItems": [],
        "matchedOtherItems": [],
        "dedupeKeySummary": {
            "kind": "dinger_file",
            "strategy": "vault_note",
            "markerCount": 1,
            "markers": [{"sourceType": "vault", "primary": "KnowledgeOS/Dinger/Pages/decision-os.md"}],
            "fingerprint": "dinger_file|vault_note|vault:KnowledgeOS/Dinger/Pages/decision-os.md",
        },
        "bridgeTraceSummary": {
            "bridge": "dinger",
            "sourceSchema": "",
            "kind": "",
            "relativePath": "KnowledgeOS/Dinger/Pages/decision-os.md",
            "title": "Decision OS",
        },
    }
    assert candidates[0]["explanation"] == candidates[0]["reason"]["summary"]


def test_dinger_file_schema_rejects_source_ref_without_primary_identifier():
    fixture = _load_json(FIXTURE_ROOT / "dinger-file-result.v1.fixture.json")
    fixture["sourceRefs"] = [{"sourceType": "paper"}]
    result = validate_payload(fixture, "knowledge-hub.dinger.file.result.v1", strict=True)
    assert result.ok is False
    assert any("/sourceRefs/0" in error and "paperId" in error for error in result.errors)


def test_os_capture_schema_rejects_missing_project_scope_or_invalid_severity():
    fixture = _load_json(FIXTURE_ROOT / "os-capture-result.v1.fixture.json")
    fixture["item"]["severity"] = "urgent-ish"  # type: ignore[index]
    fixture["item"].pop("projectId", None)  # type: ignore[index]
    result = validate_payload(fixture, "knowledge-hub.os.capture.result.v1", strict=True)
    assert result.ok is False
    assert any("/item/projectId" in error or "/item/severity" in error for error in result.errors)


def test_python_normalization_preserves_agent_run_fixture_contract():
    fixture = _load_json(FIXTURE_ROOT / "agent-run-result.v1.fixture.json")
    normalized = _normalize_run_payload(
        fixture,
        goal=str(fixture["goal"]),
        max_rounds=int(fixture["maxRounds"]),
        dry_run=bool(fixture["dryRun"]),
        role=str(fixture["role"]),
        orchestrator_mode=str(fixture["orchestratorMode"]),
    )
    assert normalized["schema"] == fixture["schema"]
    assert normalized["runId"] == fixture["runId"]
    assert normalized["playbook"]["schema"] == "knowledge-hub.foundry.agent.run.playbook.v1"
    assert normalized["playbook"]["steps"][0]["tool"] == fixture["playbook"]["steps"][0]["tool"]


def test_python_normalization_preserves_playbook_fixture_contract():
    fixture = _load_json(FIXTURE_ROOT / "agent-run-playbook.v1.fixture.json")
    normalized = _normalize_playbook(
        fixture,
        goal=str(fixture["goal"]),
        role=str(fixture["role"]),
        orchestrator_mode=str(fixture["orchestratorMode"]),
        max_rounds=int(fixture["maxRounds"]),
        source=str(fixture["source"]),
        plan=["search_knowledge", "ask_knowledge"],
        now=str(fixture["generatedAt"]),
    )
    assert normalized["schema"] == fixture["schema"]
    assert normalized["goal"] == fixture["goal"]
    assert normalized["steps"][0]["tool"] == fixture["steps"][0]["tool"]


def test_evaluate_policy_gate_matches_shared_p0_corpus():
    cases = _load_json(POLICY_CORPUS_PATH)["cases"]  # type: ignore[index]
    for case in cases:
        payload = {"jsonContent": case["text"]}
        allowed, _errors, classification = evaluate_policy_gate(payload)
        assert (classification == "P0") is bool(case["expectedP0"]), case["id"]
        assert allowed is (not bool(case["expectedP0"])), case["id"]


def test_evaluate_policy_gate_matches_shared_foundry_conformance_cases():
    cases = _load_json(POLICY_CONFORMANCE_PATH)["cases"]  # type: ignore[index]
    for case in cases:
        expected = case["expected"]
        allowed, errors, classification = evaluate_policy_gate(case["payload"])
        assert allowed is bool(expected["allowed"]), case["id"]
        assert classification == expected["classification"], case["id"]
        if not allowed:
            assert errors, case["id"]


def test_evaluate_policy_gate_uses_most_sensitive_non_p0_classification():
    allowed, errors, classification = evaluate_policy_gate(
        {
            "classification": "P3",
            "jsonContent": '{"id":"paper-1","score":0.91}',
        }
    )
    assert allowed is True
    assert errors == []
    assert classification == "P1"

    allowed, errors, classification = evaluate_policy_gate(
        {
            "classification": "P1",
            "jsonContent": "plain public summary",
        }
    )
    assert allowed is True
    assert errors == []
    assert classification == "P1"


def _parse_supported_re_flags(raw_flags: object | None) -> int:
    flags = 0
    for flag in str(raw_flags or ""):
        if flag == "i":
            flags |= re.IGNORECASE
        elif flag == "m":
            flags |= re.MULTILINE
        elif flag == "s":
            flags |= re.DOTALL
        else:
            raise AssertionError(f"unsupported regex flag in shared P0 config: {flag!r}")
    return flags


def _pattern_signature(pattern: re.Pattern[str]) -> tuple[str, int]:
    supported = re.IGNORECASE | re.MULTILINE | re.DOTALL
    return pattern.pattern, int(pattern.flags & supported)


def test_python_p0_loader_matches_shared_pattern_source_contract():
    source = _load_json(P0_PATTERN_CONFIG_PATH)
    assert isinstance(source.get("version"), str) and str(source["version"]).strip()

    raw_patterns = source.get("patterns")
    assert isinstance(raw_patterns, list) and raw_patterns
    expected_patterns: list[tuple[str, int]] = []
    for index, raw_pattern in enumerate(raw_patterns):
        assert isinstance(raw_pattern, dict), f"patterns[{index}] must be an object"
        regex_text = str(raw_pattern.get("regex", "") or "").strip()
        assert regex_text, f"patterns[{index}].regex must be non-empty"
        compiled = re.compile(regex_text, _parse_supported_re_flags(raw_pattern.get("flags")))
        expected_patterns.append(_pattern_signature(compiled))

    raw_redact_keys = source.get("redactKeys")
    assert isinstance(raw_redact_keys, list) and raw_redact_keys
    expected_redact_keys = sorted(
        {
            str(key).strip().lower()
            for key in raw_redact_keys
            if str(key).strip()
        }
    )
    assert expected_redact_keys

    get_p0_detection_config.cache_clear()
    loaded = get_p0_detection_config()
    loaded_patterns = [_pattern_signature(pattern) for pattern in loaded["patterns"]]
    loaded_redact_keys = sorted(str(key).strip().lower() for key in loaded["redact_keys"])

    assert loaded["version"] == source["version"]
    assert loaded_patterns == expected_patterns
    assert loaded_redact_keys == expected_redact_keys


def test_temp_runtime_capture_process_smoke_stays_local_and_reuses_existing_open_item(monkeypatch, tmp_path: Path):
    from knowledge_hub.interfaces.cli.commands.dinger_cmd import dinger_group

    runner = CliRunner()
    khub = _AuthorityStubKhub(_make_capture_runtime_config(tmp_path))

    captured = runner.invoke(
        dinger_group,
        [
            "capture",
            "--source-url",
            "https://example.com/rag",
            "--page-title",
            "RAG Capture",
            "--selection-text",
            "RAG combines retrieval with generation.",
            "--client",
            "browser-clipper",
            "--tag",
            "rag",
            "--json",
        ],
        obj={"khub": khub},
    )
    assert captured.exit_code == 0
    capture_payload = json.loads(captured.output)
    assert validate_payload(capture_payload, "knowledge-hub.dinger.capture.result.v1", strict=True).ok
    packet_path = Path(str(capture_payload["packetPath"]))

    captured_calls: list[list[str]] = []

    def _fake_run(args, timeout_sec=120):  # noqa: ANN001
        _ = timeout_sec
        captured_calls.append(list(args))
        assert args[:2] == ["inbox", "list"]
        return (
            {
                "schema": "knowledge-hub.os.inbox.list.result.v1",
                "status": "ok",
                "count": 1,
                "items": [
                    {
                        "id": "inbox_existing",
                        "projectId": "proj_123",
                        "kind": "dinger_file",
                        "severity": "medium",
                        "state": "open",
                        "summary": "Dinger file: RAG Capture",
                        "sourceRefs": [
                            {
                                "sourceType": "vault",
                                "noteId": "KnowledgeOS/Dinger/Pages/rag-capture.md",
                                "title": "RAG Capture",
                            },
                            {
                                "sourceType": "web",
                                "url": "https://example.com/rag",
                                "title": "RAG Capture",
                            },
                        ],
                        "createdAt": "2026-04-09T12:10:00+00:00",
                        "updatedAt": "2026-04-09T12:10:00+00:00",
                    }
                ],
                "createdAt": "2026-04-09T12:10:00+00:00",
            },
            None,
        )

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.dinger_cmd.run_foundry_project_cli", _fake_run)

    first = runner.invoke(
        dinger_group,
        ["capture-process", "--packet", str(packet_path), "--slug", "decision-os", "--json"],
        obj={"khub": khub},
    )
    assert first.exit_code == 0
    first_payload = json.loads(first.output)
    assert validate_payload(first_payload, "knowledge-hub.dinger.capture-process.result.v1", strict=True).ok

    first_item = first_payload["items"][0]
    normalized_payload = _load_json(Path(first_item["normalizedPath"]))
    filed_payload = _load_json(Path(first_item["filedResultPath"]))
    os_payload = _load_json(Path(first_item["osResultPath"]))
    state_payload = _load_json(Path(first_item["statePath"]))

    assert validate_payload(filed_payload, "knowledge-hub.dinger.file.result.v1", strict=True).ok
    assert validate_payload(os_payload, "knowledge-hub.os.capture.result.v1", strict=True).ok
    assert normalized_payload["captureId"] == capture_payload["captureId"]
    assert normalized_payload["packetStatus"] == "captured"
    assert normalized_payload["captureUrl"] == "https://example.com/rag"
    assert first_item["status"] == "linked_to_os"
    assert first_item["linkAction"] == "reused_existing"
    assert first_item["idempotent"] is False
    assert os_payload["linkAction"] == "reused_existing"
    assert os_payload["duplicateSourceRefsSkipped"] == 1
    assert os_payload["dedupeKey"]["strategy"] == "vault_note"
    assert os_payload["replay"]["action"] == "reused_existing_open_item"
    assert os_payload["reason"]["replayAction"] == "reused_existing_open_item"
    assert os_payload["reason"]["matchedOpenItems"] == [
        {
            "id": "inbox_existing",
            "state": "open",
            "summary": "Dinger file: RAG Capture",
        }
    ]
    assert os_payload["reason"]["matchedResolvedItems"] == []
    assert "reused_existing_open_item" in os_payload["explanation"]
    assert state_payload["attemptCount"] == 1
    assert state_payload["currentStatus"] == "linked_to_os"
    assert state_payload["steps"]["linked_to_os"]["linkAction"] == "reused_existing"
    assert state_payload["osBridge"]["itemId"] == "inbox_existing"
    assert state_payload["osBridge"]["dedupeKey"]["strategy"] == "vault_note"
    assert state_payload["osBridge"]["replay"]["policy"] == {
        "open": "reuse",
        "resolved": "create_new",
        "triaged": "create_new",
    }
    assert state_payload["osBridge"]["reason"]["replayAction"] == "reused_existing_open_item"
    assert "reused_existing_open_item" in state_payload["osBridge"]["explanation"]

    second = runner.invoke(
        dinger_group,
        ["capture-process", "--packet", str(packet_path), "--slug", "decision-os", "--json"],
        obj={"khub": khub},
    )
    assert second.exit_code == 0
    second_payload = json.loads(second.output)
    assert validate_payload(second_payload, "knowledge-hub.dinger.capture-process.result.v1", strict=True).ok
    assert second_payload["counts"]["idempotent"] == 1
    assert second_payload["items"][0]["idempotent"] is True

    replay_state = _load_json(Path(second_payload["items"][0]["statePath"]))
    assert replay_state["attemptCount"] == 2
    assert replay_state["currentStatus"] == "linked_to_os"
    assert captured_calls == [
        ["inbox", "list", "--state", "open", "--slug", "decision-os", "--ops-alerts-json", "[]"],
    ]

# Known follow-up: this round-trip is not the final green gate for the full
# authority file until slow tsx-backed bridge environments are isolated. On this
# worktree the command path is fast; the flaky surface is the subprocess bridge,
# not the payload validation assertions below.
def test_ts_project_cli_outputs_validate_in_python(tmp_path: Path):
    created = _run_ts_project_cli(
        tmp_path,
        [
            "project",
            "create",
            "--title",
            "Authority Hardening",
            "--slug",
            "authority-hardening",
            "--summary",
            "authority contract smoke",
            "--owner",
            "operator",
        ],
    )
    create_result = validate_payload(created, str(created["schema"]), strict=True)
    assert create_result.ok, create_result.errors

    shown = _run_ts_project_cli(
        tmp_path,
        [
            "project",
            "show",
            "--slug",
            "authority-hardening",
        ],
    )
    show_result = validate_payload(shown, str(shown["schema"]), strict=True)
    assert show_result.ok, show_result.errors

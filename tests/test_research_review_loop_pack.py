from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.interfaces.cli.commands.review_loop_cmd import review_loop_group
from knowledge_hub.core.schema_validator import validate_payload
from tests.research_review_loop_fixtures import (
    EVALUATION_PAPER_ID,
    MISSING_LOCATOR_CLAIM,
    REJECTED_CLAIM,
    SUPPORTED_CLAIM,
    UNREVIEWED_CLAIM,
    UNSUPPORTED_CLAIM,
    ReviewLoopDb,
    ReviewLoopKhub,
    claim_text_hash,
)


PACK_SCHEMA = "knowledge-hub.research-review-loop.pack.v1"


def test_research_review_loop_accept_decision_without_evidence_is_not_canonical(tmp_path: Path) -> None:
    # Given: a reviewed accept decision for a claim with no evidence spans.
    decision_file = tmp_path / "decisions.json"
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-unsupported",
                        "targetType": "claim",
                        "targetId": "claim-unsupported",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:00:00Z",
                        "reason": "The user accepted the claim, but no span exists.",
                        "claimTextHash": claim_text_hash(UNSUPPORTED_CLAIM),
                        "snippetHashes": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the report is generated through the labs CLI.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the accepted-but-unsupported claim is reviewed but cannot become canonical.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    unsupported = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-unsupported")
    assert unsupported["state"] == "accepted"
    assert unsupported["reviewDecisionId"] == "decision-unsupported"
    assert unsupported["canonicalEligible"] is False
    assert "missing valid evidence span" in unsupported["warnings"]
    assert payload["counts"]["unsupportedCanonicalRows"] == 1


def test_research_review_loop_emit_pack_writes_negative_constraints_and_excludes_unreviewed(
    tmp_path: Path,
) -> None:
    # Given: one accepted claim, one rejected claim, and one unreviewed claim.
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "pack"
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-accepted",
                        "targetType": "claim",
                        "targetId": "claim-supported",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:00:00Z",
                        "reason": "Evidence span directly supports the claim.",
                        "claimTextHash": claim_text_hash(SUPPORTED_CLAIM),
                        "snippetHashes": ["hash-supported"],
                    },
                    {
                        "decisionId": "decision-rejected",
                        "targetType": "claim",
                        "targetId": "claim-rejected",
                        "decision": "reject",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:01:00Z",
                        "reason": "The paper scope is narrower than this claim.",
                        "claimTextHash": claim_text_hash(REJECTED_CLAIM),
                        "snippetHashes": ["hash-rejected"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the pack is emitted twice from the same reviewed report payload.
    first = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )
    assert first.exit_code == 0, first.output
    pack_md = output_dir / "research_review_loop_pack.md"
    pack_json = output_dir / "research_review_loop_pack.json"
    first_markdown = pack_md.read_text(encoding="utf-8")
    first_sidecar = pack_json.read_text(encoding="utf-8")

    second = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the emitted pack is deterministic and safe for the A/B/C evaluation.
    assert second.exit_code == 0, second.output
    assert pack_md.read_text(encoding="utf-8") == first_markdown
    assert pack_json.read_text(encoding="utf-8") == first_sidecar
    assert "## REJECTED CLAIMS - DO NOT ASSERT" in first_markdown
    assert REJECTED_CLAIM in first_markdown
    assert "The paper scope is narrower than this claim." in first_markdown
    assert SUPPORTED_CLAIM in first_markdown
    assert UNREVIEWED_CLAIM not in first_markdown
    sidecar = json.loads(first_sidecar)
    assert validate_payload(sidecar, PACK_SCHEMA, strict=True).ok
    assert sidecar["canonicalWriteAllowed"] is False
    assert sidecar["acceptedClaimIds"] == ["claim-supported"]
    assert sidecar["rejectedClaimIds"] == ["claim-rejected"]


def test_research_review_loop_accept_decision_with_only_missing_locator_span_is_not_canonical(
    tmp_path: Path,
) -> None:
    # Given: an accept decision for a claim whose only evidence span lacks a locator.
    decision_file = tmp_path / "decisions.json"
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-missing-locator",
                        "targetType": "claim",
                        "targetId": "claim-missing-locator",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:02:00Z",
                        "reason": "The user accepted the claim, but the span cannot be located.",
                        "claimTextHash": claim_text_hash(MISSING_LOCATOR_CLAIM),
                        "snippetHashes": ["hash-missing-locator"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the report is generated through the labs CLI.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the reviewed claim is not canonical because blocked spans do not count.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    claim = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-missing-locator")
    span = next(item for item in payload["evidenceSpans"] if item["evidenceSpanId"] == "span-missing-locator")
    assert claim["state"] == "accepted"
    assert claim["canonicalEligible"] is False
    assert "missing valid evidence span" in claim["warnings"]
    assert span["state"] == "blocked_missing_locator"
    assert payload["counts"]["unsupportedCanonicalRows"] == 1


def test_research_review_loop_report_surfaces_structured_authority_blockers_for_noncanonical_accepts(
    tmp_path: Path,
) -> None:
    # Given: a user-reviewed accept decision whose cited span cannot be located.
    decision_file = tmp_path / "decisions.json"
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-missing-locator",
                        "targetType": "claim",
                        "targetId": "claim-missing-locator",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:02:00Z",
                        "reason": "The user accepted the claim, but the span cannot be located.",
                        "claimTextHash": claim_text_hash(MISSING_LOCATOR_CLAIM),
                        "snippetHashes": ["hash-missing-locator"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the review report is generated.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the noncanonical accept exposes machine-readable authority blockers.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    claim = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-missing-locator")
    span = next(item for item in payload["evidenceSpans"] if item["evidenceSpanId"] == "span-missing-locator")
    assert claim["state"] == "accepted"
    assert claim["canonicalEligible"] is False
    assert claim["authorityStatus"] == "blocked"
    assert claim["canonicalBlockers"] == ["missing_locatable_evidence_span"]
    assert span["state"] == "blocked_missing_locator"
    assert span["provenanceStatus"] == "blocked_missing_locator"
    assert payload["counts"]["blockedEvidenceSpanRows"] == 1
    assert payload["counts"]["authoritativeAssertionRows"] == 0


def test_research_review_loop_reviewed_status_reports_zero_authoritative_assertions(
    tmp_path: Path,
) -> None:
    # Given: a human review exists, but no accepted claim has authoritative evidence.
    decision_file = tmp_path / "decisions.json"
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-missing-locator",
                        "targetType": "claim",
                        "targetId": "claim-missing-locator",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:02:00Z",
                        "reason": "The user accepted the claim, but the span cannot be located.",
                        "claimTextHash": claim_text_hash(MISSING_LOCATOR_CLAIM),
                        "snippetHashes": ["hash-missing-locator"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the report is generated.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: review completion is not conflated with authoritative memory.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "reviewed_no_authoritative_assertions"
    assert payload["contextPackPreview"]["authoritativeAssertCount"] == 0
    assert payload["contextPackPreview"]["reviewedButNoAuthoritativeAssertions"] is True


def test_research_review_loop_emit_pack_marks_reviewed_zero_authority_without_assert_lines(
    tmp_path: Path,
) -> None:
    # Given: an accepted review decision whose only cited span is blocked.
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "pack"
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-missing-locator",
                        "targetType": "claim",
                        "targetId": "claim-missing-locator",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:10:00Z",
                        "reason": "The user accepted the claim, but the span cannot become source-backed memory.",
                        "claimTextHash": claim_text_hash(MISSING_LOCATOR_CLAIM),
                        "snippetHashes": ["hash-missing-locator"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: a pack is emitted.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the pack is review-complete but contains no ASSERT rows.
    assert result.exit_code == 0, result.output
    markdown = (output_dir / "research_review_loop_pack.md").read_text(encoding="utf-8")
    sidecar = json.loads((output_dir / "research_review_loop_pack.json").read_text(encoding="utf-8"))
    assert "ASSERT:" not in markdown
    assert "## ACCEPTED CLAIMS\n- none" in markdown
    assert sidecar["acceptedClaimIds"] == []
    assert sidecar["authoritativeAssertCount"] == 0
    assert sidecar["reviewedButNoAuthoritativeAssertions"] is True


def test_research_review_loop_context_pack_demotes_memory_cards_to_reviewed_artifacts_only(
    tmp_path: Path,
) -> None:
    # Given: a normal authoritative reviewed claim pack.
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "pack"
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-accepted",
                        "targetType": "claim",
                        "targetId": "claim-supported",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:11:00Z",
                        "reason": "Evidence span directly supports the claim.",
                        "claimTextHash": claim_text_hash(SUPPORTED_CLAIM),
                        "snippetHashes": ["hash-supported"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the report and context pack are emitted.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: generated MemoryCards are explicitly excluded from canonical context.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    sidecar = json.loads((output_dir / "research_review_loop_pack.json").read_text(encoding="utf-8"))
    expected_policy = {
        "memoryCards": "excluded_generated_unreviewed",
        "canonicalMemoryCardWriteAllowed": False,
        "projectionAllowedFromReviewedArtifactsOnly": True,
    }
    assert payload["contextPackPreview"]["memoryProjectionPolicy"] == expected_policy
    assert sidecar["memoryProjectionPolicy"] == expected_policy


def test_research_review_loop_emit_pack_reports_display_safe_output_paths(tmp_path: Path) -> None:
    # Given: absolute local output directories for a report and emitted pack.
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "report"
    pack_dir = tmp_path / "pack"
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-accepted",
                        "targetType": "claim",
                        "targetId": "claim-supported",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:22:00Z",
                        "reason": "Evidence span directly supports the claim.",
                        "claimTextHash": claim_text_hash(SUPPORTED_CLAIM),
                        "snippetHashes": ["hash-supported"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the CLI writes local artifacts and returns JSON.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--out-dir",
            str(output_dir),
            "--emit-pack",
            str(pack_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the JSON payload does not expose absolute local filesystem paths.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    preview = payload["contextPackPreview"]
    assert preview["outputPath"] == "research_review_loop_report.json"
    assert preview["packMarkdownPath"] == "research_review_loop_pack.md"
    assert preview["packJsonPath"] == "research_review_loop_pack.json"
    assert str(tmp_path) not in result.output


def test_research_review_loop_decision_with_unknown_snippet_hash_is_not_applied(tmp_path: Path) -> None:
    # Given: a decision that cites a span hash not present in the claim evidence.
    decision_file = tmp_path / "decisions.json"
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-wrong-snippet",
                        "targetType": "claim",
                        "targetId": "claim-supported",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:03:00Z",
                        "reason": "The decision cites the wrong span hash.",
                        "claimTextHash": claim_text_hash(SUPPORTED_CLAIM),
                        "snippetHashes": ["hash-not-present"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the report is generated.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--decision-file",
            str(decision_file),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the stale decision remains visible but is not applied to memory state.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    claim = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-supported")
    assert claim["state"] == "proposed"
    assert claim["reviewDecisionId"] is None
    assert payload["counts"]["reviewDecisionRows"] == 1
    assert payload["counts"]["appliedReviewDecisionRows"] == 0

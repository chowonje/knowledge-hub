from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.interfaces.cli.commands.review_loop_cmd import review_loop_group
from tests.research_review_loop_fixtures import (
    EVALUATION_PAPER_ID,
    ReviewLoopDb,
    ReviewLoopKhub,
    claim_text_hash,
)


VALID_CLAIMS_FILE_CLAIM = "The paper evaluates scientific taste with an explicit OOD-year claim."
MISSING_LOCATOR_CLAIMS_FILE_CLAIM = "The paper claim is accepted but cannot be located in the source."
MEMORY_UNIT_LOCATOR_CLAIMS_FILE_CLAIM = "The paper claim cites a memory unit that has not been resolved to source text."
RESOLVED_MEMORY_UNIT_CLAIMS_FILE_CLAIM = "The paper claim cites a memory unit resolved to source character offsets."
SECTION_LOCATOR_CLAIMS_FILE_CLAIM = "The paper claim cites a section locator instead of character offsets."
MISSING_SOURCE_HASH_CLAIMS_FILE_CLAIM = "The paper claim has character offsets but no source content hash."
EMPTY_SNIPPET_DECISION_CLAIM = "The paper claim has source evidence but the review decision cites no snippet."
CITES_BLOCKED_SPAN_CLAIM = "The paper claim has one source span but the review decision cites only a blocked span."
CROSS_SOURCE_SCOPE_CLAIM = "The paper claim cites evidence from outside the explicit paper scope."
EMPTY_SNIPPET_SPAN_CLAIM = "The paper claim has source offsets but no snippet text or explicit snippet hash."
EMPTY_SNIPPET_TEXT_EXPLICIT_HASH_CLAIM = (
    "The paper claim has source offsets and an explicit snippet hash but no snippet text."
)
UNREVIEWED_CLAIMS_FILE_CLAIM = "This claims-file row remains proposed and must not enter the pack."


def test_research_review_loop_claims_file_flows_valid_reviewed_claim_to_pack(
    tmp_path: Path,
) -> None:
    # Given: a claims file with one valid reviewed claim, one blocked reviewed claim, and one unreviewed claim.
    claims_file = tmp_path / "claims.json"
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "pack"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claimId": "claim-file-valid",
                        "claimText": VALID_CLAIMS_FILE_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-valid",
                                "locator": "chars:10-74",
                                "quote": "The paper evaluates scientific taste on an OOD-year split.",
                                "snippetHash": "hash-file-valid",
                                "sourceContentHash": "source-hash-valid",
                            }
                        ],
                    },
                    {
                        "claimId": "claim-file-missing-locator",
                        "claimText": MISSING_LOCATOR_CLAIMS_FILE_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-missing-locator",
                                "quote": "This quote has no stable locator.",
                                "snippetHash": "hash-file-missing-locator",
                            }
                        ],
                    },
                    {
                        "claimId": "claim-file-unreviewed",
                        "claimText": UNREVIEWED_CLAIMS_FILE_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-unreviewed",
                                "locator": "chars:75-114",
                                "quote": "This row remains unreviewed.",
                                "snippetHash": "hash-file-unreviewed",
                                "sourceContentHash": "source-hash-unreviewed",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-file-valid",
                        "targetType": "claim",
                        "targetId": "claim-file-valid",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:04:00Z",
                        "reason": "The locator-backed span supports the claim.",
                        "claimTextHash": claim_text_hash(VALID_CLAIMS_FILE_CLAIM),
                        "snippetHashes": ["hash-file-valid"],
                    },
                    {
                        "decisionId": "decision-file-missing-locator",
                        "targetType": "claim",
                        "targetId": "claim-file-missing-locator",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:05:00Z",
                        "reason": "The claim cannot become memory without a locator.",
                        "claimTextHash": claim_text_hash(MISSING_LOCATOR_CLAIMS_FILE_CLAIM),
                        "snippetHashes": ["hash-file-missing-locator"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the labs CLI emits a context pack from the explicit claims file.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--claims-file",
            str(claims_file),
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: only the valid reviewed claims-file row enters the consumable ASSERT pack.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    valid = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-valid")
    missing = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-missing-locator")
    assert valid["state"] == "accepted"
    assert valid["canonicalEligible"] is True
    assert missing["state"] == "accepted"
    assert missing["canonicalEligible"] is False
    assert payload["counts"]["unsupportedCanonicalRows"] == 1
    markdown = (output_dir / "research_review_loop_pack.md").read_text(encoding="utf-8")
    sidecar = json.loads((output_dir / "research_review_loop_pack.json").read_text(encoding="utf-8"))
    assert VALID_CLAIMS_FILE_CLAIM in markdown
    assert "chars:10-74 [hash-file-valid]" in markdown
    assert MISSING_LOCATOR_CLAIMS_FILE_CLAIM not in markdown
    assert UNREVIEWED_CLAIMS_FILE_CLAIM not in markdown
    assert sidecar["acceptedClaimIds"] == ["claim-file-valid"]


def test_research_review_loop_claims_file_blocks_authority_when_accept_decision_has_no_cited_snippet(
    tmp_path: Path,
) -> None:
    claims_file = tmp_path / "claims.json"
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "pack"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claimId": "claim-file-empty-snippet-decision",
                        "claimText": EMPTY_SNIPPET_DECISION_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-empty-snippet-valid",
                                "locator": "chars:300-360",
                                "quote": "This source-resolved span is not cited by the review decision.",
                                "snippetHash": "hash-empty-snippet-valid",
                                "sourceContentHash": "source-hash-empty-snippet-valid",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-file-empty-snippet",
                        "targetType": "claim",
                        "targetId": "claim-file-empty-snippet-decision",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:15:00Z",
                        "reason": "The decision omitted the cited evidence hash.",
                        "claimTextHash": claim_text_hash(EMPTY_SNIPPET_DECISION_CLAIM),
                        "snippetHashes": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--claims-file",
            str(claims_file),
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    claim = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-empty-snippet-decision")
    sidecar = json.loads((output_dir / "research_review_loop_pack.json").read_text(encoding="utf-8"))
    assert claim["state"] == "accepted"
    assert claim["canonicalEligible"] is False
    assert claim["canonicalBlockers"] == ["review_decision_missing_snippet_hash"]
    assert sidecar["acceptedClaimIds"] == []


def test_research_review_loop_claims_file_blocks_authority_when_decision_cites_only_blocked_span(
    tmp_path: Path,
) -> None:
    claims_file = tmp_path / "claims.json"
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "pack"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claimId": "claim-file-cites-blocked-span",
                        "claimText": CITES_BLOCKED_SPAN_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-cites-blocked-valid",
                                "locator": "chars:400-460",
                                "quote": "This source-resolved span is valid but not cited.",
                                "snippetHash": "hash-cites-blocked-valid",
                                "sourceContentHash": "source-hash-cites-blocked-valid",
                            },
                            {
                                "evidenceSpanId": "span-file-cites-blocked-section",
                                "locator": "section:4",
                                "quote": "This blocked section span is the only cited evidence.",
                                "snippetHash": "hash-cites-blocked-section",
                                "sourceContentHash": "source-hash-cites-blocked-section",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-file-cites-blocked-span",
                        "targetType": "claim",
                        "targetId": "claim-file-cites-blocked-span",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:16:00Z",
                        "reason": "The accepted decision cites only the blocked section span.",
                        "claimTextHash": claim_text_hash(CITES_BLOCKED_SPAN_CLAIM),
                        "snippetHashes": ["hash-cites-blocked-section"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--claims-file",
            str(claims_file),
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    claim = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-cites-blocked-span")
    sidecar = json.loads((output_dir / "research_review_loop_pack.json").read_text(encoding="utf-8"))
    assert claim["state"] == "accepted"
    assert claim["canonicalEligible"] is False
    assert claim["canonicalBlockers"] == ["non_offset_locator"]
    assert sidecar["acceptedClaimIds"] == []


def test_research_review_loop_claims_file_blocks_authority_when_span_source_is_outside_explicit_scope(
    tmp_path: Path,
) -> None:
    claims_file = tmp_path / "claims.json"
    decision_file = tmp_path / "decisions.json"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claimId": "claim-file-cross-source",
                        "claimText": CROSS_SOURCE_SCOPE_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-cross-source",
                                "sourceId": "foreign-paper-9999.99999",
                                "locator": "chars:500-560",
                                "quote": "This evidence comes from a different paper.",
                                "snippetHash": "hash-cross-source",
                                "sourceContentHash": "source-hash-cross-source",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-file-cross-source",
                        "targetType": "claim",
                        "targetId": "claim-file-cross-source",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:17:00Z",
                        "reason": "The decision cites a span outside the explicit paper scope.",
                        "claimTextHash": claim_text_hash(CROSS_SOURCE_SCOPE_CLAIM),
                        "snippetHashes": ["hash-cross-source"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--claims-file",
            str(claims_file),
            "--decision-file",
            str(decision_file),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    claim = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-cross-source")
    span = next(item for item in payload["evidenceSpans"] if item["evidenceSpanId"] == "span-file-cross-source")
    assert claim["state"] == "accepted"
    assert claim["canonicalEligible"] is False
    assert claim["canonicalBlockers"] == ["source_outside_explicit_scope"]
    assert span["state"] == "blocked_source_scope_mismatch"
    assert span["canonicalBlockers"] == ["source_outside_explicit_scope"]


def test_research_review_loop_claims_file_blocks_synthetic_empty_snippet_hash_from_authority(
    tmp_path: Path,
) -> None:
    claims_file = tmp_path / "claims.json"
    decision_file = tmp_path / "decisions.json"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claimId": "claim-file-empty-snippet-span",
                        "claimText": EMPTY_SNIPPET_SPAN_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-empty-snippet",
                                "locator": "chars:600-660",
                                "sourceContentHash": "source-hash-empty-snippet",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-file-empty-snippet-span",
                        "targetType": "claim",
                        "targetId": "claim-file-empty-snippet-span",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:18:00Z",
                        "reason": "The decision cites the old synthetic empty-content SHA-1 prefix.",
                        "claimTextHash": claim_text_hash(EMPTY_SNIPPET_SPAN_CLAIM),
                        "snippetHashes": ["da39a3ee5e6b4b0d"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--claims-file",
            str(claims_file),
            "--decision-file",
            str(decision_file),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    claim = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-empty-snippet-span")
    span = next(item for item in payload["evidenceSpans"] if item["evidenceSpanId"] == "span-file-empty-snippet")
    assert claim["state"] == "proposed"
    assert claim["canonicalEligible"] is False
    assert span["snippetHash"] == ""
    assert span["state"] == "blocked_missing_hash"
    assert span["canonicalBlockers"] == ["missing_snippet_hash"]


def test_research_review_loop_claims_file_blocks_explicit_hash_without_snippet_text_from_authority(
    tmp_path: Path,
) -> None:
    claims_file = tmp_path / "claims.json"
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "pack"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claimId": "claim-file-explicit-hash-empty-snippet",
                        "claimText": EMPTY_SNIPPET_TEXT_EXPLICIT_HASH_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-explicit-hash-empty-snippet",
                                "locator": "chars:660-720",
                                "snippetHash": "hash-explicit-empty-snippet",
                                "sourceContentHash": "source-hash-explicit-empty-snippet",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-file-explicit-hash-empty-snippet",
                        "targetType": "claim",
                        "targetId": "claim-file-explicit-hash-empty-snippet",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:18:30Z",
                        "reason": "The decision cites a hash whose span has no source text excerpt.",
                        "claimTextHash": claim_text_hash(EMPTY_SNIPPET_TEXT_EXPLICIT_HASH_CLAIM),
                        "snippetHashes": ["hash-explicit-empty-snippet"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--claims-file",
            str(claims_file),
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    claim = next(
        item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-explicit-hash-empty-snippet"
    )
    span = next(
        item for item in payload["evidenceSpans"] if item["evidenceSpanId"] == "span-file-explicit-hash-empty-snippet"
    )
    sidecar = json.loads((output_dir / "research_review_loop_pack.json").read_text(encoding="utf-8"))
    assert claim["state"] == "accepted"
    assert claim["canonicalEligible"] is False
    assert claim["canonicalBlockers"] == ["missing_snippet_text"]
    assert span["snippetHash"] == "hash-explicit-empty-snippet"
    assert span["textPreview"] == ""
    assert span["state"] == "blocked_missing_snippet_text"
    assert span["canonicalBlockers"] == ["missing_snippet_text"]
    assert sidecar["acceptedClaimIds"] == []


def test_research_review_loop_claims_file_blocks_unresolved_memory_unit_locator_from_canonical_pack(
    tmp_path: Path,
) -> None:
    # Given: a claims-file row whose locator still points at a generated MemoryCard unit.
    claims_file = tmp_path / "claims.json"
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "pack"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claimId": "claim-file-memory-unit-locator",
                        "claimText": MEMORY_UNIT_LOCATOR_CLAIMS_FILE_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-memory-unit-locator",
                                "locator": "memory-unit:paper:2603.14473:summary",
                                "quote": "The claim cites a generated summary unit instead of source characters.",
                                "snippetHash": "hash-file-memory-unit-locator",
                                "sourceContentHash": "source-hash-memory-unit-locator",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-file-memory-unit-locator",
                        "targetType": "claim",
                        "targetId": "claim-file-memory-unit-locator",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:06:00Z",
                        "reason": "The reviewer accepted the claim, but the locator is not source-resolved.",
                        "claimTextHash": claim_text_hash(MEMORY_UNIT_LOCATOR_CLAIMS_FILE_CLAIM),
                        "snippetHashes": ["hash-file-memory-unit-locator"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the labs CLI emits a pack from the claims file.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--claims-file",
            str(claims_file),
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the reviewed claim remains visible but cannot become an ASSERT pack row.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    claim = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-memory-unit-locator")
    span = next(
        item for item in payload["evidenceSpans"] if item["evidenceSpanId"] == "span-file-memory-unit-locator"
    )
    assert claim["state"] == "accepted"
    assert claim["canonicalEligible"] is False
    assert claim["canonicalBlockers"] == ["unresolved_memory_unit_locator"]
    assert span["state"] == "blocked_unresolved_memory_unit_locator"
    assert span["provenanceStatus"] == "blocked_unresolved_memory_unit_locator"
    markdown = (output_dir / "research_review_loop_pack.md").read_text(encoding="utf-8")
    sidecar = json.loads((output_dir / "research_review_loop_pack.json").read_text(encoding="utf-8"))
    assert MEMORY_UNIT_LOCATOR_CLAIMS_FILE_CLAIM not in markdown
    assert sidecar["acceptedClaimIds"] == []


def test_research_review_loop_claims_file_resolves_memory_unit_locator_only_with_source_offsets(
    tmp_path: Path,
) -> None:
    # Given: a generated memory-unit locator plus an explicit source-offset resolution.
    claims_file = tmp_path / "claims.json"
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "pack"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claimId": "claim-file-resolved-memory-unit",
                        "claimText": RESOLVED_MEMORY_UNIT_CLAIMS_FILE_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-resolved-memory-unit",
                                "locator": "memory-unit:paper:2603.14473:summary",
                                "resolvedLocator": "chars:200-260",
                                "quote": "The claim cites a generated unit that has been resolved to source text.",
                                "snippetHash": "hash-file-resolved-memory-unit",
                                "sourceContentHash": "source-hash-resolved-memory-unit",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-file-resolved-memory-unit",
                        "targetType": "claim",
                        "targetId": "claim-file-resolved-memory-unit",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:06:30Z",
                        "reason": "The memory-unit pointer was resolved to source offsets before review.",
                        "claimTextHash": claim_text_hash(RESOLVED_MEMORY_UNIT_CLAIMS_FILE_CLAIM),
                        "snippetHashes": ["hash-file-resolved-memory-unit"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the labs CLI emits a pack from the claims file.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--claims-file",
            str(claims_file),
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: the effective citation uses source offsets, while the raw generated locator remains visible.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    claim = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-resolved-memory-unit")
    span = next(
        item for item in payload["evidenceSpans"] if item["evidenceSpanId"] == "span-file-resolved-memory-unit"
    )
    assert claim["canonicalEligible"] is True
    assert claim["authorityStatus"] == "authoritative"
    assert span["locator"] == "chars:200-260"
    assert span["rawLocator"] == "memory-unit:paper:2603.14473:summary"
    assert span["locatorKind"] == "chars_offset"
    assert span["provenanceStatus"] == "source_resolved"
    markdown = (output_dir / "research_review_loop_pack.md").read_text(encoding="utf-8")
    assert "chars:200-260 [hash-file-resolved-memory-unit]" in markdown


def test_research_review_loop_claims_file_requires_chars_locator_and_source_hash_for_canonical_claims_file_span(
    tmp_path: Path,
) -> None:
    # Given: accepted claims-file rows with valid, section-only, and source-hash-missing spans.
    claims_file = tmp_path / "claims.json"
    decision_file = tmp_path / "decisions.json"
    output_dir = tmp_path / "pack"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claimId": "claim-file-valid",
                        "claimText": VALID_CLAIMS_FILE_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-valid",
                                "locator": "chars:10-74",
                                "quote": "The paper evaluates scientific taste on an OOD-year split.",
                                "snippetHash": "hash-file-valid",
                                "sourceContentHash": "source-hash-valid",
                            }
                        ],
                    },
                    {
                        "claimId": "claim-file-section-locator",
                        "claimText": SECTION_LOCATOR_CLAIMS_FILE_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-section-locator",
                                "locator": "section:2",
                                "quote": "This span points at a section, not a source character range.",
                                "snippetHash": "hash-file-section-locator",
                                "sourceContentHash": "source-hash-section-locator",
                            }
                        ],
                    },
                    {
                        "claimId": "claim-file-missing-source-hash",
                        "claimText": MISSING_SOURCE_HASH_CLAIMS_FILE_CLAIM,
                        "sourceId": EVALUATION_PAPER_ID,
                        "evidenceSpans": [
                            {
                                "evidenceSpanId": "span-file-missing-source-hash",
                                "locator": "chars:120-188",
                                "quote": "This span has character offsets but no source content hash.",
                                "snippetHash": "hash-file-missing-source-hash",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    decision_file.write_text(
        json.dumps(
            {
                "reviewDecisions": [
                    {
                        "decisionId": "decision-file-valid",
                        "targetType": "claim",
                        "targetId": "claim-file-valid",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:07:00Z",
                        "reason": "The span is source-resolved.",
                        "claimTextHash": claim_text_hash(VALID_CLAIMS_FILE_CLAIM),
                        "snippetHashes": ["hash-file-valid"],
                    },
                    {
                        "decisionId": "decision-file-section-locator",
                        "targetType": "claim",
                        "targetId": "claim-file-section-locator",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:08:00Z",
                        "reason": "The span is reviewed but not character-offset locatable.",
                        "claimTextHash": claim_text_hash(SECTION_LOCATOR_CLAIMS_FILE_CLAIM),
                        "snippetHashes": ["hash-file-section-locator"],
                    },
                    {
                        "decisionId": "decision-file-missing-source-hash",
                        "targetType": "claim",
                        "targetId": "claim-file-missing-source-hash",
                        "decision": "accept",
                        "confidence": "high",
                        "reviewer": "human",
                        "reviewedAt": "2026-06-10T00:09:00Z",
                        "reason": "The span is reviewed but cannot be tied to source content identity.",
                        "claimTextHash": claim_text_hash(MISSING_SOURCE_HASH_CLAIMS_FILE_CLAIM),
                        "snippetHashes": ["hash-file-missing-source-hash"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    # When: the labs CLI emits a pack from the claims file.
    result = CliRunner().invoke(
        review_loop_group,
        [
            "report",
            "--paper-id",
            EVALUATION_PAPER_ID,
            "--claims-file",
            str(claims_file),
            "--decision-file",
            str(decision_file),
            "--emit-pack",
            str(output_dir),
            "--json",
        ],
        obj={"khub": ReviewLoopKhub(ReviewLoopDb())},
    )

    # Then: only the reviewed row with character offsets and source identity becomes authoritative.
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    valid = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-valid")
    section = next(item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-section-locator")
    missing_hash = next(
        item for item in payload["proposedClaims"] if item["claimId"] == "claim-file-missing-source-hash"
    )
    section_span = next(
        item for item in payload["evidenceSpans"] if item["evidenceSpanId"] == "span-file-section-locator"
    )
    missing_hash_span = next(
        item for item in payload["evidenceSpans"] if item["evidenceSpanId"] == "span-file-missing-source-hash"
    )
    assert valid["canonicalEligible"] is True
    assert valid["authorityStatus"] == "authoritative"
    assert section["canonicalEligible"] is False
    assert section["canonicalBlockers"] == ["non_offset_locator"]
    assert section_span["state"] == "blocked_non_offset_locator"
    assert missing_hash["canonicalEligible"] is False
    assert missing_hash["canonicalBlockers"] == ["missing_source_content_hash"]
    assert missing_hash_span["state"] == "blocked_missing_source_content_hash"
    sidecar = json.loads((output_dir / "research_review_loop_pack.json").read_text(encoding="utf-8"))
    assert sidecar["acceptedClaimIds"] == ["claim-file-valid"]

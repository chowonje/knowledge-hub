# Text Evidence PR #149 Disposition

- status: `ready`
- decision: `abandon_current_pr_before_public_rc`
- mergeRecommended: `False`
- recutRecommendedForV01: `False`
- laterSideTrackAllowed: `True`
- PR mergeable: `CONFLICTING`
- PR mergeStateStatus: `DIRTY`
- missingDependencyRows: `2`
- privatePathLeakRows: `0`
- nextAction: `operator_close_or_abandon_pr149_without_merging_after_approval`

## Dependency Findings

| module | present | disposition |
|---|---:|---|
| `knowledge_hub.papers.complex_qa_structured_evidence_comparison_runner` | `False` | `missing_from_text_evidence_stack` |
| `knowledge_hub.papers.complex_qa_supplied_strict_evidence_grader_baseline_runner` | `False` | `missing_from_text_evidence_stack` |

## PR Files

- `CHANGELOG.md`
- `docs/PROJECT_STATE.md`
- `docs/schemas/paper-complex-qa-real-strict-evidence-availability-bridge-audit.v1.json`
- `knowledge_hub/core/schema_validator.py`
- `knowledge_hub/papers/complex_qa_real_strict_evidence_availability_bridge_audit.py`
- `tests/test_complex_qa_real_strict_evidence_availability_bridge_audit.py`

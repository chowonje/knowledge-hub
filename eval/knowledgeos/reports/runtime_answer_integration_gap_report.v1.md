# Runtime Answer Integration Gap Report

Report-only. No answer path mutation, runtime evidence creation, citation-grade promotion, table/equation parser work, complex QA execution, vault scan, external download, database mutation, or index mutation.

## Baseline

| Item | Value |
| --- | --- |
| Main HEAD | `0386e1861bd3ff11e1089a9827cbf9aa3a110c79` |
| PR stack | `#160` -> `#161` -> `#162` merged |
| Corpus manifest | 300 rows |
| Source coverage | 300/300 available |
| Parsed coverage | 300/300 available |
| Strict evidence coverage | 300/300 public-reviewable `section_text_offset` rows |
| Corpus blockers | 0 |

## Current Runtime Flow

| Step | Current state |
| --- | --- |
| `RAGAnswerRuntime` | owns ask-family runtime request/execution wrapping |
| `EvidenceAssemblyService` | assembles retrieval-backed evidence, citations, answerability, and context budget |
| `answer_payload_builder` | builds `evidencePacketContract` from the assembled `EvidencePacket` |
| `answer_contracts` | keeps only strict-provenance spans for answer citations |
| StrictEvidence JSONL | exists for 300 papers but is not read by answer runtime |
| Runtime binding records | contract exists, but still candidate-only and not answer-visible |

## Gap

The 300 StrictEvidence rows are validated source-backed artifacts, but they are not yet answer-visible. Current answer contracts can consume strict-provenance spans, but no bridge maps `papers_dir/structured_evidence/strict_evidence/{paper_id}.jsonl` into the runtime `EvidencePacket`.

The existing StrictEvidence and runtime-binding contracts explicitly keep runtime and answer integration disabled. Therefore the next tranche must add a bridge or visibility decision layer; it must not silently treat raw StrictEvidence rows as citations.

## Complex QA Split

| Bucket | Count |
| --- | ---: |
| Total questions | 50 |
| Potentially answerable with `section_text_offset` | 16 |
| Actionable section-text rows currently blocked until structured evidence | 13 |
| Blocked by table evidence | 10 |
| Blocked by equation evidence | 8 |
| Blocked by figure caption or appendix lookup | 16 |
| Stable expected-no-answer rows | 17 |

First eval slice: `method_comparison_qa` and `limitation_qa` rows with `blocked_until_structured_evidence`. Keep table, equation, figure-caption, appendix lookup, and expected-no-answer rows blocked.

## Minimal Bridge Decision

Do not feed raw StrictEvidence directly to answers.

The safe insertion point is after `EvidenceAssemblyService` creates `EvidencePacket` and before `answer_payload_builder` builds `evidencePacketContract` / `answerContract`.

Answer-visible eligibility for the first slice:

- `paperId` equals the corpus manifest `sourceIds[0]`
- `sourceContentHash` equals manifest `expectedSourceContentHash`
- StrictEvidence record validates against `knowledge-hub.paper.parsed-artifact-strict-evidence-record.v1`
- `authority.type` is `text_offset`
- `artifactType` is `section`
- `verbatimSubstringSha256` equals `authority.chars.expectedSubstringSha256`
- row is scoped to an allowed method/limitation QA category or explicit paper/query frame
- bridge emits a separate visibility decision instead of mutating StrictEvidence in place

Block immediately on source missing, hash mismatch, wrong `paperId`, invalid schema, missing text-offset authority, table/equation/appendix evidence requirements, expected-no-answer rows, or private path leakage.

## Recommended Next Tranche

Name: `strict-evidence-section-runtime-bridge-dry-run`

Start report-only:

- Build a fixture-backed bridge availability report for section-text StrictEvidence.
- Add negative tests for wrong `paperId`, hash mismatch, missing authority, schema-invalid rows, and unsupported QA categories.
- Only after those gates pass should an answer-visible adapter consume bridge candidates.

Suggested tests to extend:

- `tests/test_answer_contracts_runtime.py`
- `tests/test_complex_qa_structured_evidence_comparison_runner.py`
- a new fixture-backed strict evidence runtime bridge test

## Source References

- `knowledge_hub/ai/evidence_assembly.py:1043-1254`
- `knowledge_hub/ai/answer_payload_builder.py:135-140`
- `knowledge_hub/ai/answer_contracts.py:267-300`
- `knowledge_hub/ai/answer_contracts.py:447-523`
- `knowledge_hub/papers/parsed_artifact_strict_evidence_record_contract.py:133-169`
- `knowledge_hub/papers/strict_evidence_runtime_binding_record_contract.py:126-167`
- `knowledge_hub/papers/strict_evidence_runtime_binding_executor_apply.py:1-8`
- `tests/test_complex_qa_abstain_baseline_runner.py:62-85`
- `tests/test_complex_qa_structured_evidence_comparison_runner.py:66-151`

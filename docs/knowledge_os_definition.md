# KnowledgeOS Product Definition

Last updated: 2026-05-29

Decision record: `docs/adr/2026-05-29-knowledgeos-product-definition.md`

## Final Goal

KnowledgeOS / knowledge-hub is not a paper storage app and not a general chatbot.

The final product is:

> A local-first, evidence-first research knowledge runtime for auditable AI research workflows.

In practical terms, KnowledgeOS collects, parses, structures, retrieves, compares, and answers over AI papers and technical documents only when the supporting evidence can be traced back to local source artifacts.

More concretely, answerable claims must be backed by source hash, source location, and an original excerpt. A fluent model response is not the product boundary; the inspectable local evidence runtime is.

## Final Deliverable

The shipped product is a runtime and evidence substrate, not a GUI-first application.

The final deliverable consists of:

1. Installable `knowledge-hub` runtime
   - `khub` CLI
   - `khub-mcp` MCP server
   - local config and local stores
   - default flow: `discover -> index -> search/ask -> evidence review`
2. Local research corpus
   - paper PDF/text/source records
   - parsed artifacts
   - source hashes
   - corpus manifests
   - explicit missing/hash-mismatch diagnostics
3. Structured evidence layer
   - section and paragraph evidence
   - table evidence
   - equation evidence
   - figure-caption evidence
   - `sourceContentHash`, locator, excerpt, and snippet hash for answerable evidence
4. Answerability runtime
   - answer when evidence is sufficient
   - abstain when evidence is insufficient
   - reject visual hints, fallback chunks, locator-only anchors, memory-unit locators, and paraphrases as answerability evidence
5. Release and eval gate
   - tests and release smoke
   - public hygiene checks
   - paper QA eval
   - no-answer safety
   - private path and vault-leak prevention
   - reproducible report artifacts

## v0.1 Release Candidate

v0.1 is not the full final vision. It is the first stable research-preview slice.

The v0.1 release candidate target is:

> A section/paragraph evidence-first paper QA and compare runtime for a local AI-paper corpus.

v0.1 is complete when:

- `khub discover`, `khub index`, `khub search`, `khub ask`, and `khub trace` work on the default path.
- A 300-500 paper AI research corpus can be managed locally.
- Parsed artifact coverage reaches 80-90% for the priority corpus.
- Section/paragraph evidence chunks are connected to the answerability gate.
- Lack of evidence produces abstain/no-answer rather than unsupported synthesis.
- Visual, table, equation, and figure-caption workflows are clearly marked as labs or limited support unless their gates pass.
- Public CLI/MCP surfaces match the documented Research Preview promise.
- Full smoke, hygiene, and core eval gates are green for the release branch.
- `CHANGELOG.md`, `docs/PROJECT_STATE.md`, and release docs are current.
- The release-candidate branch and PR state are clean enough to review, merge, or intentionally hold.

## Non-Goals For v0.1

The following are part of the long-term vision, not v0.1 default-surface promises:

- GUI-first product surface.
- Obsidian plugin as the canonical product.
- Fully general table numeric QA.
- Fully general equation QA.
- Fully general figure or image understanding.
- Hosted retrieval as the default source of truth.
- Answering when only retrieval hints, fallback chunks, summaries, or locator-only anchors are available.

## Product Principle

The product should prefer a conservative, inspectable no-answer over a fluent answer that cannot be traced to source evidence.

In short:

> 근거가 추적되는 AI 논문 연구 OS.

# ADR: Hold Palantir PDF Translation As Labs Sidecar

Date: 2026-06-08

## Status

Accepted as a product-direction constraint.

## Context

The project tested a Palantir AIP Logic sidecar that can accept public text chunks, return Korean text, and render a derivative PDF. That proved external-call wiring and artifact generation, but it did not prove a product-ready workflow for full paper PDF translation.

The current user-value gap is material:

- full PDF coverage is unproven
- source PDF layout preservation is a separate parsing and rendering problem
- tables, figures, equations, and page ordering need dedicated extraction and QA
- token, credential, and external-call handling adds operational burden
- generated PDFs can look product-ready while hiding weak coverage or poor layout fidelity

The default product direction remains local-first, policy-first, and retrieval-assistant-first: `discover -> index -> search/ask -> evidence review`.

## Decision

Palantir PDF translation stays `labs` and is held from default/runtime promotion.

Palantir may be used only as an explicit external sidecar:

- Knowledge Hub owns local source intake, parsing, chunking, provenance, policy gates, and import review.
- Palantir AIP may translate or process selected low-risk chunks only after explicit operator confirmation and classification checks.
- Palantir output must return as an inspectable derivative artifact before any import or indexing.
- Palantir output must not become citation-grade evidence, default answer context, or canonical source truth without a separate promotion decision and eval gate.

The preferred first-class workflow for Korean reading remains a source-linked reading pack:

`local source -> parse/extract -> page/section/chunk anchors -> Korean Markdown/HTML reading pack -> optional derivative PDF -> index/review`

Layout-faithful translated PDF cloning is not a default product target.

## Invariants

- Original source files remain the source of truth.
- Korean reading artifacts are derivative and must carry source anchors or hashes.
- External calls require explicit opt-in, classification, and secret-safe logging.
- Labs outputs are inspectable before import and are not silently indexed.
- The default product promise stays narrower than labs: retrieval, grounded answering, and evidence review.

## Consequences

- Product work should stop spending tranches on Palantir PDF rendering quality unless it removes a named labs blocker.
- Useful next work should prove whether source-linked Korean reading packs help actual reading and recall.
- Palantir can still be valuable for organization-specific external workflows, but it is not the core Knowledge Hub engine.
- PDF rendering remains a secondary output format, not the primary success criterion.

## Follow-Ups

1. Define a one-document reading-pack acceptance gate: coverage, anchors, glossary, missing-content warnings, and user readability.
2. Keep the Palantir sidecar profile and credentialed smoke tests labs-only.
3. Add import checks only after a real reading-pack pilot proves user value.

# Public Preview Release Draft

## Repo description

`Knowledge Hub is a local-first evidence-contract RAG runtime for papers, web pages, vault notes, and grounded evidence review. Research Preview.`

## Status line

`Status: Research Preview — the supported default path is add -> index -> search/ask -> evidence review. APIs, quality bars, and experimental surfaces may change without notice.`

## Short release note

- Knowledge Hub is being opened as a **Research Preview**, not a stable release.
- The supported default path is intentionally narrow: `add -> index -> search/ask -> evidence review`.
- The public green signal is limited to a narrow smoke gate / approval slice rather than full-repo green.
- `khub provider` is the supported setup surface for local/API/Codex-MCP/custom OpenAI-compatible model choices; API keys should be configured through environment-variable references.
- Experimental and operator-facing surfaces remain outside the default public product contract.
- Source quality is still uneven; `paper` and `project` are stronger than at least one known `vault` compare path.

## Known Limits copy

Use this wording when a short limitations section is needed:

- We do not claim full-repo green.
- Source quality is uneven across `paper`, `project`, and `vault`.
- At least one known `vault` compare path can still degrade to `0 source`.
- Some Python / TypeScript boundary areas and operator surfaces remain mid-migration.
- Production use is not recommended without internal validation.

## Allowed wording

- `Research Preview`
- `Public Prototype`
- `Experimental`
- `Subject to change without notice`
- `Narrow smoke gate`

## Avoid wording

- `stable`
- `production-ready`
- `GA`
- `1.0`
- `fully tested`
- `enterprise-ready`
- `secure by default`

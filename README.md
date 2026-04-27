# Knowledge Hub

Local-first evidence-contract RAG for personal research.

Status: **Research Preview**. The supported public path is:

`add -> index -> search/ask -> evidence review`

Knowledge Hub helps a researcher or builder collect papers, web pages, and
vault notes into local stores, index them, ask grounded questions, and inspect
the evidence trail before reusing an answer.

It is not a stable release or a production automation platform. APIs, quality
bars, and experimental surfaces may change without notice.

## What Is Supported Today

The public promise is intentionally narrow:

1. Add local research sources.
2. Build or refresh the retrieval index.
3. Search or ask against the indexed corpus.
4. Review the cited evidence and source trace.

The core contract is evidence-first. When Knowledge Hub can ground an answer, it
should expose enough source information to inspect why. When evidence is weak or
missing, it should return an explicit lack-of-evidence outcome instead of
pretending certainty.

## Quick Start

From a repository checkout:

```bash
pip install -e ".[ollama]"
khub provider setup --profile local
khub doctor
khub add "large language model agent" --type paper -n 3
khub index
khub search "attention mechanism"
khub ask "Transformer의 핵심 아이디어는?"
```

`khub provider setup --profile local` keeps the first-run path off hosted
providers. If `khub doctor` reports that the local model runtime is unavailable,
start Ollama and pull the recommended local models:

```bash
ollama serve
ollama pull qwen3:14b
ollama pull nomic-embed-text
khub doctor
```

Hosted providers are optional. If you use one, configure API keys through
environment variables rather than writing raw secrets into config files.

## Public Path

### 1. Add

Use `khub add` as the public intake facade for papers and web sources:

```bash
khub add "large language model agent" --type paper -n 3
khub add "https://example.com/guide" --topic "rag"
khub add "https://youtu.be/<video-id>" --topic "agents"
```

Success means at least one source item lands in canonical local storage.

### 2. Index

Build the retrieval-facing document/vector surface:

```bash
khub index
```

Success means the retrieval surface is non-empty and readable by the runtime.

### 3. Search Or Ask

Use search when you want ranked evidence. Use ask when you want a grounded answer:

```bash
khub search "attention mechanism"
khub ask "What problem does retrieval-augmented generation solve?"
```

Success means the runtime returns either grounded results or an explicit
no-evidence response.

### 4. Review Evidence

Treat generated answers as research assistance, not final authority. Inspect the
returned source ids, snippets, citations, and follow-up reading surfaces before
copying claims into notes, papers, or decisions.

For paper-focused review, use `khub paper summary`, `khub paper evidence`, and
`khub paper related` after ingestion.

## Provider Setup

The local-first path uses Ollama for both generation and embeddings. You can
also choose API-backed or OpenAI-compatible providers:

```bash
khub provider recommend
khub provider setup --profile local
khub provider setup --profile balanced
khub provider add deepseek --from-service deepseek --use-for answer
khub provider key deepseek --env DEEPSEEK_API_KEY
```

Embeddings can send large corpus text to the selected provider, so local
embeddings are the recommended default for private notes or unpublished work.

## Known Limits

- This is a **Research Preview**, not a stable release.
- The current green signal is a narrow smoke gate / approval slice, not a
  full-repository green claim.
- Source-family quality is uneven; paper and project paths are stronger than
  some vault comparison paths.
- Local model quality, latency, and memory use vary by machine.
- Experimental and operator-facing surfaces remain outside the default public
  product contract.
- Some Python / TypeScript boundary areas are still in transition.
- Production use is not recommended without internal validation.

## More Documentation

- [Full guide](docs/full-guide.md) - preserved extended setup and command guide.
- [CLI command guide](docs/guides/cli-commands.md) - broader command inventory.
- [Architecture](docs/ARCHITECTURE.md) - repository boundaries and runtime contracts.
- [Public release checklist](docs/guides/public-release-checklist.md) - release gates and wording.

## License

MIT

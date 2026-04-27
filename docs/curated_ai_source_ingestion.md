# Curated AI Source Ingestion

## Goal
Use curated, high-signal AI sources instead of broad discovery-first crawling. This reduces junk and improves ontology density and note quality.

## Seed Files
- `data/curation/ai_watchlists/continuous_sources.yaml`
- `data/curation/ai_watchlists/reference_sources.yaml`
- `data/curation/ai_watchlists/hub_sources.yaml`
- `data/curation/ai_watchlists/priority_documents.yaml`
- `data/curation/ai_watchlists/priority_documents.txt`

## How To Use
### Immediate ingest for high-signal single documents
```bash
khub crawl run \
  --url-file data/curation/ai_watchlists/priority_documents.txt \
  --topic "ai-core" \
  --source web \
  --profile safe \
  --source-policy fixed \
  --index --extract-concepts --json
```

### Continuous sources
Run recurring crawls for the official blogs in `continuous_sources.yaml`. These are discovery-friendly and should be materialized selectively.

### Specialist reference sources
Run static or slow-moving concept references from `reference_sources.yaml` when you want background context for concept notes without mixing them into the continuous latest feed.

```bash
khub crawl reference-sync \
  --watchlist-file data/curation/ai_watchlists/reference_sources.yaml \
  --topic "concept-reference" \
  --source web \
  --profile safe \
  --source-policy fixed \
  --index --extract-concepts --json
```

Operating rule:
- Prefer paper/web evidence first.
- Use specialist references as background context and terminology support.
- Keep Wikipedia as a manual fallback rather than a default source of truth.

### Hub sources
Use hub pages for discovery and follow-up extraction. Do not materialize them directly as final notes unless the page itself is high-value.

## Raw Source Storage
Original fetched source artifacts are stored on the external drive under:

- raw: `<pipeline-storage-root>/raw/{source}/{yyyy}/{mm}/{dd}/{record_id}/`
- normalized: `<pipeline-storage-root>/normalized/{source}/{yyyy}/{mm}/{dd}/{record_id}.json`
- indexed: `<pipeline-storage-root>/indexed/{source}/{yyyy}/{mm}/{dd}/{record_id}.json`

### Raw folder contents
Typical raw folder contents:
- `content.raw`
- `metadata.json`

Example:
- `<pipeline-storage-root>/raw/web/2026/03/05/<record_id>/content.raw`
- `<pipeline-storage-root>/raw/web/2026/03/05/<record_id>/metadata.json`

## Truth Sources
- raw source of truth: configured local storage root
- state/ontology source of truth: `~/.khub/knowledge.db`
- final human-readable notes: Obsidian vault under `AI/AI_Papers`

## Operating Rule
- Discover with curated seeds.
- Store raw locally.
- Extract ontology conservatively.
- Materialize only high-signal records into Obsidian.

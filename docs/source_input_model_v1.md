# Source Input Model v1

## Purpose
Workstream 1 redefines the input plane so external documents arrive as standardized, trustworthy records instead of ad-hoc crawl output.

## Discovery Object
`DiscoveredSourceItem`

Fields:
- `identity.source_vendor`
- `identity.source_channel`
- `identity.source_channel_type`
- `identity.source_item_id`
- `source_name`
- `source_type`
- `url`
- `canonical_url`
- `title_hint`
- `published_at`
- `author`
- `tags[]`
- `provenance.method`
- `provenance.origin_url`
- `provenance.entry_ref`
- `provenance.discovered_at`
- `provenance.rank`
- `metadata.metadata_precedence`

## Normalized Payload v2
Stored under `normalized/{source}/.../{record_id}.json`.

Required keys:
- `schema = knowledge-hub.normalized.web-record.v2`
- `record_id`
- `source`
- `url`
- `canonical_url`
- `domain`
- `fetched_at`
- `content_sha256`
- `title`
- `title_hint`
- `description`
- `author`
- `published_at`
- `source_name`
- `source_type`
- `source_vendor`
- `source_channel`
- `source_channel_type`
- `source_item_id`
- `tags`
- `freshness_days`
- `discovery`
- `metadata_quality`
- `content_text`
- `quality_score`
- `crawl_engine`
- `job_id`
- `run_id`

## Metadata Quality v1
`metadata_quality`
- `completeness`: weighted completeness score
- `consistency_flags`: missing or mismatched metadata indicators
- `precedence_used`: actual resolution source per field

Truth precedence default:
1. source feed metadata
2. page embedded metadata
3. extractor-specific HTML parse
4. generic crawler inference

## Downstream Consumer Contract
### Claim extractor
Expects:
- `canonical_url`
- `source_channel`
- `source_item_id`
- `published_at`
- `author`
- `content_text`

### Feature layer
Expects:
- `source_vendor`
- `source_channel`
- `source_type`
- `published_at`
- `freshness_days`
- `tags`
- `metadata_quality`

### Materializer / note generation
Expects:
- `title_hint`
- `canonical_url`
- `published_at`
- `author`
- `source_name`

## Dedupe Priority
Discovery-level dedupe key order:
1. `source_item_id`
2. `canonical_url_normalized`
3. `normalized_url`
4. `lightweight_content_hash`
5. `source_channel + normalized_title_hint + published_at`

Ingest-level dedupe remains:
- `canonical_url_hash`
- `content_sha256`

# Paper Corpus Quality Audit

This document is a read-only audit checklist and command pack for the local paper corpus. It is not a remediation runbook. Do not use it to materialize, rebuild, backfill, repair, or apply changes.

The audit separates route-ready regression risk from answer and public-reading quality backlog. A corpus row can be route-ready while still needing reading-quality remediation.

## Audit Purpose

### Route-Ready Regression Guard

Use this lane to check whether `source=paper` can still route, select cards, preserve compare targets, and return grounded evidence without no-result or hard-gate regressions.

Route-ready checks focus on:

- paper metadata existence for target ids
- non-stale `paper_cards_v2`
- evidence anchor presence
- compare target pair preservation
- no `no_result` or ask-v2 hard gate on v0 paper eval rows

### Answer/Public-Reading Quality Backlog Detection

Use this lane to find corpus quality debt that affects answer quality, public reading surfaces, or future remediation priority.

Backlog checks focus on:

- fallback summaries
- weak or non-ok `paper_memory_cards`
- weak or unsupported `claim_cards_v1`
- duplicate or LaTeX-heavy anchors
- missing or blocked `paper_section_cards_v1`
- metadata hygiene gaps

## Risk Levels

| Level | Meaning | Blocks route-ready? |
| --- | --- | --- |
| P0 route-ready blocker | Missing target paper/card/anchors, stale selected card, no-result, ask-v2 hard gate, or compare target drift | Yes |
| P1 answer-acceptance blocker | Live answer path has generation fallback, read timeout, p95 latency breach, missing citation trace, or unsupported final claims | Blocks answer acceptance, not route-ready by itself |
| P2 corpus quality backlog | Non-ok cards, weak ClaimCards, fallback summaries, LaTeX/duplicate anchors, or missing section cards when fallback evidence still routes | No |
| P3 metadata hygiene | Missing authors/year/field/source paths where route and evidence still work | No |

## Read-Only Command Pack

Set the active DB explicitly before running the checks:

```bash
DB=/Users/won/.khub/knowledge.db
```

All SQLite commands below use `sqlite3 -readonly`. They should not create tables, write artifacts, or mutate corpus state.

### Paper Metadata Completeness

```bash
sqlite3 -readonly -header -column "$DB" "
SELECT COUNT(*) AS total,
       SUM(CASE WHEN COALESCE(title,'')='' OR trim(COALESCE(title,''))='' THEN 1 ELSE 0 END) AS missing_title,
       SUM(CASE WHEN trim(COALESCE(authors,''))='' THEN 1 ELSE 0 END) AS missing_authors,
       SUM(CASE WHEN COALESCE(year,0)=0 THEN 1 ELSE 0 END) AS missing_year,
       SUM(CASE WHEN trim(COALESCE(field,''))='' THEN 1 ELSE 0 END) AS missing_field,
       SUM(CASE WHEN trim(COALESCE(pdf_path,''))='' THEN 1 ELSE 0 END) AS missing_pdf_path,
       SUM(CASE WHEN trim(COALESCE(text_path,''))='' THEN 1 ELSE 0 END) AS missing_text_path,
       SUM(CASE WHEN COALESCE(indexed,0)=0 THEN 1 ELSE 0 END) AS not_indexed
FROM papers;"
```

### paper_cards_v2 Anchor, Stale, and Quality

```bash
sqlite3 -readonly -header -column "$DB" "
WITH card_stats AS (
  SELECT pc.*,
         (SELECT COUNT(*) FROM evidence_anchors_v2 a WHERE a.card_id=pc.card_id) AS anchor_count,
         (SELECT COUNT(DISTINCT COALESCE(snippet_hash,'')) FROM evidence_anchors_v2 a WHERE a.card_id=pc.card_id) AS distinct_snippets,
         (SELECT COUNT(*) FROM evidence_anchors_v2 a
          WHERE a.card_id=pc.card_id
            AND (COALESCE(a.excerpt,'') LIKE '%\begin{%'
              OR COALESCE(a.excerpt,'') LIKE '%\cref{%'
              OR COALESCE(a.excerpt,'') LIKE '%\section{%'
              OR COALESCE(a.excerpt,'') LIKE '%\documentclass%')) AS latex_anchor_count
  FROM paper_cards_v2 pc
)
SELECT COUNT(*) AS total_cards,
       SUM(CASE WHEN COALESCE(stale,0)!=0 THEN 1 ELSE 0 END) AS stale_cards,
       SUM(CASE WHEN COALESCE(quality_flag,'')!='ok' THEN 1 ELSE 0 END) AS non_ok_cards,
       SUM(CASE WHEN anchor_count=0 THEN 1 ELSE 0 END) AS no_anchor_cards,
       SUM(CASE WHEN anchor_count>1 AND distinct_snippets=1 THEN 1 ELSE 0 END) AS duplicate_anchor_cards,
       SUM(CASE WHEN latex_anchor_count>0 THEN 1 ELSE 0 END) AS latex_anchor_cards,
       SUM(CASE WHEN trim(COALESCE(method_core,''))='' THEN 1 ELSE 0 END) AS empty_method,
       SUM(CASE WHEN trim(COALESCE(result_core,''))='' THEN 1 ELSE 0 END) AS empty_result,
       SUM(CASE WHEN trim(COALESCE(limitations_core,''))='' THEN 1 ELSE 0 END) AS empty_limitations
FROM card_stats;"
```

Detailed card outliers:

```bash
sqlite3 -readonly -header -column "$DB" "
WITH card_stats AS (
  SELECT COALESCE(pc.paper_id,'') AS paper_id,
         COALESCE(pc.title,'') AS title,
         COALESCE(pc.quality_flag,'missing') AS quality_flag,
         COALESCE(pc.stale,0) AS stale,
         (SELECT COUNT(*) FROM evidence_anchors_v2 a WHERE a.card_id=pc.card_id) AS anchor_count,
         (SELECT COUNT(DISTINCT COALESCE(snippet_hash,'')) FROM evidence_anchors_v2 a WHERE a.card_id=pc.card_id) AS distinct_snippets,
         (SELECT COUNT(*) FROM evidence_anchors_v2 a
          WHERE a.card_id=pc.card_id
            AND (COALESCE(a.excerpt,'') LIKE '%\begin{%'
              OR COALESCE(a.excerpt,'') LIKE '%\cref{%'
              OR COALESCE(a.excerpt,'') LIKE '%\section{%'
              OR COALESCE(a.excerpt,'') LIKE '%\documentclass%')) AS latex_anchor_count
  FROM paper_cards_v2 pc
)
SELECT paper_id, title, quality_flag, stale, anchor_count, distinct_snippets, latex_anchor_count
FROM card_stats
WHERE stale!=0
   OR quality_flag!='ok'
   OR anchor_count=0
   OR latex_anchor_count>0
   OR (anchor_count>1 AND distinct_snippets=1)
ORDER BY stale DESC, anchor_count ASC, latex_anchor_count DESC, paper_id
LIMIT 50;"
```

### paper_memory_cards Quality and Stale

```bash
sqlite3 -readonly -header -column "$DB" "
SELECT COALESCE(quality_flag,'missing') AS quality_flag,
       COALESCE(stale,0) AS stale,
       COUNT(*) AS n
FROM paper_memory_cards
GROUP BY COALESCE(quality_flag,'missing'), COALESCE(stale,0)
ORDER BY n DESC;"
```

Weak memory-card slots:

```bash
sqlite3 -readonly -header -column "$DB" "
SELECT paper_id,
       title,
       COALESCE(quality_flag,'missing') AS quality_flag,
       COALESCE(stale,0) AS stale,
       CASE WHEN trim(COALESCE(problem_context,''))='' THEN 1 ELSE 0 END AS empty_problem,
       CASE WHEN trim(COALESCE(method_core,''))='' THEN 1 ELSE 0 END AS empty_method,
       CASE WHEN trim(COALESCE(evidence_core,''))='' THEN 1 ELSE 0 END AS empty_evidence,
       CASE WHEN trim(COALESCE(limitations,''))='' THEN 1 ELSE 0 END AS empty_limitations
FROM paper_memory_cards
WHERE COALESCE(stale,0)!=0
   OR COALESCE(quality_flag,'')!='ok'
   OR trim(COALESCE(problem_context,''))=''
   OR trim(COALESCE(method_core,''))=''
   OR trim(COALESCE(evidence_core,''))=''
ORDER BY stale DESC, quality_flag, paper_id
LIMIT 50;"
```

### claim_cards_v1 Weak or Unsupported Status

```bash
sqlite3 -readonly -header -column "$DB" "
SELECT COALESCE(quality_flag,'missing') AS quality_flag,
       COALESCE(trust_level,'missing') AS trust_level,
       COALESCE(evidence_strength,'missing') AS evidence_strength,
       COALESCE(stale,0) AS stale,
       COUNT(*) AS n
FROM claim_cards_v1
WHERE COALESCE(source_kind,'')='paper'
GROUP BY COALESCE(quality_flag,'missing'),
         COALESCE(trust_level,'missing'),
         COALESCE(evidence_strength,'missing'),
         COALESCE(stale,0)
ORDER BY n DESC;"
```

Per-paper weak ClaimCard rollup:

```bash
sqlite3 -readonly -header -column "$DB" "
WITH claim_rollup AS (
  SELECT COALESCE(paper_id,'') AS paper_id,
         COUNT(*) AS total_claim_cards,
         SUM(CASE WHEN COALESCE(quality_flag,'')='ok' THEN 1 ELSE 0 END) AS ok_claim_cards,
         SUM(CASE WHEN COALESCE(quality_flag,'')!='ok' THEN 1 ELSE 0 END) AS weak_claim_cards,
         SUM(CASE WHEN COALESCE(evidence_strength,'')='weak' THEN 1 ELSE 0 END) AS weak_evidence_claim_cards,
         SUM(CASE WHEN COALESCE(claim_text,'') LIKE '%\begin{%'
                   OR COALESCE(claim_text,'') LIKE '%\cref{%'
                   OR COALESCE(claim_text,'') LIKE '%\section{%'
                   OR COALESCE(claim_text,'') LIKE '%\documentclass%'
                  THEN 1 ELSE 0 END) AS latex_claim_cards
  FROM claim_cards_v1
  WHERE COALESCE(source_kind,'')='paper'
  GROUP BY COALESCE(paper_id,'')
)
SELECT paper_id,
       total_claim_cards,
       ok_claim_cards,
       weak_claim_cards,
       weak_evidence_claim_cards,
       latex_claim_cards
FROM claim_rollup
WHERE COALESCE(paper_id,'')!=''
  AND (ok_claim_cards=0 OR weak_claim_cards>0 OR latex_claim_cards>0)
ORDER BY ok_claim_cards ASC, latex_claim_cards DESC, weak_claim_cards DESC, paper_id
LIMIT 50;"
```

### document_memory_units and paper_section_cards_v1 Coverage

```bash
sqlite3 -readonly -header -column "$DB" "
SELECT COUNT(*) AS document_units,
       COUNT(DISTINCT document_id) AS documents,
       SUM(CASE WHEN COALESCE(stale,0)!=0 THEN 1 ELSE 0 END) AS stale_units
FROM document_memory_units
WHERE COALESCE(source_type,'')='paper' OR COALESCE(document_id,'') LIKE 'paper:%';

SELECT COUNT(*) AS section_cards,
       COUNT(DISTINCT paper_id) AS papers_with_section_cards,
       SUM(CASE WHEN COALESCE(stale,0)!=0 THEN 1 ELSE 0 END) AS stale_section_cards
FROM paper_section_cards_v1;

WITH roles AS (
  SELECT paper_id,
         GROUP_CONCAT(DISTINCT COALESCE(role,'')) AS roles
  FROM paper_section_cards_v1
  WHERE COALESCE(stale,0)=0
  GROUP BY paper_id
)
SELECT COUNT(*) AS papers_with_fresh_section_cards,
       SUM(CASE WHEN instr(','||roles||',', ',problem,')>0 THEN 1 ELSE 0 END) AS has_problem,
       SUM(CASE WHEN instr(','||roles||',', ',method,')>0 THEN 1 ELSE 0 END) AS has_method,
       SUM(CASE WHEN instr(','||roles||',', ',results,')>0 THEN 1 ELSE 0 END) AS has_results,
       SUM(CASE WHEN instr(','||roles||',', ',limitations,')>0 THEN 1 ELSE 0 END) AS has_limitations
FROM roles;"
```

Per-paper section coverage:

```bash
sqlite3 -readonly -header -column "$DB" "
WITH section_rollup AS (
  SELECT p.arxiv_id AS paper_id,
         p.title,
         (SELECT COUNT(*) FROM document_memory_units d
          WHERE d.document_id='paper:'||p.arxiv_id AND COALESCE(d.stale,0)=0) AS fresh_document_units,
         (SELECT COUNT(*) FROM paper_section_cards_v1 s
          WHERE s.paper_id=p.arxiv_id AND COALESCE(s.stale,0)=0) AS fresh_section_cards,
         (SELECT GROUP_CONCAT(DISTINCT COALESCE(s.role,''))
          FROM paper_section_cards_v1 s
          WHERE s.paper_id=p.arxiv_id AND COALESCE(s.stale,0)=0) AS section_roles
  FROM papers p
)
SELECT paper_id, title, fresh_document_units, fresh_section_cards, COALESCE(section_roles,'') AS section_roles
FROM section_rollup
WHERE fresh_document_units>0 AND fresh_section_cards=0
ORDER BY paper_id
LIMIT 50;"
```

### Fallback Summary Detection

```bash
find /Users/won/.khub/papers/summaries -name summary.json -print0 |
xargs -0 jq -r '
  select(.fallbackUsed == true) |
  [.paperId, .paperTitle, .parserUsed, .llmRoute, ((.warnings // []) | join("|"))] |
  @tsv
'
```

### Mamba `2312.00752` Specific Check

```bash
sqlite3 -readonly -header -column "$DB" "
SELECT p.arxiv_id,
       p.title,
       CASE WHEN trim(COALESCE(p.authors,''))!='' THEN 1 ELSE 0 END AS has_authors,
       CASE WHEN COALESCE(p.year,0)!=0 THEN 1 ELSE 0 END AS has_year,
       CASE WHEN trim(COALESCE(p.field,''))!='' THEN 1 ELSE 0 END AS has_field,
       CASE WHEN trim(COALESCE(p.pdf_path,''))!='' THEN 1 ELSE 0 END AS has_pdf_path,
       CASE WHEN trim(COALESCE(p.text_path,''))!='' THEN 1 ELSE 0 END AS has_text_path,
       COALESCE(p.indexed,0) AS is_indexed,
       COALESCE(pm.quality_flag,'missing') AS memory_quality,
       COALESCE(pm.stale,0) AS memory_stale,
       COALESCE(pc.quality_flag,'missing') AS card_quality,
       COALESCE(pc.stale,0) AS card_stale,
       (SELECT COUNT(*) FROM evidence_anchors_v2 a WHERE a.paper_id=p.arxiv_id) AS anchors,
       (SELECT COUNT(DISTINCT COALESCE(snippet_hash,'')) FROM evidence_anchors_v2 a WHERE a.paper_id=p.arxiv_id) AS distinct_anchor_snippets,
       (SELECT COUNT(*) FROM evidence_anchors_v2 a
        WHERE a.paper_id=p.arxiv_id
          AND (COALESCE(a.excerpt,'') LIKE '%\begin{%'
            OR COALESCE(a.excerpt,'') LIKE '%\cref{%'
            OR COALESCE(a.excerpt,'') LIKE '%\section{%'
            OR COALESCE(a.excerpt,'') LIKE '%\documentclass%')) AS latex_anchors,
       (SELECT COUNT(*) FROM claim_cards_v1 cc WHERE cc.paper_id=p.arxiv_id AND COALESCE(cc.quality_flag,'')='ok') AS ok_claim_cards,
       (SELECT COUNT(*) FROM claim_cards_v1 cc WHERE cc.paper_id=p.arxiv_id AND COALESCE(cc.quality_flag,'')!='ok') AS weak_claim_cards,
       (SELECT COUNT(*) FROM document_memory_units d WHERE d.document_id='paper:'||p.arxiv_id AND COALESCE(d.stale,0)=0) AS doc_units,
       (SELECT COUNT(*) FROM paper_section_cards_v1 s WHERE s.paper_id=p.arxiv_id AND COALESCE(s.stale,0)=0) AS section_cards
FROM papers p
LEFT JOIN paper_memory_cards pm ON pm.paper_id=p.arxiv_id
LEFT JOIN paper_cards_v2 pc ON pc.paper_id=p.arxiv_id
WHERE p.arxiv_id='2312.00752';"
```

```bash
jq '{paperId,paperTitle,parserUsed,fallbackUsed,llmRoute,warnings,summary}' \
/Users/won/.khub/papers/summaries/2312.00752/summary.json
```

## Blocker Decision Table

| Signal | Classification | Action |
| --- | --- | --- |
| Target paper missing from `papers` | P0 route-ready blocker | Stop route-ready signoff |
| Target paper has no `paper_cards_v2` row | P0 route-ready blocker | Stop route-ready signoff |
| Target `paper_cards_v2.stale != 0` | P0 route-ready blocker | Stop route-ready signoff |
| Target card has `anchor_count=0` | P0 route-ready blocker | Stop route-ready signoff |
| Eval row has `no_result=1` | P0 route-ready blocker | Stop route-ready signoff |
| Eval row has `ask_v2_hard_gate=1` | P0 route-ready blocker | Stop route-ready signoff |
| Compare row does not preserve resolved pair | P0 route-ready blocker | Stop route-ready signoff |
| Live smoke has generation fallback or read timeout | P1 answer-acceptance blocker | Keep route-ready separate; block answer acceptance |
| Live smoke p95 latency exceeds threshold | P1 answer-acceptance blocker | Keep route-ready separate; block answer acceptance |
| Citation trace missing in live answer | P1 answer-acceptance blocker | Keep route-ready separate; block answer acceptance |
| `paper_memory_cards.quality_flag != 'ok'` | P2 corpus quality backlog | Queue remediation |
| `paper_cards_v2.quality_flag != 'ok'` but anchors exist and route works | P2 corpus quality backlog | Queue remediation |
| ClaimCards are all weak or `needs_review` | P2 corpus quality backlog | Queue remediation |
| Summary `fallbackUsed=true` | P2 corpus quality backlog | Queue remediation |
| Anchors duplicated or LaTeX-heavy | P2 corpus quality backlog | Queue remediation |
| `paper_section_cards_v1` missing, while card/anchor route works | P2 corpus quality backlog | Queue remediation |
| Missing authors/year/field | P3 metadata hygiene | Track as cleanup |

## Mamba `2312.00752` Current Risk Note

Mamba `2312.00752` is currently a route-ready non-blocker and an answer/public-reading backlog item.

Read-only observations from the v0 paper route-ready work:

- Local metadata, source artifacts, `paper_cards_v2`, paper memory, and vector/lexical indexing exist.
- `paper_cards_v2` and `paper_memory_cards` were non-stale in the route-ready check.
- `paper_section_cards_v1` coverage was `0` for Mamba because section-card materialization was blocked by `problem_only_sections`.
- The structured summary artifact had `fallbackUsed=true`.
- Mamba ClaimCards were weak / `needs_review` rather than accepted strong claims.
- Evidence anchors were duplicated and LaTeX-heavy.

This combination must not be used as a v0 route-ready blocker while the compare route still preserves `1706.03762 | 2312.00752`, selects non-stale cards, and avoids no-result / ask-v2 hard gate. It should be tracked as P2 answer/public-reading backlog.

## Forbidden Commands

During this read-only audit, do not run commands that can write DB rows, create artifacts, rebuild cards, materialize summaries, or apply repairs.

Forbidden examples:

```bash
khub labs section-cards build ...
scripts/rebuild_memory_stores.py ...
khub paper canon-quality-audit ...
khub ... --apply
khub ... build
```

Also avoid materialization, backfill, rebuild, repair, apply, or artifact-writing audit options such as `--artifact-dir`.

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return [{str(k): str(v or '') for k, v in row.items()} for row in csv.DictReader(handle)]


def _parse_doc_ids(raw: str) -> list[str]:
    if not raw:
        return []
    normalized = raw.replace('|', ';').replace(',', ';')
    return [item.strip() for item in normalized.split(';') if item.strip()]


def _fetch_search_payload(query: str, *, cache_dir: Path | None, top_k: int) -> dict[str, Any]:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        slug = f"q_{abs(hash(query)) & 0xffffffff:08x}.json"
        cache_path = cache_dir / slug
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding='utf-8'))
    out = subprocess.check_output(['khub', 'search', query, '--json'], text=True)
    payload = json.loads(out)
    payload['results'] = list(payload.get('results') or [])[:top_k]
    if cache_path is not None:
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def _load_search_from_cache(query: str, cache_dir: Path) -> dict[str, Any]:
    for path in cache_dir.glob('*.json'):
        payload = json.loads(path.read_text(encoding='utf-8'))
        if str(payload.get('query') or '').strip() == query:
            return payload
    raise FileNotFoundError(query)


def _collect_global_doc_frequency(pairwise_rows: list[dict[str, str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in pairwise_rows:
        for key in ('baseline_top1_doc_id', 'candidate_top1_doc_id', 'baseline_top3_doc_ids', 'candidate_top3_doc_ids'):
            for doc_id in _parse_doc_ids(row.get(key, '')):
                counter[doc_id] += 1
    return counter


def _build_pairwise_index(pairwise_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get('query') or '').strip(): row for row in pairwise_rows if str(row.get('query') or '').strip()}


def _expected_source_matches(source_type: str, expected_source: str) -> bool:
    return bool(expected_source and expected_source not in {'all', 'mixed'} and source_type == expected_source)


def _candidate_map(query_row: dict[str, str], pairwise_row: dict[str, str] | None, search_payload: dict[str, Any], global_freq: Counter[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    candidates: dict[str, dict[str, Any]] = {}
    evidence: list[str] = []
    expected_source = str(query_row.get('expected_primary_source') or '').strip()

    def ensure(doc_id: str) -> dict[str, Any]:
        if doc_id not in candidates:
            candidates[doc_id] = {
                'doc_id': doc_id,
                'title': '',
                'source_type': '',
                'runtime_rank': None,
                'baseline_rank': None,
                'candidate_rank': None,
                'freq': global_freq.get(doc_id, 0),
            }
        return candidates[doc_id]

    for rank, item in enumerate(list(search_payload.get('results') or []), start=1):
        doc_id = str(item.get('documentId') or '').strip()
        if not doc_id:
            continue
        rec = ensure(doc_id)
        rec['title'] = str(item.get('title') or rec['title'])
        rec['source_type'] = str(item.get('sourceType') or rec['source_type'])
        rec['runtime_rank'] = rank
    if search_payload.get('results'):
        evidence.append('current-runtime-topk')

    if pairwise_row:
        b_top1 = str(pairwise_row.get('baseline_top1_doc_id') or '').strip()
        if b_top1:
            rec = ensure(b_top1)
            rec['baseline_rank'] = 1
            rec['title'] = rec['title'] or str(pairwise_row.get('baseline_top1_title') or '')
            rec['source_type'] = rec['source_type'] or str(pairwise_row.get('baseline_top1_source_type') or '')
        for rank, doc_id in enumerate(_parse_doc_ids(pairwise_row.get('baseline_top3_doc_ids', '')), start=1):
            rec = ensure(doc_id)
            rec['baseline_rank'] = rank if rec['baseline_rank'] is None else min(rec['baseline_rank'], rank)

        c_top1 = str(pairwise_row.get('candidate_top1_doc_id') or '').strip()
        if c_top1:
            rec = ensure(c_top1)
            rec['candidate_rank'] = 1
            rec['title'] = rec['title'] or str(pairwise_row.get('candidate_top1_title') or '')
            rec['source_type'] = rec['source_type'] or str(pairwise_row.get('candidate_top1_source_type') or '')
        for rank, doc_id in enumerate(_parse_doc_ids(pairwise_row.get('candidate_top3_doc_ids', '')), start=1):
            rec = ensure(doc_id)
            rec['candidate_rank'] = rank if rec['candidate_rank'] is None else min(rec['candidate_rank'], rank)
        evidence.append('pairwise-topk')

    for rec in candidates.values():
        score = 0.0
        if rec['runtime_rank'] is not None:
            score += max(0.0, 6.0 - float(rec['runtime_rank']))
        if rec['baseline_rank'] is not None:
            score += 3.0 if rec['baseline_rank'] == 1 else 1.5
        if rec['candidate_rank'] is not None:
            score += 3.0 if rec['candidate_rank'] == 1 else 1.5
        if rec['baseline_rank'] is not None and rec['candidate_rank'] is not None:
            score += 2.0
        if _expected_source_matches(str(rec['source_type'] or ''), expected_source):
            score += 2.0
        rec['gold_seed_score'] = round(score, 3)

        hn = 0.0
        if expected_source and expected_source not in {'all', 'mixed'} and rec['source_type'] and rec['source_type'] != expected_source:
            hn += 2.0
        if rec['freq'] >= 3:
            hn += 1.5
        if rec['runtime_rank'] == 1 and not _expected_source_matches(str(rec['source_type'] or ''), expected_source):
            hn += 1.0
        if rec['baseline_rank'] == 1 or rec['candidate_rank'] == 1:
            hn += 0.5
        rec['hard_negative_seed_score'] = round(hn, 3)

    return candidates, evidence


def _seed_gold_and_hard_negatives(query_row: dict[str, str], candidates: dict[str, dict[str, Any]]) -> tuple[list[str], list[str], str]:
    expected_source = str(query_row.get('expected_primary_source') or '').strip()
    ordered = sorted(
        candidates.values(),
        key=lambda item: (float(item['gold_seed_score']), -int(item['runtime_rank'] or 99), int(item['baseline_rank'] or 99), int(item['candidate_rank'] or 99)),
        reverse=True,
    )
    if expected_source and expected_source not in {'all', 'mixed'}:
        source_matched = [rec for rec in ordered if str(rec.get('source_type') or '') == expected_source]
        candidate_pool = source_matched
    else:
        candidate_pool = ordered
    gold: list[str] = []
    for rec in candidate_pool:
        if float(rec['gold_seed_score']) <= 0:
            continue
        gold.append(rec['doc_id'])
        if len(gold) >= 3:
            break

    hard_candidates = []
    for rec in candidates.values():
        if rec['doc_id'] in gold:
            continue
        if float(rec['hard_negative_seed_score']) >= 2.0:
            hard_candidates.append(rec)
    hard_candidates.sort(key=lambda item: (float(item['hard_negative_seed_score']), float(item['gold_seed_score'])), reverse=True)
    hard = [rec['doc_id'] for rec in hard_candidates[:3]]

    reason_bits = []
    if expected_source and expected_source not in {'all', 'mixed'}:
        reason_bits.append(f'expected_source={expected_source}')
    overlap = [rec['doc_id'] for rec in candidates.values() if rec['baseline_rank'] is not None and rec['candidate_rank'] is not None]
    if overlap:
        reason_bits.append(f'cross-model-overlap={len(overlap)}')
    runtime = [rec['doc_id'] for rec in candidates.values() if rec['runtime_rank'] is not None]
    if runtime:
        reason_bits.append(f'current-runtime={len(runtime)}')
    return gold, hard, ', '.join(reason_bits) if reason_bits else 'runtime only'


def build_seed_rows(*, query_rows: list[dict[str, str]], pairwise_rows: list[dict[str, str]], use_live_search: bool, cache_dir: Path | None, top_k: int) -> list[dict[str, str]]:
    pairwise_index = _build_pairwise_index(pairwise_rows)
    global_freq = _collect_global_doc_frequency(pairwise_rows)
    output: list[dict[str, str]] = []
    for row in query_rows:
        query = str(row.get('query') or '').strip()
        if not query:
            continue
        if use_live_search:
            payload = _fetch_search_payload(query, cache_dir=cache_dir, top_k=top_k)
        else:
            assert cache_dir is not None
            payload = _load_search_from_cache(query, cache_dir)
            payload['results'] = list(payload.get('results') or [])[:top_k]
        pairwise_row = pairwise_index.get(query)
        candidates, evidence = _candidate_map(row, pairwise_row, payload, global_freq)
        gold_seed, hard_seed, reason = _seed_gold_and_hard_negatives(row, candidates)
        runtime_results = list(payload.get('results') or [])[:top_k]
        output.append({
            'query': query,
            'query_variant_group': str(row.get('query_variant_group') or ''),
            'source': str(row.get('source') or ''),
            'query_type': str(row.get('query_type') or ''),
            'temporal_query': str(row.get('temporal_query') or ''),
            'expected_primary_source': str(row.get('expected_primary_source') or ''),
            'expected_answer_style': str(row.get('expected_answer_style') or ''),
            'difficulty': str(row.get('difficulty') or ''),
            'current_gold_doc_ids': str(row.get('gold_doc_ids') or ''),
            'current_hard_negative_doc_ids': str(row.get('hard_negative_doc_ids') or ''),
            'suggested_gold_doc_ids': ';'.join(gold_seed),
            'suggested_hard_negative_doc_ids': ';'.join(hard_seed),
            'runtime_top5_doc_ids': ';'.join(str(item.get('documentId') or '').strip() for item in runtime_results if str(item.get('documentId') or '').strip()),
            'runtime_top5_titles': ' || '.join(str(item.get('title') or '').strip() for item in runtime_results if str(item.get('documentId') or '').strip()),
            'runtime_top5_source_types': ';'.join(str(item.get('sourceType') or '').strip() for item in runtime_results if str(item.get('documentId') or '').strip()),
            'baseline_top3_doc_ids': str(pairwise_row.get('baseline_top3_doc_ids') or '') if pairwise_row else '',
            'baseline_top1_title': str(pairwise_row.get('baseline_top1_title') or '') if pairwise_row else '',
            'candidate_top3_doc_ids': str(pairwise_row.get('candidate_top3_doc_ids') or '') if pairwise_row else '',
            'candidate_top1_title': str(pairwise_row.get('candidate_top1_title') or '') if pairwise_row else '',
            'seed_inputs': ','.join(evidence) if evidence else 'none',
            'seed_reason': reason,
            'final_gold_doc_ids': '',
            'final_hard_negative_doc_ids': '',
            'review_notes': str(row.get('notes') or ''),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description='Build a seed review sheet for embedding gold/hard-negative curation.')
    parser.add_argument('--query-csv', required=True)
    parser.add_argument('--pairwise-machine-csv', action='append', default=[])
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--search-cache-dir')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--use-live-search', action='store_true')
    args = parser.parse_args()

    query_rows = _read_csv_rows(Path(args.query_csv))
    pairwise_rows: list[dict[str, str]] = []
    for raw in args.pairwise_machine_csv:
        pairwise_rows.extend(_read_csv_rows(Path(raw)))

    cache_dir = Path(args.search_cache_dir) if args.search_cache_dir else None
    rows = build_seed_rows(
        query_rows=query_rows,
        pairwise_rows=pairwise_rows,
        use_live_search=bool(args.use_live_search),
        cache_dir=cache_dir,
        top_k=args.top_k,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'outputCsv': str(output_path), 'rowCount': len(rows)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

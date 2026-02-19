"""전체 논문 요약+번역 배치 처리"""
import sqlite3, requests, time, os, re, json
from pathlib import Path

for line in open('/Users/won/Desktop/allinone/knowledge-hub/.env'):
    if '=' in line and not line.startswith('#'):
        k, v = line.strip().split('=', 1)
        os.environ[k] = v

DB_PATH = '/Users/won/Desktop/allinone/knowledge-hub/data/knowledge.db'
API_KEY = os.environ['OPENAI_API_KEY']

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# ── Step 1: abstract 확보 ──
papers = conn.execute('SELECT arxiv_id, title, notes FROM papers ORDER BY arxiv_id').fetchall()
papers = [dict(p) for p in papers]

need_abstract = [p for p in papers if not p.get('abstract')]
# notes 필드에 abstract가 없으므로 전부 필요

print(f'=== Step 1: Abstract 확보 ({len(papers)}편) ===', flush=True)

abstract_map = {}

# Semantic Scholar batch API
aids = [p['arxiv_id'] for p in papers]
for i in range(0, len(aids), 50):
    chunk = aids[i:i+50]
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={"fields": "title,abstract,externalIds"},
                json={"ids": [f"ArXiv:{a}" for a in chunk]},
                timeout=60,
            )
            if resp.status_code == 200:
                for paper in resp.json():
                    if paper and paper.get("abstract"):
                        ext = paper.get("externalIds", {})
                        aid = ext.get("ArXiv", "")
                        if aid:
                            abstract_map[aid] = paper["abstract"]
                break
            elif resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f'  S2 error: {e}', flush=True)
            time.sleep(5)

print(f'  Semantic Scholar: {len(abstract_map)}편 abstract 확보', flush=True)

# arXiv API로 나머지
missing = [a for a in aids if a not in abstract_map]
if missing:
    print(f'  arXiv API로 {len(missing)}편 추가 조회...', flush=True)
    import xml.etree.ElementTree as ET
    for i in range(0, len(missing), 10):
        chunk = missing[i:i+10]
        url = f"http://export.arxiv.org/api/query?id_list={','.join(chunk)}&max_results={len(chunk)}"
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    ns = {'a': 'http://www.w3.org/2005/Atom'}
                    root = ET.fromstring(resp.text)
                    for entry in root.findall('a:entry', ns):
                        id_el = entry.find('a:id', ns)
                        summary_el = entry.find('a:summary', ns)
                        if id_el is not None and summary_el is not None:
                            aid = id_el.text.split('/abs/')[-1].split('v')[0]
                            abstract_map[aid] = ' '.join(summary_el.text.split())
                    break
                else:
                    time.sleep(10)
            except Exception:
                time.sleep(10)
        time.sleep(3)

print(f'  총 {len(abstract_map)}편 abstract 확보\n', flush=True)

# ── Step 2+3: 요약 + 번역 (OpenAI batch) ──
print(f'=== Step 2+3: 요약 + 번역 ===', flush=True)

def openai_chat(messages, max_tokens=1000):
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": "gpt-4o-mini", "messages": messages, "max_tokens": max_tokens, "temperature": 0.3},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

success = 0
for idx, p in enumerate(papers, 1):
    aid = p['arxiv_id']
    title = p['title']
    abstract = abstract_map.get(aid, '')

    if not abstract:
        print(f'  [{idx}/{len(papers)}] {aid} - abstract 없음, 스킵', flush=True)
        continue

    print(f'  [{idx}/{len(papers)}] {title[:50]}...', end=' ', flush=True)

    try:
        # 요약 + 번역을 한 번의 API 호출로
        result = openai_chat([
            {"role": "system", "content": "당신은 AI 논문 전문가입니다."},
            {"role": "user", "content": f"""다음 논문의 초록을 분석해주세요.

제목: {title}
초록: {abstract}

다음 형식으로 JSON만 응답하세요:
{{
  "summary_ko": "한국어 요약 (3-5문장, 핵심 기여와 방법론 포함)",
  "abstract_ko": "초록 전체 한국어 번역"
}}"""}
        ], max_tokens=2000)

        # JSON 파싱
        result = result.strip()
        if result.startswith('```'):
            result = re.sub(r'^```\w*\n?', '', result)
            result = re.sub(r'\n?```$', '', result)

        data = json.loads(result)
        summary_ko = data.get('summary_ko', '')
        abstract_ko = data.get('abstract_ko', '')

        # DB 업데이트
        conn.execute(
            "UPDATE papers SET notes = ? WHERE arxiv_id = ?",
            (f"요약: {summary_ko}\n\n원문 초록: {abstract[:200]}...\ncitations: {p['notes'] or ''}", aid)
        )
        conn.commit()

        print(f'OK', flush=True)
        success += 1

        # Obsidian 노트 업데이트용 데이터 저장
        p['summary_ko'] = summary_ko
        p['abstract_ko'] = abstract_ko
        p['abstract'] = abstract

    except Exception as e:
        print(f'FAIL ({e})', flush=True)

    if idx % 10 == 0:
        time.sleep(1)

print(f'\n=== {success}/{len(papers)}편 요약+번역 완료 ===', flush=True)

# ── Step 4: Obsidian 노트 업데이트 ──
print(f'\n=== Step 4: Obsidian 노트 업데이트 ===', flush=True)

vault_dirs = [
    Path("/Users/won/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Papers"),
    Path("/Users/won/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/AI/AI_Papers"),
]

updated_notes = 0
for p in papers:
    if not p.get('summary_ko'):
        continue

    aid = p['arxiv_id']
    for vault_dir in vault_dirs:
        for md_file in vault_dir.glob('*.md'):
            content = md_file.read_text(encoding='utf-8', errors='ignore')
            if f'arxiv_id' not in content:
                continue
            if aid not in content:
                continue

            # 요약 섹션이 없으면 추가
            if '## 요약' not in content:
                insert_pos = content.find('## Abstract')
                if insert_pos == -1:
                    insert_pos = content.find('---', content.find('---') + 1) + 3
                if insert_pos > 0:
                    summary_block = f"\n\n## 요약\n\n{p['summary_ko']}\n\n## 초록 (한국어)\n\n{p['abstract_ko']}\n"
                    content = content[:insert_pos] + summary_block + content[insert_pos:]
                    md_file.write_text(content, encoding='utf-8')
                    updated_notes += 1

print(f'{updated_notes}개 Obsidian 노트 업데이트 완료', flush=True)

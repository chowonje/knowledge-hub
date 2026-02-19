import re, requests, time, xml.etree.ElementTree as ET
from pathlib import Path

VAULT_AI = Path("/Users/won/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/AI/AI_Papers")
VAULT_PAPERS = Path("/Users/won/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Papers")

def safe_title(title):
    safe = re.sub(r'[\\/:*?"<>|]', '', title).strip()
    return re.sub(r'\s+', ' ', safe)[:100].strip()

arxiv_pattern = re.compile(r'^(\d{4}\.\d{4,5})(\.(?:pdf|txt|md))?$')

# 1) Obsidian Papers/ 노트에서 제목 추출
title_map = {}
for f in VAULT_PAPERS.glob('*.md'):
    text = f.read_text(encoding='utf-8', errors='ignore')[:500]
    m = re.search(r'arxiv_id:\s*"?(\S+?)"?\s*$', text, re.MULTILINE)
    if m:
        title_map[m.group(1).strip('"')] = f.stem

# 2) 남은 ID 확인
remaining_ids = sorted(set(
    m.group(1) for f in VAULT_AI.iterdir()
    if (m := arxiv_pattern.match(f.name)) and m.group(1) not in title_map
))

# 3) arXiv API로 나머지 조회 (5개씩, 간격 5초)
if remaining_ids:
    print(f'{len(remaining_ids)}편 arXiv API 조회...', flush=True)
    for i in range(0, len(remaining_ids), 5):
        chunk = remaining_ids[i:i+5]
        url = f"http://export.arxiv.org/api/query?id_list={','.join(chunk)}&max_results={len(chunk)}"
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    ns = {'a': 'http://www.w3.org/2005/Atom'}
                    root = ET.fromstring(resp.text)
                    for entry in root.findall('a:entry', ns):
                        id_el = entry.find('a:id', ns)
                        title_el = entry.find('a:title', ns)
                        if id_el is not None and title_el is not None:
                            aid = id_el.text.split('/abs/')[-1].split('v')[0]
                            t = ' '.join(title_el.text.split())
                            if t and 'Error' not in t:
                                title_map[aid] = t
                    break
                else:
                    time.sleep(10)
            except Exception:
                time.sleep(10)
        time.sleep(5)

# 4) 리네임
renamed = 0
for f in sorted(VAULT_AI.iterdir()):
    m = arxiv_pattern.match(f.name)
    if not m:
        continue
    aid = m.group(1)
    title = title_map.get(aid)
    if not title:
        continue
    new_name = f"{safe_title(title)}{f.suffix}"
    new_path = f.parent / new_name
    if not new_path.exists():
        f.rename(new_path)
        renamed += 1

still = sorted(f.name for f in VAULT_AI.iterdir() if arxiv_pattern.match(f.name))
print(f'{renamed}개 리네임, {len(still)}개 남음', flush=True)
for n in still:
    print(f'  {n}')

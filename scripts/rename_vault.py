"""Obsidian AI_Papers 폴더의 arxiv ID 파일들을 논문 제목으로 리네임"""
import re, requests, time
from pathlib import Path

VAULT_DIR = Path("/Users/won/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/AI/AI_Papers")

def safe_title(title):
    safe = re.sub(r'[\\/:*?"<>|]', '', title).strip()
    return re.sub(r'\s+', ' ', safe)[:100].strip()

arxiv_pattern = re.compile(r'^(\d{4}\.\d{4,5})(\.(?:pdf|txt|md))?$')
remaining = sorted(set(
    m.group(1) for f in VAULT_DIR.iterdir()
    if (m := arxiv_pattern.match(f.name))
))
print(f'{len(remaining)} IDs 남음', flush=True)

title_map = {}
ids_list = [f"ArXiv:{aid}" for aid in remaining]

for attempt in range(5):
    try:
        resp = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": "title,externalIds"},
            json={"ids": ids_list},
            timeout=60,
        )
        if resp.status_code == 200:
            for paper in resp.json():
                if paper and paper.get("title"):
                    ext = paper.get("externalIds", {})
                    aid = ext.get("ArXiv", "")
                    if aid:
                        title_map[aid] = paper["title"]
            break
        else:
            print(f'  status {resp.status_code}, retry {5*(attempt+1)}s...', flush=True)
            time.sleep(5 * (attempt + 1))
    except Exception as e:
        print(f'  error: {e}, retry...', flush=True)
        time.sleep(5)

print(f'{len(title_map)}편 제목 확보', flush=True)

renamed = 0
for f in sorted(VAULT_DIR.iterdir()):
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

still = sorted(f.name for f in VAULT_DIR.iterdir() if arxiv_pattern.match(f.name))
print(f'{renamed}개 리네임, {len(still)}개 남음', flush=True)
for n in still:
    print(f'  {n}')

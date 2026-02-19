"""
Google NotebookLM Enterprise API 클라이언트

노트북 생성, 소스 추가, 논문 동기화를 지원합니다.
API docs: https://cloud.google.com/gemini/enterprise/notebooklm-enterprise/docs/api-notebooks

필요: GOOGLE_CLOUD_PROJECT, NOTEBOOKLM_LOCATION 환경변수 또는 config 설정
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from rich.console import Console

console = Console()


class NotebookLMClient:
    """NotebookLM Enterprise REST API 클라이언트"""

    def __init__(self, project_number: str, location: str = "global"):
        self.project_number = project_number
        self.location = location
        self.base_url = (
            f"https://{location}-discoveryengine.googleapis.com/v1alpha"
            f"/projects/{project_number}/locations/{location}"
        )
        self._token = None

    @property
    def token(self) -> str:
        if not self._token:
            self._token = self._get_access_token()
        return self._token

    @staticmethod
    def _get_access_token() -> str:
        """gcloud CLI에서 액세스 토큰 획득"""
        import subprocess
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        token = os.environ.get("GOOGLE_ACCESS_TOKEN", "")
        if token:
            return token
        raise RuntimeError(
            "Google Cloud 인증이 필요합니다.\n"
            "  1) gcloud auth login && gcloud auth application-default login\n"
            "  2) 또는 GOOGLE_ACCESS_TOKEN 환경변수 설정"
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, headers=self._headers(), timeout=60, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    # ── Notebook CRUD ──

    def create_notebook(self, title: str, description: str = "") -> dict:
        """새 노트북 생성"""
        body = {"display_name": title}
        if description:
            body["description"] = description
        return self._request("POST", "/notebooks", json=body)

    def list_notebooks(self, page_size: int = 20) -> List[dict]:
        """노트북 목록 조회"""
        resp = self._request("GET", f"/notebooks?pageSize={page_size}")
        return resp.get("notebooks", [])

    def get_notebook(self, notebook_id: str) -> dict:
        return self._request("GET", f"/notebooks/{notebook_id}")

    def delete_notebook(self, notebook_id: str) -> dict:
        return self._request("DELETE", f"/notebooks/{notebook_id}")

    # ── Source Management ──

    def add_source_text(self, notebook_id: str, title: str, text: str) -> dict:
        """텍스트 소스 추가"""
        body = {
            "sources": [{
                "display_name": title,
                "inline_source": {"content": text, "mime_type": "text/plain"},
            }]
        }
        return self._request(
            "POST",
            f"/notebooks/{notebook_id}/sources:batchCreate",
            json=body,
        )

    def add_source_url(self, notebook_id: str, url: str) -> dict:
        """URL 소스 추가"""
        body = {
            "sources": [{
                "url_source": {"uri": url},
            }]
        }
        return self._request(
            "POST",
            f"/notebooks/{notebook_id}/sources:batchCreate",
            json=body,
        )

    def list_sources(self, notebook_id: str) -> List[dict]:
        resp = self._request("GET", f"/notebooks/{notebook_id}/sources")
        return resp.get("sources", [])

    # ── 논문 동기화 ──

    def sync_paper(self, notebook_id: str, paper: dict) -> bool:
        """논문 메타데이터+요약을 노트북 소스로 추가"""
        title = paper.get("title", "")
        arxiv_id = paper.get("arxiv_id", "")

        parts = [f"# {title}", f"arXiv: {arxiv_id}"]

        if paper.get("authors"):
            parts.append(f"저자: {paper['authors']}")
        if paper.get("year"):
            parts.append(f"연도: {paper['year']}")
        if paper.get("field"):
            parts.append(f"분야: {paper['field']}")

        notes = paper.get("notes", "")
        if notes and len(notes) > 30:
            parts.append(f"\n## 요약\n{notes}")

        text = "\n".join(parts)

        try:
            self.add_source_text(notebook_id, f"[{arxiv_id}] {title[:60]}", text)
            if paper.get("pdf_path") and Path(paper["pdf_path"]).exists():
                arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                self.add_source_url(notebook_id, arxiv_url)
            return True
        except Exception as e:
            console.print(f"  [red]소스 추가 실패: {e}[/red]")
            return False


def get_notebooklm_client(config) -> Optional[NotebookLMClient]:
    """Config에서 NotebookLM 클라이언트 생성"""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        project = config.get_nested("notebooklm", "project_number", default="")
    location = os.environ.get("NOTEBOOKLM_LOCATION", "global")
    if not location:
        location = config.get_nested("notebooklm", "location", default="global")

    if not project:
        return None

    return NotebookLMClient(project_number=project, location=location)


def generate_study_pack(paper: dict, output_dir: str) -> str:
    """
    NotebookLM 없이도 사용 가능한 로컬 스터디 팩 생성.
    구조화된 마크다운 문서로 논문 학습에 필요한 모든 정보를 정리.
    """
    title = paper.get("title", "Unknown")
    arxiv_id = paper.get("arxiv_id", "")

    lines = [
        f"# {title}",
        "",
        "## 기본 정보",
        f"- **arXiv:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})",
        f"- **저자:** {paper.get('authors', '')}",
        f"- **연도:** {paper.get('year', '')}",
        f"- **분야:** {paper.get('field', '')}",
        "",
    ]

    notes = paper.get("notes", "")
    if notes and len(notes) > 30:
        lines.extend(["## 요약", "", notes, ""])

    translated_path = paper.get("translated_path", "")
    if translated_path and Path(translated_path).exists():
        lines.extend([
            "## 전체 번역",
            f"[번역 파일 열기]({translated_path})",
            "",
        ])

    lines.extend([
        "## 학습 체크리스트",
        "- [ ] 초록 읽기",
        "- [ ] 핵심 기여(contribution) 파악",
        "- [ ] 방법론 이해",
        "- [ ] 실험 결과 분석",
        "- [ ] 한계점 및 향후 연구 파악",
        "",
        "## 메모",
        "_여기에 학습 메모를 작성하세요_",
        "",
    ])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    import re
    safe_title = re.sub(r'[\\/:*?"<>|]', '', title).strip()
    safe_title = re.sub(r'\s+', ' ', safe_title)[:100].strip()
    out_path = Path(output_dir) / f"{safe_title}_study.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)

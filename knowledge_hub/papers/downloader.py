"""
arXiv 논문 다운로더

CSV에서 논문 목록을 읽고 PDF/텍스트를 다운로드합니다.
"""

import csv
import os
import tarfile
import time
from pathlib import Path
from typing import List, Optional

import requests
from rich.console import Console
from rich.table import Table

console = Console()


def arxiv_pdf_url(arxiv_id: str) -> str:
    base_id = arxiv_id.strip().split("v")[0]
    return f"https://arxiv.org/pdf/{base_id}.pdf"


def arxiv_text_url(arxiv_id: str) -> str:
    base_id = arxiv_id.strip().split("v")[0]
    return f"https://export.arxiv.org/e-print/{base_id}"


def extract_text_from_tar(tar_path: str) -> Optional[str]:
    """tar에서 .tex 메인 파일 추출"""
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            tex_files = [m for m in tar.getmembers() if m.name.endswith(".tex")]
            if not tex_files:
                return None
            main_tex = max(tex_files, key=lambda m: m.size)
            return tar.extractfile(main_tex).read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _safe_filename(title: str, arxiv_id: str) -> str:
    """논문 제목을 안전한 파일명으로 변환"""
    import re
    safe = re.sub(r'[\\/:*?"<>|]', '', title).strip()
    safe = re.sub(r'\s+', ' ', safe)[:100].strip()
    if not safe:
        return arxiv_id
    return safe


class PaperDownloader:
    """arXiv 논문 다운로드 관리"""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _find_existing(self, arxiv_id: str, title: str, ext: str) -> Optional[Path]:
        """기존 파일 찾기 (현재/이전 형식 호환)"""
        if title:
            title_path = self.save_dir / f"{_safe_filename(title, arxiv_id)}{ext}"
            if title_path.exists():
                return title_path
        old_path = self.save_dir / f"{arxiv_id}{ext}"
        if old_path.exists():
            return old_path
        for p in self.save_dir.glob(f"*[{arxiv_id}]{ext}"):
            return p
        return None

    def download_single(self, arxiv_id: str, title: str = "") -> dict:
        """단일 논문 다운로드"""
        result = {
            "arxiv_id": arxiv_id,
            "title": title,
            "pdf": None,
            "text": None,
            "success": False,
        }

        base = _safe_filename(title, arxiv_id)
        pdf_path = self.save_dir / f"{base}.pdf"
        text_path = self.save_dir / f"{base}.txt"

        existing_pdf = self._find_existing(arxiv_id, title, ".pdf")
        existing_txt = self._find_existing(arxiv_id, title, ".txt")

        # PDF 다운로드
        if existing_pdf:
            if existing_pdf != pdf_path and title:
                existing_pdf.rename(pdf_path)
            result["pdf"] = str(pdf_path if title else existing_pdf)
        else:
            try:
                url = arxiv_pdf_url(arxiv_id)
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200 and "application/pdf" in resp.headers.get(
                    "Content-Type", ""
                ):
                    pdf_path.write_bytes(resp.content)
                    result["pdf"] = str(pdf_path)
                    console.print(f"  [green]PDF 저장: {pdf_path.name}[/green]")
            except Exception as e:
                console.print(f"  [red]PDF 다운로드 실패: {e}[/red]")

        # 텍스트 소스 다운로드
        if existing_txt:
            if existing_txt != text_path and title:
                existing_txt.rename(text_path)
            result["text"] = str(text_path if title else existing_txt)
        else:
            try:
                url = arxiv_text_url(arxiv_id)
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200:
                    ct = resp.headers.get("Content-Type", "").lower()
                    if "tar" in ct or "gzip" in ct or "tgz" in ct:
                        tar_path = str(text_path).replace(".txt", ".tar.gz")
                        Path(tar_path).write_bytes(resp.content)
                        text_content = extract_text_from_tar(tar_path)
                        if text_content:
                            text_path.write_text(text_content, encoding="utf-8")
                            result["text"] = str(text_path)
                        os.remove(tar_path)
                    elif "text/plain" in ct:
                        text_path.write_bytes(resp.content)
                        result["text"] = str(text_path)
            except Exception:
                pass

        result["success"] = result["pdf"] is not None
        return result

    def download_from_csv(self, csv_path: str) -> List[dict]:
        """CSV에서 논문 목록 읽고 일괄 다운로드"""
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        console.print(f"\n[bold]논문 다운로드 시작[/bold] ({len(rows)}개)")
        console.print(f"저장 경로: {self.save_dir}\n")

        results = []
        for idx, row in enumerate(rows, 1):
            title = (row.get("논문제목") or row.get("title") or "").strip()
            arxiv_id = (row.get("arXivID") or row.get("arxiv_id") or "").strip()

            if not arxiv_id or arxiv_id == "-":
                continue

            console.print(f"[{idx}/{len(rows)}] {title[:60]}")
            result = self.download_single(arxiv_id, title)
            results.append(result)
            time.sleep(1)

        success = sum(1 for r in results if r["success"])
        console.print(f"\n[bold green]다운로드 완료: {success}/{len(results)} 성공[/bold green]")
        return results

"""
논문 번역 모듈

OpenAI API를 사용하여 논문을 한국어로 번역합니다.
"""

import os
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def _load_env():
    """환경 변수에서 API 키 로드"""
    if os.getenv("OPENAI_API_KEY"):
        return
    for env_path in [".env", os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            except Exception:
                pass


class PaperTranslator:
    """OpenAI 기반 논문 번역"""

    def __init__(self, model: str = "gpt-4.1-mini"):
        _load_env()
        from openai import OpenAI

        self.model = model
        self.client = OpenAI()

    def translate_chunk(self, text: str, max_retries: int = 3) -> str:
        """텍스트 청크를 한국어로 번역"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional Korean translator specializing in academic papers "
                                "in computer vision and machine learning. "
                                "Translate the following English text into natural, academic Korean. "
                                "Preserve all equations, symbols, citations, figure/table numbers, and technical terms. "
                                "Output only the translated Korean text."
                            ),
                        },
                        {"role": "user", "content": f"Translate to Korean:\n\n{text}"},
                    ],
                    temperature=0.3,
                )
                return response.choices[0].message.content
            except Exception as e:
                console.print(f"  [yellow]번역 재시도 ({attempt + 1}/{max_retries}): {e}[/yellow]")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
        return text

    def proofread_chunk(self, text: str, max_retries: int = 3) -> str:
        """번역된 텍스트 교열"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Korean academic editor. "
                                "Edit the Korean text to read naturally. "
                                "Do not change meaning, equations, symbols, citations, or technical terms. "
                                "Output only the edited Korean text."
                            ),
                        },
                        {"role": "user", "content": f"교열할 텍스트:\n\n{text}"},
                    ],
                    temperature=0.3,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
        return text

    def translate_paper(
        self,
        text_path: str,
        output_dir: str,
        arxiv_id: str,
        title: str,
        proofread: bool = True,
        generate_pdf: bool = False,
    ) -> Optional[str]:
        """
        논문 전체 번역

        Returns:
            번역된 마크다운 파일 경로 (또는 실패 시 None)
        """
        try:
            text = Path(text_path).read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"  [red]텍스트 읽기 실패: {e}[/red]")
            return None

        console.print(f"  텍스트 길이: {len(text)} 문자")

        # 청크 분할
        chunk_size = int(os.getenv("PAPER_TRANSLATE_CHUNK_SIZE", "5000"))
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        console.print(f"  청크 수: {len(chunks)}")

        # 번역
        translated = []
        for idx, chunk in enumerate(chunks, 1):
            console.print(f"    [{idx}/{len(chunks)}] 번역 중...")
            translated.append(self.translate_chunk(chunk))

        # 교열
        if proofread:
            console.print("  교열 중...")
            for idx in range(len(translated)):
                console.print(f"    [{idx + 1}/{len(translated)}] 교열 중...")
                translated[idx] = self.proofread_chunk(translated[idx])

        final_text = "\n\n".join(translated)

        # Markdown 저장
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        md_path = os.path.join(output_dir, f"{arxiv_id}_translated.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"**arXiv ID:** {arxiv_id}\n\n")
            f.write(f"**상태:** 번역{' + 교열' if proofread else ''} 완료\n\n")
            f.write("---\n\n")
            f.write(final_text)

        console.print(f"  [green]저장: {md_path}[/green]")

        # PDF 생성 (선택사항)
        if generate_pdf:
            self._generate_pdf(final_text, output_dir, arxiv_id, title)

        return md_path

    def _generate_pdf(self, text: str, output_dir: str, arxiv_id: str, title: str):
        """번역 결과 PDF 생성"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

            font_name = "Helvetica"
            for fp in [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            ]:
                if os.path.exists(fp):
                    try:
                        pdfmetrics.registerFont(TTFont("Korean", fp))
                        font_name = "Korean"
                        break
                    except Exception:
                        continue

            pdf_path = os.path.join(output_dir, f"{arxiv_id}_translated.pdf")
            doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle("T", parent=styles["Heading1"], fontName=font_name, fontSize=16)
            body_style = ParagraphStyle("B", parent=styles["BodyText"], fontName=font_name, fontSize=10, leading=14)

            story = [Paragraph(title, title_style), Spacer(1, 0.3 * inch)]
            for para in text.split("\n\n"):
                if para.strip():
                    safe = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    story.append(Paragraph(safe, body_style))
            doc.build(story)
            console.print(f"  [green]PDF 저장: {pdf_path}[/green]")
        except ImportError:
            console.print("  [yellow]reportlab 미설치, PDF 생성 건너뜀[/yellow]")
        except Exception as e:
            console.print(f"  [yellow]PDF 생성 실패: {e}[/yellow]")

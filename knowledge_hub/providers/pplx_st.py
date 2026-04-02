"""
Perplexity Embeddings local provider via SentenceTransformers.

Mac ARM-friendly path when Docker/TEI is unavailable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from knowledge_hub.providers.base import BaseEmbedder, ProviderInfo

log = logging.getLogger("khub.providers.pplx_st")


class PPLXEmbeddingError(RuntimeError):
    """Structured runtime error for pplx-st embedding failures."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class PPLXSentenceTransformerEmbedder(BaseEmbedder):
    """Local embedding provider using sentence-transformers."""

    def __init__(
        self,
        model: str = "perplexity-ai/pplx-embed-v1-0.6b",
        batch_size: int = 8,
        device: str = "auto",
        torch_num_threads: int = 1,
        disable_tokenizers_parallelism: bool = True,
        max_chars_per_chunk: int = 1000,
        chunk_overlap_chars: int = 200,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = True,
        load_timeout_sec: int = 600,
        encode_timeout_sec: int = 180,
        auto_batch_backoff: bool = True,
        min_batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.batch_size = int(batch_size)
        self.device = self._resolve_device(device)
        self.torch_num_threads = max(1, int(torch_num_threads))
        self.disable_tokenizers_parallelism = bool(disable_tokenizers_parallelism)
        self.max_chars_per_chunk = max(200, int(max_chars_per_chunk))
        self.chunk_overlap_chars = max(0, min(int(chunk_overlap_chars), self.max_chars_per_chunk - 1))
        self.normalize_embeddings = bool(normalize_embeddings)
        self.trust_remote_code = bool(trust_remote_code)
        self.load_timeout_sec = max(1, int(load_timeout_sec))
        self.encode_timeout_sec = max(1, int(encode_timeout_sec))
        self.auto_batch_backoff = bool(auto_batch_backoff)
        self.min_batch_size = max(1, int(min_batch_size))
        self._encoder = None
        self._model_ref = model
        self._last_failures: list[dict] = []
        self._last_retries: int = 0

    @staticmethod
    def _resolve_device(device: str) -> str:
        wanted = (device or "auto").strip().lower()
        if wanted and wanted != "auto":
            return wanted
        probe = (
            "import sys\n"
            "device='cpu'\n"
            "try:\n"
            "    import torch\n"
            "    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():\n"
            "        device='mps'\n"
            "    elif torch.cuda.is_available():\n"
            "        device='cuda'\n"
            "except Exception:\n"
            "    device='cpu'\n"
            "sys.stdout.write(device)\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception as error:
            log.warning("device auto-detection failed; fallback to cpu: %s", error)
            return "cpu"
        if result.returncode != 0:
            error_text = (result.stderr or "").strip() or f"exit={result.returncode}"
            log.warning("device auto-detection failed; fallback to cpu: %s", error_text)
            return "cpu"
        resolved = (result.stdout or "").strip().lower()
        return resolved if resolved in {"cpu", "mps", "cuda"} else "cpu"

    @staticmethod
    def _repo_cache_dirs(repo_id: str) -> list[Path]:
        safe = (repo_id or "").replace("/", "--")
        hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        return [
            hf_home / "hub" / f"models--{safe}",
            hf_home / "modules" / "transformers_modules" / safe,
        ]

    def _cleanup_repo_cache(self, repo_id: str):
        for path in self._repo_cache_dirs(repo_id):
            try:
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
            except Exception:
                continue

    @staticmethod
    def _enforce_timeout(elapsed_sec: float, timeout_sec: int, code: str, context: str):
        if elapsed_sec > max(1, int(timeout_sec)):
            raise PPLXEmbeddingError(
                code,
                f"{context} timeout after {elapsed_sec:.1f}s (> {timeout_sec}s)",
            )

    def _resolve_model_ref(self, snapshot_download) -> str:
        model_ref = self.model
        if "/" not in self.model or os.path.exists(self.model):
            return model_ref

        try:
            return snapshot_download(repo_id=self.model)
        except Exception as first_error:
            log.warning("snapshot_download failed for %s: %s", self.model, first_error)
            self._cleanup_repo_cache(self.model)
            try:
                return snapshot_download(repo_id=self.model, force_download=True)
            except Exception as second_error:
                log.warning("snapshot_download retry failed for %s: %s", self.model, second_error)
                return model_ref

    @property
    def encoder(self):
        if self._encoder is None:
            self._apply_runtime_safety_defaults()
            self._ensure_repo_custom_modules()
            try:
                from sentence_transformers import SentenceTransformer
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers 패키지 필요: pip install 'knowledge-hub-cli[st]'"
                ) from exc

            model_ref = self._resolve_model_ref(snapshot_download)
            self._model_ref = model_ref
            started = time.monotonic()
            log.info(
                "pplx-st model loading start: model=%s ref=%s device=%s trust_remote_code=%s",
                self.model,
                model_ref,
                self.device,
                self.trust_remote_code,
            )

            try:
                self._encoder = SentenceTransformer(
                    model_ref,
                    device=self.device,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                self._enforce_timeout(
                    elapsed_sec=elapsed,
                    timeout_sec=self.load_timeout_sec,
                    code="MODEL_LOAD_TIMEOUT",
                    context="model load",
                )
                raise PPLXEmbeddingError(
                    "MODEL_LOAD_FAILED",
                    (
                        f"model={self.model}, ref={model_ref}, trust_remote_code={self.trust_remote_code}, "
                        f"device={self.device}, hint='retry with cpu or clear HF cache'"
                    ),
                ) from exc

            elapsed = time.monotonic() - started
            self._enforce_timeout(
                elapsed_sec=elapsed,
                timeout_sec=self.load_timeout_sec,
                code="MODEL_LOAD_TIMEOUT",
                context="model load",
            )
            log.info("pplx-st model loading done: elapsed=%.2fs", elapsed)
        return self._encoder

    def _ensure_repo_custom_modules(self):
        """Perplexity ST models may reference repo-local modules (e.g. st_quantize.py)."""
        try:
            # Some Perplexity checkpoints import `Module` from sentence_transformers.models.
            # Newer/older sentence-transformers releases may not export it, so add a shim.
            import torch
            import sentence_transformers.models as st_models
            if not hasattr(st_models, "Module"):
                class _CompatModule(torch.nn.Module):
                    @classmethod
                    def load(cls, *args, **kwargs):
                        return cls()

                    def save(self, *args, **kwargs):
                        return None

                st_models.Module = _CompatModule

            from huggingface_hub import hf_hub_download
            mod_path = hf_hub_download(repo_id=self.model, filename="st_quantize.py")
            mod_dir = os.path.dirname(mod_path)
            if mod_dir not in sys.path:
                sys.path.insert(0, mod_dir)
        except Exception as error:
            # Not all models provide custom modules; keep running with base model.
            log.warning("optional custom module bootstrap failed for %s: %s", self.model, error)

    def _apply_runtime_safety_defaults(self):
        """Avoid common tokenizer/torch thread lockups on local Macs."""
        if self.disable_tokenizers_parallelism and "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ.setdefault("OMP_NUM_THREADS", str(self.torch_num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.torch_num_threads))

    def _split_text_by_chars(self, text: str) -> list[str]:
        clean = (text or "").strip()
        if not clean:
            return []
        if len(clean) <= self.max_chars_per_chunk:
            return [clean]

        step = self.max_chars_per_chunk - self.chunk_overlap_chars
        chunks: list[str] = []
        start = 0
        n = len(clean)
        while start < n:
            end = min(n, start + self.max_chars_per_chunk)
            piece = clean[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= n:
                break
            start += step
        return chunks

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return vec / norm

    def _encode_clean_texts(
        self,
        texts: list[str],
        show_progress: bool = False,
        batch_size_override: int | None = None,
    ) -> list[list[float]]:
        """Encode texts with automatic char-based chunk splitting + mean pooling."""
        chunk_texts: list[str] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        for text in texts:
            chunks = self._split_text_by_chars(text)
            if not chunks:
                spans.append((cursor, cursor))
                continue
            chunk_texts.extend(chunks)
            spans.append((cursor, cursor + len(chunks)))
            cursor += len(chunks)

        if not chunk_texts:
            return [[] for _ in texts]

        active_batch_size = max(1, int(batch_size_override or self.batch_size))
        started = time.monotonic()
        vectors = self.encoder.encode(
            chunk_texts,
            batch_size=active_batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        elapsed = time.monotonic() - started
        self._enforce_timeout(
            elapsed_sec=elapsed,
            timeout_sec=self.encode_timeout_sec,
            code="ENCODE_TIMEOUT",
            context=f"encode(batch_size={active_batch_size}, chunks={len(chunk_texts)})",
        )
        vectors = np.asarray(vectors)

        pooled: list[list[float]] = []
        for start, end in spans:
            if end <= start:
                pooled.append([])
                continue
            mean_vec = vectors[start:end].mean(axis=0)
            if self.normalize_embeddings:
                mean_vec = self._l2_normalize(mean_vec)
            pooled.append(mean_vec.tolist())
        return pooled

    def _encode_with_backoff(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        if not texts:
            return []

        current_batch = max(self.min_batch_size, int(self.batch_size))
        while True:
            try:
                return self._encode_clean_texts(
                    texts,
                    show_progress=show_progress,
                    batch_size_override=current_batch,
                )
            except Exception as exc:
                if not self.auto_batch_backoff or current_batch <= self.min_batch_size:
                    raise
                next_batch = max(self.min_batch_size, current_batch // 2)
                self._last_retries += 1
                self._last_failures.append(
                    {
                        "stage": "embed_batch",
                        "errorCode": "ENCODE_BATCH_FAILED",
                        "message": str(exc),
                        "batchSize": current_batch,
                        "nextBatchSize": next_batch,
                    }
                )
                log.warning(
                    "pplx-st encode failed at batch=%s, retry with batch=%s: %s",
                    current_batch,
                    next_batch,
                    exc,
                )
                current_batch = next_batch

    def reset_last_status(self):
        self._last_failures = []
        self._last_retries = 0

    def get_last_status(self) -> dict:
        return {
            "retries": int(self._last_retries),
            "failures": list(self._last_failures),
        }

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩할 수 없습니다")
        self.reset_last_status()
        vec = self._encode_with_backoff([text], show_progress=False)[0]
        if not vec:
            raise ValueError("텍스트 임베딩 생성 실패")
        return vec

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[Optional[List[float]]]:
        self.reset_last_status()
        clean = [t for t in texts if t and t.strip()]
        if not clean:
            return [None] * len(texts)

        vectors: list[list[float]] = []
        try:
            vectors = self._encode_with_backoff(clean, show_progress=show_progress)
        except Exception as exc:
            log.warning("pplx-st full-batch encode failed; fallback to per-item: %s", exc)
            self._last_failures.append(
                {
                    "stage": "embed_batch",
                    "errorCode": "ENCODE_FULL_BATCH_FAILED",
                    "message": str(exc),
                    "batchSize": int(self.batch_size),
                }
            )
            vectors = []
            for idx, text in enumerate(clean):
                try:
                    single = self._encode_with_backoff([text], show_progress=False)
                    vectors.append(single[0] if single else [])
                except Exception as item_exc:
                    self._last_failures.append(
                        {
                            "stage": "embed_batch",
                            "errorCode": "ENCODE_ITEM_FAILED",
                            "message": str(item_exc),
                            "itemIndex": idx,
                        }
                    )
                    vectors.append([])

        results, clean_idx = [], 0
        for t in texts:
            if t and t.strip():
                vector = vectors[clean_idx] if clean_idx < len(vectors) else []
                results.append(vector if vector else None)
                clean_idx += 1
            else:
                results.append(None)
        return results

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        return ProviderInfo(
            name="pplx-st",
            display_name="Perplexity Embeddings (Local/ST)",
            supports_llm=False,
            supports_embedding=True,
            requires_api_key=False,
            is_local=True,
            default_embed_model="perplexity-ai/pplx-embed-v1-0.6b",
            available_models=[
                "perplexity-ai/pplx-embed-v1-0.6b",
                "perplexity-ai/pplx-embed-v1-4b",
                "perplexity-ai/pplx-embed-context-v1-0.6b",
                "perplexity-ai/pplx-embed-context-v1-4b",
            ],
        )

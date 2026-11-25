"""Persistent chunk index: Semantica ``VectorStore`` (FAISS) + local embeddings.

Two Semantica 0.7.0 behaviours are guarded here:

* if the embedding model cannot be loaded, ``TextEmbedder`` / ``VectorStore``
  fall back to hash or **random** vectors and only log a warning - we refuse to
  index in that state;
* with ``metric="inner_product"`` FAISS results are scored ``1/(1+d)``, which
  inverts the ranking.  We keep the default L2 metric; embeddings are unit
  length, so cosine similarity is ``1 - d/2``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from semantica.vector_store import VectorStore

from backend.config import Settings
from backend.schemas import Chunk, Evidence

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    pass


def create_vector_store(model_name: str) -> tuple[VectorStore, int]:
    """A FAISS-backed ``VectorStore`` using a verified, real embedding model."""
    store = VectorStore(backend="faiss", dimension=384)
    if store.embedder is None:
        raise EmbeddingUnavailableError("Semantica could not create an embedding generator.")
    store.embedder.set_text_model("fastembed", model_name)
    method = store.embedder.get_text_method()
    if method != "fastembed":
        raise EmbeddingUnavailableError(
            f"Embedding model '{model_name}' did not load (active method: {method}). "
            "The first run downloads it and needs internet access."
        )
    dimension = int(np.asarray(store.embed("dimension probe")).shape[-1])
    if dimension != store.dimension:
        store = VectorStore(backend="faiss", dimension=dimension)
        store.embedder.set_text_model("fastembed", model_name)
    return store, dimension


class VectorIndex:
    def __init__(self, settings: Settings) -> None:
        self._model = settings.embedding_model
        self.path: Path = settings.vector_dir / "chunks"
        self.store, self.dimension = create_vector_store(self._model)
        if (self.path / "index.bin").exists():
            self.store.load(str(self.path))

    # ------------------------------------------------------------------ write
    def replace_document(self, document_id: int, chunks: list[Chunk]) -> None:
        """Swap a document's chunks (FAISS skips ids that already exist)."""
        self._delete(document_id)
        if chunks:
            texts = [chunk.text for chunk in chunks]
            vectors = self.store.embed_batch(texts)
            self.store.store_vectors(
                vectors,
                metadata=[
                    # ContextRetriever reads "content" from vector metadata.
                    {**chunk.model_dump(), "content": chunk.text}
                    for chunk in chunks
                ],
                ids=[chunk.chunk_id for chunk in chunks],
            )
        self._save()

    def remove_document(self, document_id: int) -> None:
        self._delete(document_id)
        self._save()

    def clear(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        self.store, self.dimension = create_vector_store(self._model)

    def _delete(self, document_id: int) -> None:
        ids = [c["chunk_id"] for c in self.chunks() if c["document_id"] == document_id]
        if ids:
            self.store.delete_vectors(ids)

    def _save(self) -> None:
        if self.count() or (self.path / "index.bin").exists():
            self.store.save(str(self.path))

    # ------------------------------------------------------------------ read
    def count(self) -> int:
        try:
            return int(self.store.count())
        except Exception:  # raised while no index has been created yet
            return 0

    def chunks(self) -> list[dict]:
        total = self.count()
        if not total:
            return []
        return [dict(row["metadata"]) for row in self.store.scan_vectors(offset=0, limit=total)]

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self.store.embed(text), dtype=np.float32).reshape(-1)

    def similarities(self, query: str, texts: list[str]) -> list[float]:
        """Cosine similarity of ``query`` to each text (same embedding model)."""
        if not texts:
            return []
        query_vector = self.embed(query)
        matrix = np.asarray(self.store.embed_batch(texts), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
        return [float(v) for v in matrix @ query_vector / np.where(norms == 0, 1, norms)]

    def search(self, question: str, k: int = 4) -> list[Evidence]:
        total = self.count()
        if not total:
            return []
        results = self.store.search_vectors(self.embed(question), k=min(k, total))
        evidence = []
        for result in results:
            metadata = result["metadata"]
            evidence.append(
                Evidence(
                    kind="chunk",
                    text=metadata["text"],
                    score=round(1.0 - float(result["distance"]) / 2.0, 4),
                    document_ids=[metadata["document_id"]],
                    filenames=[metadata["filename"]],
                    location=metadata["location"],
                    reference=metadata["chunk_id"],
                )
            )
        return sorted(evidence, key=lambda e: e.score or 0.0, reverse=True)

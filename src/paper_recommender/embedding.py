from __future__ import annotations

import hashlib
import inspect
import re

import numpy as np

from paper_recommender.oai import OaiRecord


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")


def embedding_text(record: OaiRecord) -> str:
    return "\n".join(
        [
            record.title or "",
            record.abstract or "",
            " ".join(record.categories),
        ]
    )


class HashingTextEmbedder:
    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed_record(self, record: OaiRecord) -> np.ndarray:
        return self.embed_text(embedding_text(record))

    def embed_records(self, records: list[OaiRecord]) -> np.ndarray:
        return self.embed_texts([embedding_text(record) for record in records])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_text(text) for text in texts], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


class SentenceTransformerTextEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str = "auto",
        model_loader=None,
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_device(device)
        self._model = _load_sentence_transformer(model_name, model_loader, self.device)
        self.dimensions = _embedding_dimensions(self._model)

    def embed_record(self, record: OaiRecord) -> np.ndarray:
        return self.embed_text(embedding_text(record))

    def embed_records(self, records: list[OaiRecord]) -> np.ndarray:
        return self.embed_texts([embedding_text(record) for record in records])

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


def make_embedder(
    backend: str = "sentence-transformers",
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    dimensions: int = 256,
    model_loader=None,
):
    if backend == "sentence-transformers":
        return SentenceTransformerTextEmbedder(model_name, device=device, model_loader=model_loader)
    if backend == "hashing":
        return HashingTextEmbedder(dimensions=dimensions)
    raise ValueError(f"unknown embedder backend: {backend}")


def _load_sentence_transformer(model_name: str, model_loader, device: str):
    if model_loader is not None:
        if _model_loader_accepts_device(model_loader):
            return model_loader(model_name, device=device)
        return model_loader(model_name)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for the default embedder. "
            "Install the embed extra or run with --embedder hashing for smoke tests."
        ) from exc
    return SentenceTransformer(model_name, device=device)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model_loader_accepts_device(model_loader) -> bool:
    try:
        parameters = inspect.signature(model_loader).parameters
    except (TypeError, ValueError):
        return False
    return "device" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _embedding_dimensions(model) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))

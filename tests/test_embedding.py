import numpy as np
import pytest

from paper_recommender.embedding import (
    DEFAULT_MODEL_NAME,
    HashingTextEmbedder,
    SentenceTransformerTextEmbedder,
    embedding_text,
    make_embedder,
)
from paper_recommender.oai import OaiRecord


def test_embedding_text_combines_title_abstract_and_categories() -> None:
    record = OaiRecord(
        arxiv_id="1706.03762",
        oai_datestamp="2024-01-01",
        deleted=False,
        title="Attention",
        abstract="Transformer models",
        categories=("cs.CL", "cs.LG"),
        published_date="2017-06-12",
        updated_date=None,
    )

    assert embedding_text(record) == "Attention\nTransformer models\ncs.CL cs.LG"


def test_hashing_text_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingTextEmbedder(dimensions=32)

    first = embedder.embed_text("attention transformer language")
    second = embedder.embed_text("attention transformer language")

    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), np.float32(1.0))


def test_hashing_text_embedder_scores_overlap_above_unrelated_text() -> None:
    embedder = HashingTextEmbedder(dimensions=64)
    query = embedder.embed_text("attention transformer language model")
    related = embedder.embed_text("transformer language representation")
    unrelated = embedder.embed_text("galaxy telescope redshift")

    assert float(query @ related) > float(query @ unrelated)


def test_hashing_text_embedder_returns_zero_for_empty_text() -> None:
    vector = HashingTextEmbedder(dimensions=16).embed_text("   ")

    assert np.allclose(vector, np.zeros(16, dtype=np.float32))


def test_sentence_transformer_text_embedder_uses_configured_model() -> None:
    loaded_names: list[str] = []

    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, texts, normalize_embeddings, convert_to_numpy, show_progress_bar):
            assert texts == ["a paper", "another paper"]
            assert normalize_embeddings is True
            assert convert_to_numpy is True
            assert show_progress_bar is False
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def load_model(name: str):
        loaded_names.append(name)
        return FakeModel()

    embedder = SentenceTransformerTextEmbedder("BAAI/bge-small-en-v1.5", model_loader=load_model)

    vectors = embedder.embed_texts(["a paper", "another paper"])

    assert loaded_names == ["BAAI/bge-small-en-v1.5"]
    assert embedder.dimensions == 3
    assert vectors.dtype == np.float32
    assert np.allclose(vectors[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_make_embedder_defaults_to_bge_small_model() -> None:
    loaded_names: list[str] = []

    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 384

    def load_model(name: str):
        loaded_names.append(name)
        return FakeModel()

    embedder = make_embedder("sentence-transformers", model_loader=load_model)

    assert isinstance(embedder, SentenceTransformerTextEmbedder)
    assert loaded_names == [DEFAULT_MODEL_NAME]
    assert embedder.dimensions == 384


def test_sentence_transformer_text_embedder_prefers_new_dimension_method() -> None:
    class FakeModel:
        def get_embedding_dimension(self) -> int:
            return 384

        def get_sentence_embedding_dimension(self) -> int:
            raise AssertionError("legacy dimension method should not be called")

    embedder = SentenceTransformerTextEmbedder(
        "BAAI/bge-small-en-v1.5",
        model_loader=lambda _name: FakeModel(),
    )

    assert embedder.dimensions == 384


def test_make_embedder_can_return_hashing_fallback() -> None:
    embedder = make_embedder("hashing", dimensions=16)

    assert isinstance(embedder, HashingTextEmbedder)
    assert embedder.dimensions == 16


def test_make_embedder_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown embedder backend"):
        make_embedder("unknown")

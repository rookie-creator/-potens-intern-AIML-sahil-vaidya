"""
embedder.py
-----------
Phase 5: Embeddings

Converts chunk text into dense vector embeddings for semantic search.

Model choice: BAAI/bge-small-en-v1.5
  - 384-dimensional embeddings, ~130MB — fast on CPU, no GPU required.
  - Strong retrieval performance for its size (outperforms larger, older
    models like all-MiniLM on most retrieval benchmarks, e.g. MTEB).
  - BGE models are trained with a "query prefix" convention: queries should
    be prefixed with "Represent this sentence for searching relevant
    passages: " for best retrieval quality, while the passages/chunks being
    indexed do NOT need a prefix. This is handled by embed_queries() below,
    separately from embed_chunks().

This module is deliberately kept separate from build_index.py:
  - embedder.py  -> knows how to turn text into vectors (model logic only)
  - build_index.py -> orchestrates loading chunks, embedding them, and
                       writing the FAISS index + metadata to disk

This separation means retriever.py can import embed_queries() directly at
query time without needing to know anything about FAISS index construction.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None  # lazy-loaded singleton so the model is only loaded once per process


def get_model() -> SentenceTransformer:
    """Load (or return cached) embedding model."""
    global _model
    if _model is None:
        print(f"[embedder] Loading model: {MODEL_NAME} (first load may take a moment)...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[embedder] Model loaded. Embedding dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_chunks(texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of chunk texts (passages) for indexing.
    No query prefix is applied — passages are embedded as-is per BGE convention.

    Returns a numpy array of shape (len(texts), 384), L2-normalized so that
    cosine similarity == dot product (required for FAISS IndexFlatIP).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2 normalize -> enables cosine sim via dot product
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def embed_queries(queries: List[str], show_progress: bool = False) -> np.ndarray:
    """
    Embed user queries at retrieval time.
    Applies the BGE query prefix, which measurably improves retrieval
    accuracy compared to embedding the raw query.
    """
    model = get_model()
    prefixed = [QUERY_PREFIX + q for q in queries]
    embeddings = model.encode(
        prefixed,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def embed_single_query(query: str) -> np.ndarray:
    """Convenience wrapper for embedding a single query string."""
    return embed_queries([query])[0]


if __name__ == "__main__":
    # Quick manual test: run `python ingestion/embedder.py` from the project root
    import json

    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    sample_texts = [c["text"] for c in chunks[:5]]
    print(f"Embedding {len(sample_texts)} sample chunks...")
    vectors = embed_chunks(sample_texts, show_progress=False)
    print(f"Output shape: {vectors.shape}")
    print(f"Vector norm (should be ~1.0 due to normalization): {np.linalg.norm(vectors[0]):.4f}")

    # Sanity check: embed a query and compare similarity to the 5 sample chunks
    query_vec = embed_single_query("How many casual leaves do employees get?")
    similarities = vectors @ query_vec  # dot product == cosine similarity (both normalized)
    print("\nSimilarity of query to each sample chunk:")
    for i, (chunk, sim) in enumerate(zip(chunks[:5], similarities)):
        print(f"  [{sim:.4f}] {chunk['document']} / {chunk['section']}")
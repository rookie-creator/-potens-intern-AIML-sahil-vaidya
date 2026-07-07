"""
build_index.py
--------------
Offline ingestion pipeline for the RAG system.

Pipeline:

documents/
    ↓
loader.py
    ↓
chunker.py
    ↓
embedder.py
    ↓
FAISS Index
    ↓
metadata.pkl

Run:
    python ingestion/build_index.py
"""

import pickle
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.append(".")

from ingestion.loader import load_all_documents
from ingestion.chunker import chunk_documents, save_chunks
from ingestion.embedder import (
    embed_chunks,
    embed_single_query,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

DOCUMENTS_DIR = Path("documents")
DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")

CHUNKS_PATH = DATA_DIR / "chunks.json"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"

# --------------------------------------------------
# Build Index
# --------------------------------------------------


def build_index():

    print("=" * 60)
    print("Building FAISS Document Index")
    print("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)
    VECTORSTORE_DIR.mkdir(exist_ok=True)

    # --------------------------------------------------
    # STEP 1
    # --------------------------------------------------

    print("\n[1/4] Loading documents...")

    records = load_all_documents(str(DOCUMENTS_DIR))

    if len(records) == 0:
        raise RuntimeError(
            "No documents were loaded.\n"
            "Ensure documents/ contains valid PDFs."
        )

    print(f"Loaded {len(records)} page records.")

    # --------------------------------------------------
    # STEP 2
    # --------------------------------------------------

    print("\n[2/4] Chunking...")

    chunks = chunk_documents(records)

    if len(chunks) == 0:
        raise RuntimeError("Chunker produced zero chunks.")

    save_chunks(chunks, str(CHUNKS_PATH))

    print(f"Generated {len(chunks)} chunks.")

    # --------------------------------------------------
    # STEP 3
    # --------------------------------------------------

    print("\n[3/4] Creating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    vectors = embed_chunks(
        texts,
        show_progress=True
    )

    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)

    vectors = vectors.astype("float32")

    if len(vectors) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    print(f"Embedding Dimension : {vectors.shape[1]}")
    print(f"Total Embeddings    : {vectors.shape[0]}")

    # --------------------------------------------------
    # STEP 4
    # --------------------------------------------------

    print("\n[4/4] Building FAISS index...")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    metadata = {
        i: chunks[i]
        for i in range(len(chunks))
    }

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("\nIndex Successfully Built")
    print("-" * 50)
    print(f"Vectors Stored : {index.ntotal}")
    print(f"Dimension      : {dimension}")
    print(f"FAISS Index    : {FAISS_INDEX_PATH}")
    print(f"Metadata File  : {METADATA_PATH}")

    return index, metadata


# --------------------------------------------------
# Sanity Check
# --------------------------------------------------


def sanity_check(index, metadata):

    print("\n")
    print("=" * 60)
    print("Running Retrieval Test")
    print("=" * 60)

    query = "How many casual leaves do employees get?"

    query_vector = embed_single_query(query)

    query_vector = query_vector.reshape(1, -1).astype("float32")

    top_k = 3

    scores, indices = index.search(query_vector, top_k)

    print(f"\nQuery : {query}\n")

    for rank, (idx, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1,
    ):

        chunk = metadata[int(idx)]

        print(f"Result #{rank}")
        print(f"Similarity : {score:.4f}")
        print(f"Document   : {chunk['document']}")
        print(f"Page       : {chunk['page']}")
        print(f"Section    : {chunk.get('section','Unknown')}")
        print(f"Chunk ID   : {chunk.get('chunk_id','N/A')}")
        print()

        preview = chunk["text"][:250].replace("\n", " ")

        print(preview)
        print("-" * 60)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    index, metadata = build_index()

    sanity_check(index, metadata)
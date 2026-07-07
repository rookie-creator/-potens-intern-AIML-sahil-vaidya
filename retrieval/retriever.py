"""
retriever.py
------------
Online retrieval layer for the RAG system.

Responsibilities
----------------
1. Load FAISS index and metadata
2. Embed incoming queries
3. Retrieve candidate chunks from FAISS
4. Rerank retrieved chunks
5. Filter low-confidence results
6. Build LLM-ready context
7. Return citation-ready metadata

This module never calls the LLM.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from ingestion.embedder import embed_single_query
from retrieval.reranker import rerank

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

VECTORSTORE_DIR = Path("vectorstore")

FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"

DEFAULT_TOP_K = 5

# Number of candidates retrieved from FAISS before reranking
CANDIDATE_K = 20

# Below this score we consider retrieval unreliable
DEFAULT_SIMILARITY_THRESHOLD = 0.55


# ------------------------------------------------------------------
# Retriever
# ------------------------------------------------------------------

class Retriever:

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.threshold = similarity_threshold

        self.index = None
        self.metadata = None

        self._load()

    # --------------------------------------------------------------

    def _load(self):

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {self.metadata_path}"
            )

        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[Retriever] Loaded {self.index.ntotal} vectors.")

    # --------------------------------------------------------------

    def search(
        self,
        question: str,
        k: int = CANDIDATE_K,
    ) -> List[Dict]:

        query_vector = embed_single_query(question)

        query_vector = (
            np.array(query_vector)
            .reshape(1, -1)
            .astype("float32")
        )

        scores, indices = self.index.search(query_vector, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            chunk = dict(self.metadata[int(idx)])

            chunk["score"] = float(score)

            results.append(chunk)

        return results

    # --------------------------------------------------------------

    def has_enough_context(
        self,
        results: List[Dict],
    ) -> bool:

        if len(results) == 0:
            return False

        # Use FAISS similarity score for hallucination gate
        return results[0]["score"] >= self.threshold

    # --------------------------------------------------------------

    def build_context(
        self,
        results: List[Dict],
    ) -> str:

        sections = []

        for chunk in results:

            section = f"""
Document : {chunk['document']}
Page     : {chunk['page']}
Section  : {chunk.get('section', 'Unknown')}
Chunk ID : {chunk.get('chunk_id', 'N/A')}

{chunk['text']}
"""

            sections.append(section.strip())

        return "\n\n" + ("\n" + "-" * 70 + "\n").join(sections)

    # --------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        k: int = DEFAULT_TOP_K,
    ) -> Dict:

        # Step 1: Retrieve candidate chunks from FAISS
        candidate_results = self.search(
            question=question,
            k=CANDIDATE_K,
        )

        # Step 2: Rerank them
        results = rerank(
            question=question,
            results=candidate_results,
            top_k=k,
        )

        # Step 3: Check confidence
        enough = self.has_enough_context(results)

        return {
            "question": question,
            "enough_context": enough,
            "context": self.build_context(results) if results else "",
            "results": results,
        }


# ------------------------------------------------------------------
# Singleton Loader
# ------------------------------------------------------------------

_retriever = None


def get_retriever():

    global _retriever

    if _retriever is None:
        _retriever = Retriever()

    return _retriever


# ------------------------------------------------------------------
# Convenience API
# ------------------------------------------------------------------

def retrieve(
    question: str,
    k: int = DEFAULT_TOP_K,
):

    retriever = get_retriever()

    return retriever.retrieve(
        question=question,
        k=k,
    )


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":

    retriever = Retriever()

    query = "How many casual leaves are employees allowed?"

    output = retriever.retrieve(query)

    print("=" * 80)

    print("Enough Context :", output["enough_context"])

    print()

    for i, chunk in enumerate(output["results"], start=1):

        print(
            f"{i}. "
            f"{chunk['document']} | "
            f"Page {chunk['page']} | "
            f"Vector Score: {chunk['score']:.4f} | "
            f"Reranker Score: {chunk['reranker_score']:.4f}"
        )

    print("\n")

    print(output["context"])
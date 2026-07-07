"""
reranker.py
-----------
Cross-encoder reranker for the RAG system.

Purpose
-------
The vector retriever retrieves chunks based on embedding similarity.
The reranker then re-scores those retrieved chunks by jointly considering
the user's query and each chunk.

Pipeline

Question
    ↓
FAISS Retriever (Top-K)
    ↓
CrossEncoder Reranker
    ↓
Sorted Chunks
    ↓
LLM

This module is currently independent.
It will be integrated into retriever.py later.
"""

from typing import List, Dict

from sentence_transformers import CrossEncoder

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-reranker-base"


# --------------------------------------------------
# Reranker
# --------------------------------------------------

class Reranker:

    def __init__(self):

        print(f"[Reranker] Loading model: {MODEL_NAME}")

        self.model = CrossEncoder(MODEL_NAME)

        print("[Reranker] Model loaded.")

    def rerank(
        self,
        question: str,
        results: List[Dict],
        top_k: int = None,
    ) -> List[Dict]:
        """
        Re-score retrieved chunks.

        Parameters
        ----------
        question : str
            User query

        results : List[Dict]
            Retrieved chunks from FAISS

        top_k : int
            Number of chunks to return

        Returns
        -------
        List[Dict]
            Sorted chunks
        """

        if len(results) == 0:
            return []

        pairs = [
            (question, chunk["text"])
            for chunk in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for chunk, score in zip(results, scores):

            item = dict(chunk)

            item["reranker_score"] = float(score)

            reranked.append(item)

        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked


# --------------------------------------------------
# Singleton
# --------------------------------------------------

_reranker = None


def get_reranker():

    global _reranker

    if _reranker is None:
        _reranker = Reranker()

    return _reranker


# --------------------------------------------------
# Convenience Function
# --------------------------------------------------

def rerank(
    question: str,
    results: List[Dict],
    top_k: int = None,
) -> List[Dict]:

    model = get_reranker()

    return model.rerank(
        question=question,
        results=results,
        top_k=top_k,
    )


# --------------------------------------------------
# Demo
# --------------------------------------------------

if __name__ == "__main__":

    sample_results = [

        {
            "document": "leave_policy.pdf",
            "page": 1,
            "section": "Casual Leave",
            "text": "Employees receive 12 casual leaves annually."
        },

        {
            "document": "security_policy.pdf",
            "page": 2,
            "section": "Passwords",
            "text": "Passwords must be changed every ninety days."
        },

        {
            "document": "remote_work.pdf",
            "page": 1,
            "section": "Eligibility",
            "text": "Employees may work remotely up to three days per week."
        }

    ]

    query = "How many casual leaves do employees get?"

    ranked = rerank(
        question=query,
        results=sample_results,
    )

    print("\nResults\n")

    for i, chunk in enumerate(ranked, start=1):

        print(f"{i}. {chunk['document']}")
        print(f"   Score   : {chunk['reranker_score']:.4f}")
        print(f"   Section : {chunk['section']}")
        print()
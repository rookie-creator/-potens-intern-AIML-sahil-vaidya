"""
citations.py
------------
Citation builder for the RAG system.

Responsibilities
----------------
Convert retrieved chunks into citation objects that can be returned
alongside the generated answer.

Expected Citation Format
------------------------
[
    {
        "file": "leave_policy.pdf",
        "page": 1,
        "chunk": "chunk_0025",
        "snippet": "Employees receive 12 casual leaves annually..."
    }
]

This module does NOT:
- Retrieve documents
- Generate answers
- Call the LLM
"""

from typing import Dict, List


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

SNIPPET_LENGTH = 180


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _create_snippet(text: str, max_length: int = SNIPPET_LENGTH) -> str:
    """
    Create a short snippet from the chunk text.

    Parameters
    ----------
    text : str
        Original chunk text

    max_length : int
        Maximum snippet length

    Returns
    -------
    str
        Shortened snippet
    """

    text = " ".join(text.split())

    if len(text) <= max_length:
        return text

    return text[:max_length].rstrip() + "..."


# ------------------------------------------------------------------
# Citation Builder
# ------------------------------------------------------------------

def build_citations(results: List[Dict]) -> List[Dict]:
    """
    Convert retrieved chunks into citation objects.

    Parameters
    ----------
    results : List[Dict]
        Retriever output

    Returns
    -------
    List[Dict]
        Citation list
    """

    citations = []

    seen = set()

    for chunk in results:

        key = (
            chunk["document"],
            chunk["page"],
            chunk["chunk_id"],
        )

        # Avoid duplicate citations
        if key in seen:
            continue

        seen.add(key)

        citations.append({

            "file": chunk["document"],

            "page": chunk["page"],

            "chunk": chunk["chunk_id"],

            "snippet": _create_snippet(
                chunk["text"]
            )

        })

    return citations


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":

    sample_results = [

        {
            "document": "leave_policy.pdf",
            "page": 1,
            "chunk_id": "chunk_0025",
            "section": "Casual Leave",
            "text": (
                "Employees receive 12 casual leaves annually. "
                "Casual leave may be taken in units of half a day "
                "or more with prior approval."
            ),
            "score": 0.91
        },

        {
            "document": "remote_work.pdf",
            "page": 2,
            "chunk_id": "chunk_0014",
            "section": "Eligibility",
            "text": (
                "Employees may work remotely up to three days "
                "per week subject to manager approval."
            ),
            "score": 0.82
        }

    ]

    citations = build_citations(sample_results)

    print("\nGenerated Citations\n")

    for citation in citations:

        print(citation)
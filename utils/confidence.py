"""
confidence.py
-------------

Utility functions for estimating confidence in RAG answers.

Purpose
-------
The LLM itself does not provide a reliable confidence score.
Instead, we estimate confidence using retrieval quality.

Current Strategy
----------------
Confidence is based on the similarity scores of the retrieved chunks.

Higher similarity
    -> Higher confidence

Lower similarity
    -> Lower confidence

This confidence score can later be used to:
    - Display confidence in the UI
    - Trigger Human-in-the-loop review
    - Reject hallucination-prone answers
"""

from __future__ import annotations

from typing import Dict, List

# ------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------

HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.60


# ------------------------------------------------------------------
# Confidence Calculation
# ------------------------------------------------------------------

def calculate_confidence(results: List[Dict]) -> float:
    """
    Calculate confidence from retrieved chunks.

    Parameters
    ----------
    results : List[Dict]

    Returns
    -------
    float
        Confidence score between 0.0 and 1.0
    """

    if not results:
        return 0.0

    scores = [
        chunk["score"]
        for chunk in results
        if "score" in chunk
    ]

    if not scores:
        return 0.0

    confidence = sum(scores) / len(scores)

    confidence = max(0.0, min(1.0, confidence))

    return round(confidence, 3)


# ------------------------------------------------------------------
# Confidence Label
# ------------------------------------------------------------------

def confidence_label(score: float) -> str:
    """
    Convert numerical confidence into
    a human-readable label.
    """

    if score >= HIGH_CONFIDENCE:
        return "High"

    if score >= MEDIUM_CONFIDENCE:
        return "Medium"

    return "Low"


# ------------------------------------------------------------------
# Human-in-the-loop Decision
# ------------------------------------------------------------------

def requires_human_review(score: float) -> bool:
    """
    Decide whether a human should review
    the answer before it is accepted.
    """

    return score < MEDIUM_CONFIDENCE


# ------------------------------------------------------------------
# Complete Confidence Report
# ------------------------------------------------------------------

def build_confidence_report(results: List[Dict]) -> Dict:
    """
    Build a complete confidence report.

    Returns
    -------
    {
        confidence: float,
        level: str,
        human_review: bool
    }
    """

    score = calculate_confidence(results)

    return {
        "confidence": score,
        "level": confidence_label(score),
        "human_review": requires_human_review(score),
    }


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":

    retrieved_chunks = [

        {"score": 0.91},

        {"score": 0.86},

        {"score": 0.79},

        {"score": 0.74},

        {"score": 0.69},
    ]

    report = build_confidence_report(retrieved_chunks)

    print("=" * 60)

    print("Confidence Report")

    print(report)
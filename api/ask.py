"""
ask.py
------
FastAPI endpoint for Document Question Answering.

Pipeline

POST /ask

        ↓

Receive Question

        ↓

Retriever (FAISS)

        ↓

Reranker

        ↓

Enough Context?

        ↓

Gemini

        ↓

Citation Builder

        ↓

Return JSON Response

Responsibilities
----------------
1. Accept user question
2. Retrieve relevant chunks
3. Prevent hallucination
4. Generate answer using Gemini
5. Return answer with citations
"""

from __future__ import annotations

import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from retrieval.retriever import retrieve
from utils.citations import build_citations
from utils.confidence import calculate_confidence
from utils.confidence import build_confidence_report

# ------------------------------------------------------------
# Load Environment Variables
# ------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found inside .env"
    )

genai.configure(api_key=API_KEY)

# ------------------------------------------------------------
# Gemini Model
# ------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash"

model = genai.GenerativeModel(MODEL_NAME)

# ------------------------------------------------------------
# FastAPI Router
# ------------------------------------------------------------

router = APIRouter(
    prefix="/api",
    tags=["Question Answering"],
)

# ------------------------------------------------------------
# Request Model
# ------------------------------------------------------------

class AskRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=2,
        description="Question asked by the user"
    )

# ------------------------------------------------------------
# Citation Model
# ------------------------------------------------------------

class Citation(BaseModel):

    file: str

    page: int

    chunk: str

    snippet: str

# ------------------------------------------------------------
# Response Model
# ------------------------------------------------------------

class AskResponse(BaseModel):

    success: bool

    question: str

    answer: str

    citations: List[Citation]
    
    confidence: float
    
    confidence_level: str

    human_review: bool

    retrieved_chunks: int

# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

PROMPT_TEMPLATE = """
You are an AI assistant that answers questions ONLY using the supplied document context.

Rules:

1. Use ONLY the supplied context.
2. Never use outside knowledge.
3. Never guess.
4. Never invent information.
5. If the answer is not present in the context, reply exactly with:

The uploaded documents do not contain enough information to answer this question.

6. Keep answers concise.
7. Do not mention "context" in your answer.

==========================================================
DOCUMENT CONTEXT
==========================================================

{context}

==========================================================
QUESTION
==========================================================

{question}

==========================================================
ANSWER
==========================================================
"""

# ------------------------------------------------------------
# Gemini Generation
# ------------------------------------------------------------

def generate_answer(
    question: str,
    context: str,
) -> str:
    """
    Generate answer from Gemini using retrieved context.
    """

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    try:

        response = model.generate_content(prompt)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gemini Error : {str(e)}"
        )

    if response is None:

        raise HTTPException(
            status_code=500,
            detail="Gemini returned an empty response."
        )

    if not hasattr(response, "text"):

        raise HTTPException(
            status_code=500,
            detail="Gemini response contains no text."
        )

    answer = response.text.strip()

    if answer == "":
        raise HTTPException(
            status_code=500,
            detail="Gemini returned an empty answer."
        )

    return answer


# ------------------------------------------------------------
# POST /ask
# ------------------------------------------------------------

@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask questions over the indexed documents"
)
def ask_question(request: AskRequest):

    """
    Complete RAG Pipeline

    Question
        ↓
    Retriever
        ↓
    Reranker
        ↓
    Enough Context?
        ↓
    Gemini
        ↓
    Citations
        ↓
    JSON Response
    """

    question = request.question.strip()

    if question == "":
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    retrieval = retrieve(question)

    if not retrieval["enough_context"]:

        return AskResponse(

            success=False,

            question=question,

            answer=(
                "The uploaded documents do not contain enough "
                "information to answer this question."
            ),

            citations=[],
            
            confidence=0.0,
            
            confidence_level="Low",

            human_review=True,

            retrieved_chunks=0,
        )

    # --------------------------------------------------------
    # Retrieved Context
    # --------------------------------------------------------

    context = retrieval["context"]

    retrieved_chunks = retrieval["results"]

    # --------------------------------------------------------
    # Generate LLM Answer
    # --------------------------------------------------------

    answer = generate_answer(
        question=question,
        context=context,
    )

    # --------------------------------------------------------
    # Build Citations
    # --------------------------------------------------------

    citations = build_citations(
        retrieved_chunks
    )
    confidence = build_confidence_report(retrieved_chunks)

    # --------------------------------------------------------
    # Return Response
    # --------------------------------------------------------

    return AskResponse(

        success=True,

        question=question,

        answer=answer,

        citations=citations,
        
        confidence=confidence["confidence"],

        confidence_level=confidence["level"],
        
        human_review=confidence["human_review"],

        retrieved_chunks=len(retrieved_chunks),
    )
    
    
# ------------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------------

@router.get(
    "/health",
    summary="Health check for the Question Answering API"
)
def health_check():

    return {
        "status": "healthy",
        "service": "RAG Question Answering API"
    }


# ------------------------------------------------------------
# Root Endpoint
# ------------------------------------------------------------

@router.get(
    "/",
    summary="API Information"
)
def root():

    return {
        "message": "RAG Document Question Answering API",
        "endpoints": [
            "/api/ask",
            "/api/health"
        ]
    }


# ------------------------------------------------------------
# Local Testing
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Testing ask.py")
    print("=" * 60)

    sample_question = "How many casual leaves do employees get?"

    try:

        retrieval = retrieve(sample_question)

        print("\nQuestion:")
        print(sample_question)

        print("\nEnough Context:")
        print(retrieval["enough_context"])

        if retrieval["enough_context"]:

            answer = generate_answer(
                sample_question,
                retrieval["context"]
            )

            citations = build_citations(
                retrieval["results"]
            )

            print("\nAnswer:")
            print(answer)

            print("\nCitations:")

            for citation in citations:

                print(
                    f"- {citation['file']} "
                    f"(Page {citation['page']}, "
                    f"{citation['chunk']})"
                )

        else:

            print(
                "\nThe uploaded documents do not contain enough "
                "information to answer this question."
            )

    except Exception as e:

        print(f"\nError: {e}")

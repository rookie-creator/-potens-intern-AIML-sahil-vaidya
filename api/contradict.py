"""
contradict.py
-------------

FastAPI endpoint for document contradiction detection.

Pipeline

POST /api/contradict

        ↓

Receive Request

        ↓

Load all chunks of Document A

        ↓

Load all chunks of Document B

        ↓

Gemini Comparison

        ↓

Return structured response

Responsibilities
----------------
1. Compare two indexed documents
2. Focus comparison on a user-provided topic
3. Detect conflicts
4. Explain reasoning
5. Return evidence from both documents
"""

from __future__ import annotations

import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from retrieval.retriever import get_document_chunks

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found inside .env"
    )

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

model = genai.GenerativeModel(MODEL_NAME)

# ------------------------------------------------------------
# Router
# ------------------------------------------------------------

router = APIRouter(
    prefix="/api",
    tags=["Contradiction Detection"],
)

# ------------------------------------------------------------
# Request Model
# ------------------------------------------------------------

class ContradictRequest(BaseModel):

    document_1: str = Field(
        ...,
        description="First document filename"
    )

    document_2: str = Field(
        ...,
        description="Second document filename"
    )

    topic: str = Field(
        ...,
        description="Topic to compare"
    )

# ------------------------------------------------------------
# Citation Model
# ------------------------------------------------------------

class Citation(BaseModel):

    file: str

    page: int

    chunk: str

# ------------------------------------------------------------
# Response Model
# ------------------------------------------------------------

class ContradictResponse(BaseModel):

    success: bool

    document_1: str

    document_2: str

    topic: str

    conflict: bool

    reasoning: str

    citations: List[Citation]

# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

PROMPT_TEMPLATE = """
You are comparing two company documents.

Compare ONLY the supplied documents.

Focus ONLY on this topic:

{topic}

Rules

1. Ignore every other topic.

2. Decide whether the documents contradict each other.

3. If they agree, say they agree.

4. If one document omits the topic,
do NOT call it a contradiction.

5. Never use outside knowledge.

Return ONLY this format.

Conflict:
True or False

Reason:
<short explanation>

=========================================================
DOCUMENT A
=========================================================

{document_a}

=========================================================
DOCUMENT B
=========================================================

{document_b}
"""

# ------------------------------------------------------------
# Build Document Text
# ------------------------------------------------------------

def build_document_context(chunks: List[dict]) -> str:
    """
    Convert chunk list into a readable document.
    """

    sections = []

    for chunk in chunks:

        section = f"""
Page : {chunk['page']}
Section : {chunk.get('section','Unknown')}

{chunk['text']}
"""

        sections.append(section.strip())

    return "\n\n".join(sections)

# ------------------------------------------------------------
# Gemini Comparison
# ------------------------------------------------------------

def compare_documents(
    topic: str,
    document_a: str,
    document_b: str,
) -> str:

    prompt = PROMPT_TEMPLATE.format(

        topic=topic,

        document_a=document_a,

        document_b=document_b,
    )

    try:

        response = model.generate_content(prompt)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    if response is None or not hasattr(response, "text"):

        raise HTTPException(
            status_code=500,
            detail="Gemini returned an empty response."
        )

    return response.text.strip()

# ------------------------------------------------------------
# Parse Gemini Response
# ------------------------------------------------------------

def parse_response(response: str):

    """
    Convert Gemini output into:

    conflict (bool)
    reasoning (str)
    """

    lines = response.splitlines()

    conflict = False

    reasoning = response.strip()

    for line in lines:

        lower = line.lower().strip()

        if lower.startswith("conflict"):

            if "true" in lower:
                conflict = True
            else:
                conflict = False

        elif lower.startswith("reason"):

            reasoning = line.split(":", 1)[1].strip()

    return conflict, reasoning


# ------------------------------------------------------------
# POST /contradict
# ------------------------------------------------------------

@router.post(
    "/contradict",
    response_model=ContradictResponse,
    summary="Compare two indexed documents"
)
def contradict(request: ContradictRequest):

    document_1 = request.document_1.strip()

    document_2 = request.document_2.strip()

    topic = request.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )

    # --------------------------------------------------------
    # Load document chunks
    # --------------------------------------------------------

    chunks_a = get_document_chunks(document_1)

    chunks_b = get_document_chunks(document_2)

    if len(chunks_a) == 0:

        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_1}"
        )

    if len(chunks_b) == 0:

        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_2}"
        )

    # --------------------------------------------------------
    # Build document context
    # --------------------------------------------------------

    context_a = build_document_context(chunks_a)

    context_b = build_document_context(chunks_b)

    # --------------------------------------------------------
    # Gemini Comparison
    # --------------------------------------------------------

    response = compare_documents(

        topic=topic,

        document_a=context_a,

        document_b=context_b,
    )

    conflict, reasoning = parse_response(response)

    # --------------------------------------------------------
    # Build Evidence
    # --------------------------------------------------------

    citations = []

    for chunk in chunks_a:

        if topic.lower() in chunk["text"].lower():

            citations.append({

                "file": chunk["document"],

                "page": chunk["page"],

                "chunk": chunk["chunk_id"],
            })

    for chunk in chunks_b:

        if topic.lower() in chunk["text"].lower():

            citations.append({

                "file": chunk["document"],

                "page": chunk["page"],

                "chunk": chunk["chunk_id"],
            })

    # remove duplicates

    unique = []

    seen = set()

    for citation in citations:

        key = (
            citation["file"],
            citation["page"],
            citation["chunk"],
        )

        if key not in seen:

            seen.add(key)

            unique.append(citation)

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return ContradictResponse(

        success=True,

        document_1=document_1,

        document_2=document_2,

        topic=topic,

        conflict=conflict,

        reasoning=reasoning,

        citations=unique,
    )
    
# ------------------------------------------------------------
# Health Endpoint
# ------------------------------------------------------------

@router.get(
    "/health",
    summary="Health check"
)
def health():

    return {
        "status": "healthy",
        "service": "Document Contradiction API"
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
        "service": "Document Contradiction API",
        "endpoint": "/api/contradict"
    }


# ------------------------------------------------------------
# Local Test
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("Testing Document Contradiction")
    print("=" * 70)

    document_a = "leave_policy.pdf"
    document_b = "employee_handbook.pdf"
    topic = "Casual Leave"

    try:

        chunks_a = get_document_chunks(document_a)

        chunks_b = get_document_chunks(document_b)

        print(f"\nDocument A Chunks : {len(chunks_a)}")
        print(f"Document B Chunks : {len(chunks_b)}")

        context_a = build_document_context(chunks_a)

        context_b = build_document_context(chunks_b)

        response = compare_documents(
            topic=topic,
            document_a=context_a,
            document_b=context_b,
        )

        conflict, reasoning = parse_response(response)

        print("\nConflict :", conflict)

        print("\nReasoning :")

        print(reasoning)

    except Exception as e:

        print("\nError")

        print(str(e))    
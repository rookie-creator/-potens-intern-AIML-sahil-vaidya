"""
app.py
------

Main FastAPI application.

Responsibilities
----------------
1. Create FastAPI app
2. Register API routers
3. Enable CORS
4. Expose root endpoint
5. Ready for Streamlit frontend

Run:

    uvicorn app:app --reload

Swagger UI:

    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ask import router as ask_router
from api.contradict import router as contradict_router

# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------

app = FastAPI(
    title="RAG Document Question Answering System",
    description=(
        "A Retrieval-Augmented Generation (RAG) system built with "
        "FAISS, Sentence Transformers, Gemini, and FastAPI."
    ),
    version="1.0.0",
)

# ------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------

# Allows Streamlit (or any frontend) to call the API.
# During development we allow all origins.
# Restrict this in production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Register Routers
# ------------------------------------------------------------------

app.include_router(ask_router)
app.include_router(contradict_router)

# ------------------------------------------------------------------
# Root Endpoint
# ------------------------------------------------------------------

@app.get("/", tags=["Home"])
def home():
    return {
        "message": "RAG Document QA API is running.",
        "version": "1.0.0",
        "documentation": "/docs",
        "available_endpoints": [
            "/api/ask",
            "/api/contradict",
            "/api/health",
        ],
    }

# ------------------------------------------------------------------
# Health Endpoint
# ------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "application": "RAG Document QA System",
    }

# ------------------------------------------------------------------
# Run Directly
# ------------------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
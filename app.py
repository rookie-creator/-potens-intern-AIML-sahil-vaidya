from fastapi import FastAPI

app = FastAPI(title="RAG System")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG System API is running"}

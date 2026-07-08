#  RAG Document Question Answering System

A **Retrieval-Augmented Generation (RAG)** system that enables users to ask natural language questions over a collection of documents and receive accurate, citation-backed answers. The system also supports contradiction detection, multilingual queries, confidence scoring, and a professional web interface.

---

#  Project Description

The **RAG Document Question Answering System** is an AI-powered application that allows users to ask questions across multiple documents using semantic search and Large Language Models.

Instead of relying solely on an LLM, the system first retrieves the most relevant document chunks using **FAISS Vector Search** and **Sentence Transformers**, reranks them using a **Cross-Encoder**, and finally generates grounded responses using **Google Gemini**.

To prevent hallucinations, the system only answers questions supported by the indexed documents. If sufficient information is unavailable, it explicitly informs the user instead of generating unsupported responses.

The project also includes:

-  Citation-backed answers
-  Document contradiction detection
-  Multilingual query support
-  Confidence scoring
-  FastAPI backend
-  Professional Streamlit dashboard

---

#  Features

-  Semantic document search
-  Retrieval-Augmented Generation (RAG)
-  Citation-backed answers
-  Document contradiction detection
-  Multilingual query support
-  FAISS vector search
-  Cross-Encoder reranking
-  Confidence scoring
-  Hallucination prevention
-  FastAPI REST API
-  Modern Streamlit interface

---

# 🛠️ Tech Stack

## Backend

- Python
- FastAPI
- FAISS
- Sentence Transformers
- Google Gemini API
- LangChain Text Splitters

## Frontend

- Streamlit
- HTML/CSS (Glassmorphism UI)

## AI Models

- **BAAI/bge-small-en-v1.5** – Embeddings
- **Cross-Encoder (ms-marco-MiniLM-L-6-v2)** – Reranker
- **Google Gemini 1.5 Flash** – Answer Generation

---

#  Folder Structure

```text
RAG-System/
├── app.py
├── ui.py
├── requirements.txt
├── README.md
│
├── documents/
│   ├── company_policy.pdf
│   ├── employee_handbook.pdf
│   ├── leave_policy.pdf
│   ├── remote_work.pdf
│   ├── security_policy.pdf
│   └── code_of_conduct.pdf
│
├── ingestion/
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   └── build_index.py
│
├── retrieval/
│   ├── retriever.py
│   └── reranker.py
│
├── api/
│   ├── ask.py
│   └── contradict.py
│
├── utils/
│   ├── citations.py
│   ├── confidence.py
│   └── translation.py
│
├── vectorstore/
│   ├── faiss.index
│   └── metadata.pkl
│
├── data/
│   └── chunks.json
│
└── evaluation/
    └── eval.json
```

---

#  Getting Started

## 1. Clone the Repository

```bash
git clone <your-github-repository>
```

---

## 2. Navigate to the Project Directory

```bash
cd RAG-System
```

---

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

---

## 4. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Build the FAISS Index

```bash
python ingestion/build_index.py
```

---

## 7. Start the FastAPI Server

```bash
uvicorn app:app --reload
```

---

## 8. Launch the Streamlit Dashboard

```bash
streamlit run ui.py
```

---

#  Chunking Strategy

The system uses a **heading-aware recursive chunking strategy** to maximize retrieval quality.

### Step 1 — Document Loading

Documents are parsed page-by-page using **PyMuPDF**, preserving page metadata for accurate citations.

### Step 2 — Section Detection

Regular expressions detect numbered headings such as:

```text
1. Casual Leave
2. Sick Leave
3. Earned Leave
```

Each heading becomes a semantic section.

### Step 3 — Recursive Chunking

Each section is split using LangChain's `RecursiveCharacterTextSplitter`.

Configuration:

- **Chunk Size:** 500 words
- **Chunk Overlap:** 100 words

This preserves semantic context while improving retrieval performance.

---

#  Embedding Strategy

Each chunk is embedded using:

```text
BAAI/bge-small-en-v1.5
```

Key Features:

- 384-dimensional embeddings
- L2-normalized vectors
- Optimized for semantic retrieval
- Fast CPU inference

---

#  Vector Store

The project uses **FAISS IndexFlatIP** for semantic similarity search.

Because all embeddings are normalized, cosine similarity is equivalent to inner-product similarity.

The vector database stores:

- Document embeddings
- Page numbers
- Chunk IDs
- Section names
- Citation metadata

---

# Retrieval Pipeline

```text
User Question
      │
      ▼
Embedding
      │
      ▼
FAISS Retrieval
      │
      ▼
Top-20 Chunks
      │
      ▼
Cross Encoder Reranker
      │
      ▼
Top-5 Chunks
      │
      ▼
Google Gemini
      │
      ▼
Final Answer
```

---

#  API Endpoints

## Ask Questions

```http
POST /api/ask
```

Returns:

- Answer
- Citations
- Confidence Score
- Retrieved Chunks

---

## Contradiction Detection

```http
POST /api/contradict
```

Returns:

- Conflict Status
- Reasoning
- Supporting Citations

---

## Health Check

```http
GET /health
```

---

# Hallucination Prevention

The system strictly follows a Retrieval-Augmented Generation pipeline.

If the retrieved context is insufficient, the application responds with:

> "The provided documents do not contain enough information to answer this question."

This ensures that unsupported answers are never generated.

---

#  Confidence Scoring

Each response includes a confidence score calculated using:

- FAISS similarity score
- Cross-Encoder reranker score
- Number of retrieved chunks

Low-confidence responses are flagged for potential human review.

---

#  Multilingual Support

The application supports multilingual queries through:

- Language Detection (`langdetect`)
- Translation (`deep-translator`)

Workflow:

```text
User Query
      │
Language Detection
      │
Translation (if required)
      │
Retrieval
      │
Gemini
      │
Translate Back
```

---

#  What Makes This Project Different

##  Citation-Based Answers

Every answer includes:

- Source document
- Page number
- Chunk ID
- Supporting snippet

making responses completely traceable.

---

##  Contradiction Detection

The system compares two documents on a specific topic and identifies conflicting information using semantic reasoning.

---

##  Reranking Layer

Instead of relying solely on FAISS similarity, retrieved chunks are reranked using a Cross-Encoder model for improved answer quality.

---

##  Hallucination Prevention

The application never fabricates information. If the documents lack sufficient evidence, it explicitly declines to answer.

---

# Future Improvements

- Hybrid Search (BM25 + Dense Retrieval)
- ChromaDB / pgvector integration
- OCR support for scanned PDFs
- User authentication
- Document upload interface
- Conversation memory
- Evaluation dashboard
- Kubernetes deployment

---

#  AI Use Log

| Tool | Approx. Messages | Purpose |
|------|------------------|---------|
| Google Gemini | ~15 | Assisted in answer generation during RAG inference and contradiction reasoning. |
| ChatGPT (GPT-5.5) | ~150 | Assisted with project architecture, chunking strategy, FAISS pipeline, FastAPI APIs, Streamlit dashboard, documentation, debugging, confidence scoring, reranking integration, and README preparation. |

---

# Screenshots

*(Add screenshots of your application before submission.)*

- Home Page
- Ask Question Interface
- Answer with Citations
- Contradiction Checker
- FastAPI Swagger Documentation

---

#  License

This project was developed for educational purposes as part of an AI/LLM assignment demonstrating Retrieval-Augmented Generation (RAG), semantic search, and document question answering.

"""
loader.py
---------
Phase 3: Document Ingestion

Responsible for reading raw files (PDF, DOCX, TXT, MD) from the `documents/`
folder and converting them into a uniform, page-aware format that the rest
of the pipeline (chunker -> embedder -> retriever) can consume.

Output format (list of dicts), one entry per page/section:
{
    "document": "leave_policy.pdf",
    "page": 1,
    "text": "Leave Policy\nApplicable to all full-time employees..."
}

For TXT/MD (which have no real pages), the whole file is treated as page 1.
"""

import os
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
import docx  # python-docx


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_pdf(filepath: str) -> List[Dict]:
    """Extract text page-by-page from a PDF using PyMuPDF."""
    records = []
    doc_name = os.path.basename(filepath)

    with fitz.open(filepath) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                # Skip genuinely empty pages (e.g. blank separator pages)
                continue
            records.append({
                "document": doc_name,
                "page": page_number,
                "text": text,
            })

    return records


def load_docx(filepath: str) -> List[Dict]:
    """
    Extract text from a DOCX file.

    DOCX has no native concept of a 'page' (pagination is a rendering detail,
    not part of the file format), so we approximate pages by grouping
    paragraphs. This is a reasonable approximation for chunking purposes;
    exact page numbers would require rendering the document.
    """
    doc_name = os.path.basename(filepath)
    document = docx.Document(filepath)

    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)

    if not full_text:
        return []

    # Approximate: ~40 lines per "page" for downstream page-number metadata
    lines_per_page = 40
    records = []
    for i in range(0, len(paragraphs), lines_per_page):
        chunk_lines = paragraphs[i:i + lines_per_page]
        page_number = (i // lines_per_page) + 1
        records.append({
            "document": doc_name,
            "page": page_number,
            "text": "\n".join(chunk_lines),
        })

    return records


def load_text(filepath: str) -> List[Dict]:
    """Extract text from a plain TXT or Markdown file. Treated as a single page."""
    doc_name = os.path.basename(filepath)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    if not text:
        return []

    return [{
        "document": doc_name,
        "page": 1,
        "text": text,
    }]


def load_document(filepath: str) -> List[Dict]:
    """Dispatch to the correct loader based on file extension."""
    ext = Path(filepath).suffix.lower()

    if ext == ".pdf":
        return load_pdf(filepath)
    elif ext == ".docx":
        return load_docx(filepath)
    elif ext in (".txt", ".md"):
        return load_text(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext} ({filepath})")


def load_all_documents(documents_dir: str = "documents") -> List[Dict]:
    """
    Load every supported document in `documents_dir`.

    Returns a flat list of {document, page, text} records across all files,
    ready to be passed into the chunker (Phase 4).
    """
    documents_path = Path(documents_dir)
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    all_records = []
    skipped = []

    for filepath in sorted(documents_path.iterdir()):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(filepath.name)
            continue

        try:
            records = load_document(str(filepath))
            all_records.extend(records)
            print(f"[loader] Loaded {filepath.name}: {len(records)} page(s)")
        except Exception as e:
            print(f"[loader] FAILED to load {filepath.name}: {e}")

    if skipped:
        print(f"[loader] Skipped unsupported files: {skipped}")

    return all_records


if __name__ == "__main__":
    # Quick manual test: run `python ingestion/loader.py` from the project root
    records = load_all_documents("documents")
    print(f"\nTotal pages extracted: {len(records)}")
    if records:
        print("\nSample record:")
        sample = records[0]
        print(f"  document : {sample['document']}")
        print(f"  page     : {sample['page']}")
        print(f"  text     : {sample['text'][:150]}...")
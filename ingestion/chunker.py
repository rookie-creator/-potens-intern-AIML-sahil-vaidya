"""
chunker.py
----------
Phase 4: Chunking (the most important part of the pipeline)

Strategy: recursive chunking, respecting document structure top-down:

    Heading  -->  Paragraph  -->  Sentence  -->  Chunk

Rationale (for your README):
Naively splitting every N characters cuts sentences and sections in half,
which destroys retrieval quality — a chunk might start mid-sentence with no
context about which policy section it belongs to. Instead, we:

  1. First split each page's text into SECTIONS using heading patterns
     (e.g. "1. Casual Leave", "2. Sick Leave") detected via regex. This keeps
     semantically related content grouped together and lets us attach a
     `section` label to every chunk's metadata.
  2. Within each section, we use LangChain's RecursiveCharacterTextSplitter,
     which tries to split on paragraph breaks first, then sentences, then
     words — only falling back to a hard character cut as a last resort.
  3. Chunk size: ~500 words, overlap: ~100 words. The overlap means a chunk
     boundary that lands mid-idea still carries the preceding context into
     the next chunk, so retrieval doesn't lose meaning at the edges.

Output: list of dicts, each with {chunk_id, document, page, section, text},
saved to data/chunks.json.
"""

import json
import re
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- Config ----
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 100

# Matches headings like "1. Casual Leave", "2. Sick Leave", "10. Third-Party Software"
HEADING_PATTERN = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][^\n]{2,80})\s*$", re.MULTILINE)


def word_len(text: str) -> int:
    """Length function counting words instead of characters, so chunk size
    config (500 words) matches what we actually pass to the splitter."""
    return len(text.split())


def split_into_sections(text: str) -> List[Dict]:
    """
    Split a page's raw text into (section_title, section_text) pairs using
    numbered heading detection. Any text before the first heading is kept
    under section "Introduction".
    """
    matches = list(HEADING_PATTERN.finditer(text))

    if not matches:
        # No headings found on this page — treat the whole page as one section
        return [{"section": "General", "text": text.strip()}]

    sections = []

    # Text before the first heading (e.g. title, intro line)
    if matches[0].start() > 0:
        intro = text[:matches[0].start()].strip()
        if intro:
            sections.append({"section": "Introduction", "text": intro})

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append({"section": title, "text": section_text})

    return sections


def chunk_documents(records: List[Dict]) -> List[Dict]:
    """
    Take loader output (list of {document, page, text}) and produce
    heading-aware, overlapping chunks with full metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_WORDS,
        chunk_overlap=CHUNK_OVERLAP_WORDS,
        length_function=word_len,
        separators=["\n\n", "\n", ". ", " ", ""],  # paragraph -> line -> sentence -> word -> char
    )

    chunks = []
    chunk_counter = 1

    for record in records:
        document = record["document"]
        page = record["page"]
        page_text = record["text"]

        sections = split_into_sections(page_text)

        for section in sections:
            section_title = section["section"]
            section_text = section["text"]

            if word_len(section_text) <= CHUNK_SIZE_WORDS:
                # Section fits in a single chunk — no need to split further
                pieces = [section_text]
            else:
                pieces = splitter.split_text(section_text)

            for piece in pieces:
                if not piece.strip():
                    continue
                chunk_id = f"chunk_{chunk_counter:04d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "document": document,
                    "page": page,
                    "section": section_title,
                    "text": piece.strip(),
                })
                chunk_counter += 1

    return chunks


def save_chunks(chunks: List[Dict], output_path: str = "data/chunks.json") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"[chunker] Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    # Quick manual test: run `python ingestion/chunker.py` from the project root
    import sys
    sys.path.append(".")
    from ingestion.loader import load_all_documents

    records = load_all_documents("documents")
    chunks = chunk_documents(records)
    save_chunks(chunks, "data/chunks.json")

    print(f"\nTotal chunks created: {len(chunks)}")
    print("\nSample chunk:")
    sample = chunks[3] if len(chunks) > 3 else chunks[0]
    print(json.dumps(sample, indent=2)[:500])

    # Word count sanity check across all chunks
    word_counts = [word_len(c["text"]) for c in chunks]
    print(f"\nWord count range: min={min(word_counts)}, max={max(word_counts)}, "
          f"avg={sum(word_counts)/len(word_counts):.0f}")
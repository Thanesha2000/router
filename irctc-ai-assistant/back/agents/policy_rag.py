"""
policy_rag.py
Location: back/agents/policy_rag.py

Exact folder structure this file expects:
    back/agents/
        policy_rag.py          <- THIS FILE
        data/
            irctc_policies/
                raw_pdfs/      <- PDFs go here (required)
        vectorstore/
            faiss_index/       <- auto-created on first --build

TXT fallback folder is OPTIONAL. If it doesn't exist, PDFs-only is fine.
"""

from __future__ import annotations

import os
import re
import time
import pickle

import numpy as np
import faiss
import fitz                          # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
from google import genai

load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))



_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PDF_DIR   = os.path.join(_THIS_DIR, "..", "data", "irctc_policies", "raw_pdfs")
TXT_DIR   = os.path.join(_THIS_DIR, "..", "data", "irctc_policies", "txt_fallbacks")
INDEX_DIR = os.path.join(_THIS_DIR, "..", "vectorstore", "faiss_index")


INDEX_PATH  = os.path.join(INDEX_DIR, "policy.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50
TOP_K         = 3

HIGH_CONF = 0.80    # score >= 0.80  → return chunk directly (no Gemini)
LOW_CONF  = 0.35    # score <  0.35  → return None (caller uses general_agent)
                    # 0.35 <= score < 0.80 → Gemini summarizes

GEMINI_MODEL        = "gemini-2.5-flash"
GEMINI_MAX_ATTEMPTS = 3
GEMINI_BACKOFF      = [2.0, 4.0, 8.0]   # wait before attempt 2, 3

# ─────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETONS
# Loaded once. Reused on every subsequent request.
# ─────────────────────────────────────────────────────────────────────

_embedding_model : SentenceTransformer | None = None
_faiss_index     : faiss.Index         | None = None
_chunk_store     : list[dict]          | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print("[policy_rag] Loading embedding model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[policy_rag] Embedding model ready.")
    return _embedding_model


# ─────────────────────────────────────────────────────────────────────
# DOCUMENT LOADING
# ─────────────────────────────────────────────────────────────────────

def _extract_pdf_pages(filepath: str) -> list[tuple[int, str]]:
    """
    Opens a PDF with PyMuPDF and returns [(page_number, text), ...].
    Page numbers are 1-indexed.
    Pages with no extractable text (scanned/image-only) are skipped.
    """
    results: list[tuple[int, str]] = []
    doc = fitz.open(filepath)
    try:
        for i in range(len(doc)):
            text = doc[i].get_text()
            if text and text.strip():
                results.append((i + 1, text))
    finally:
        doc.close()
    return results


def _load_and_chunk_all() -> list[dict]:
    """
    Loads every PDF from PDF_DIR and every .txt from TXT_DIR.
    TXT_DIR is optional — if folder doesn't exist, it's silently skipped.

    Returns a flat list of chunk dicts:
        {"text": str, "source": str, "page": int | None}

    page is an int for PDF chunks (which page it came from).
    page is None for TXT chunks (no page concept for plain text).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []

    # ── Load PDFs ─────────────────────────────────────────────────────
    if not os.path.isdir(PDF_DIR):
        print(f"[policy_rag] PDF_DIR not found: {PDF_DIR}")
    else:
        pdf_files = [f for f in sorted(os.listdir(PDF_DIR))
                     if f.lower().endswith(".pdf")]

        if not pdf_files:
            print(f"[policy_rag] No PDF files found in: {PDF_DIR}")
        else:
            print(f"[policy_rag] Found {len(pdf_files)} PDF(s) in {PDF_DIR}")
            for filename in pdf_files:
                filepath = os.path.join(PDF_DIR, filename)
                try:
                    pages = _extract_pdf_pages(filepath)
                    page_chunks = 0
                    for page_num, page_text in pages:
                        for piece in splitter.split_text(page_text):
                            chunks.append({
                                "text":   piece,
                                "source": filename,
                                "page":   page_num,
                            })
                            page_chunks += 1
                    print(f"  [+] {filename}: {len(pages)} pages → {page_chunks} chunks")
                except Exception as e:
                    print(f"  [!] {filename}: could not read — {e}")

    # ── Load TXT fallbacks (optional) ─────────────────────────────────
    if not os.path.isdir(TXT_DIR):
        # This is fine — txt_fallbacks is optional
        pass
    else:
        txt_files = [f for f in sorted(os.listdir(TXT_DIR))
                     if f.lower().endswith(".txt")]

        if txt_files:
            print(f"[policy_rag] Found {len(txt_files)} TXT fallback(s) in {TXT_DIR}")
            for filename in txt_files:
                filepath = os.path.join(TXT_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                    txt_chunks = 0
                    for piece in splitter.split_text(text):
                        chunks.append({
                            "text":   piece,
                            "source": filename,
                            "page":   None,
                        })
                        txt_chunks += 1
                    print(f"  [+] {filename}: {txt_chunks} chunks")
                except Exception as e:
                    print(f"  [!] {filename}: could not read — {e}")

    return chunks


# ─────────────────────────────────────────────────────────────────────
# PHASE 1 — BUILD INDEX (run once)
# ─────────────────────────────────────────────────────────────────────

def build_vector_db() -> None:
    """
    Loads all documents → chunks → embeds → saves FAISS index.

    Run:   python policy_rag.py --build
    Re-run whenever you add or replace PDFs in raw_pdfs/.
    """
    print("\n" + "=" * 55)
    print("  Building Policy Vector Database")
    print(f"  PDF source : {PDF_DIR}")
    print(f"  TXT source : {TXT_DIR} (optional)")
    print(f"  Index dest : {INDEX_DIR}")
    print("=" * 55)

    chunks = _load_and_chunk_all()

    if not chunks:
        print("\n[policy_rag] No chunks produced.")
        print(f"  Check that PDFs exist in: {PDF_DIR}")
        return

    print(f"\n[policy_rag] Total chunks: {len(chunks)}")
    print("[policy_rag] Generating embeddings...")

    model      = _get_embedding_model()
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # required — must match query time
        show_progress_bar=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\n[policy_rag] Done. {index.ntotal} vectors saved.")
    print(f"  Index : {INDEX_PATH}")
    print(f"  Chunks: {CHUNKS_PATH}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────
# PHASE 2A — LOAD INDEX INTO MEMORY
# ─────────────────────────────────────────────────────────────────────

def _ensure_index_loaded() -> bool:
    """
    Loads FAISS index + chunk metadata into module-level cache.
    First call reads from disk. Every later call returns immediately.
    Returns False only if the index was never built.
    """
    global _faiss_index, _chunk_store

    if _faiss_index is not None and _chunk_store is not None:
        return True

    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        print("[policy_rag] Index not found. Run: python policy_rag.py --build")
        return False

    _faiss_index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        _chunk_store = pickle.load(f)

    print(f"[policy_rag] Index loaded: {_faiss_index.ntotal} vectors in cache.")
    return True


# ─────────────────────────────────────────────────────────────────────
# PHASE 2B — RETRIEVE
# ─────────────────────────────────────────────────────────────────────

def _retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Embeds query, searches FAISS, returns top-k chunks with scores.
    normalize_embeddings=True MUST match what was used in build_vector_db().
    """
    model        = _get_embedding_model()
    query_vector = model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = _faiss_index.search(query_vector, k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _chunk_store[idx]
        results.append({
            "text":   chunk["text"],
            "source": chunk["source"],
            "page":   chunk.get("page"),
            "score":  float(score),
        })
    return results


# ─────────────────────────────────────────────────────────────────────
# PHASE 2C — FORMAT DIRECT ANSWER (no Gemini)
# ─────────────────────────────────────────────────────────────────────

def _citation(chunks: list[dict]) -> str:
    """
    Builds deduplicated source citation string.
    PDF  → "refund_rules.pdf (page 3)"
    TXT  → "tatkal_rules.txt"
    """
    seen: list[str] = []
    for c in chunks:
        label = (
            f"{c['source']} (page {c['page']})"
            if c.get("page") is not None
            else c["source"]
        )
        if label not in seen:
            seen.append(label)
    return ", ".join(seen)


def _format_direct(chunks: list[dict]) -> str:
    """
    Returns raw chunk text + citation. No LLM involved.
    Used for: score >= HIGH_CONF, AND as Gemini fallback.
    """
    body = "\n\n".join(c["text"].strip() for c in chunks)
    return f"{body}\n\n📄 Source: {_citation(chunks)}"


# ─────────────────────────────────────────────────────────────────────
# PHASE 2D — GEMINI WITH RETRY
# ─────────────────────────────────────────────────────────────────────

def _build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        "[{source}{page}]\n{text}".format(
            source=c["source"],
            page=f", page {c['page']}" if c.get("page") is not None else "",
            text=c["text"],
        )
        for c in chunks
    )
    return (
        "You are an IRCTC railway policy assistant.\n\n"
        "Answer the question using ONLY the policy context below.\n"
        "Do not use outside knowledge.\n"
        "Do not invent numbers, amounts, or rules not in the context.\n"
        "Be concise (3-5 sentences).\n\n"
        f"POLICY CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer using only the context above."
    )


def _call_gemini(prompt: str) -> str | None:
    """
    Calls Gemini up to GEMINI_MAX_ATTEMPTS times with exponential backoff.
    Returns response text on success.
    Returns None on ANY failure — 503, 429, timeout, empty, exception.
    NEVER returns error message strings.
    """
    for attempt in range(GEMINI_MAX_ATTEMPTS):
        try:
            response = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            if response and response.text and response.text.strip():
                return response.text.strip()
        except Exception:
            pass

        if attempt < GEMINI_MAX_ATTEMPTS - 1:
            time.sleep(GEMINI_BACKOFF[attempt])

    return None


# ─────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────

def answer_policy_query(user_question: str) -> str | None:
    """
    Called by chatbot.py for every policy_query intent.

    Returns:
        str  — direct chunk text  (score >= 0.80, no Gemini call)
        str  — Gemini summary     (0.35 <= score < 0.80)
               or direct chunk fallback if Gemini fails
        None — score < 0.35 (chatbot.py will call general_agent instead)

    NEVER raises an exception.
    NEVER returns API error messages or connection failure strings.
    """
    if not user_question or not user_question.strip():
        return None

    user_question = user_question.strip()

    if not _ensure_index_loaded():
        return None

    chunks = _retrieve(user_question, k=TOP_K)
    if not chunks:
        return None

    best_score = chunks[0]["score"]
    print(f"[policy_rag] best_score={best_score:.3f} | source={chunks[0]['source']}")

    # Tier 1 — high confidence: return chunk directly, zero Gemini calls
    if best_score >= HIGH_CONF:
        print(f"[policy_rag] Tier 1 (>= {HIGH_CONF}): direct chunk")
        return _format_direct(chunks[:1])

    # Tier 3 — low confidence: no relevant policy found
    if best_score < LOW_CONF:
        print(f"[policy_rag] Tier 3 (< {LOW_CONF}): returning None")
        return None

    # Tier 2 — medium confidence: try Gemini, fall back to direct chunk
    print(f"[policy_rag] Tier 2 ({LOW_CONF}–{HIGH_CONF}): calling Gemini...")
    prompt      = _build_rag_prompt(user_question, chunks)
    gemini_text = _call_gemini(prompt)

    if gemini_text:
        print("[policy_rag] Gemini responded successfully.")
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", gemini_text)
        return f"{clean}\n\n📄 Source: {_citation(chunks)}"

    # Gemini failed — direct chunk fallback (never an error message)
    print("[policy_rag] Gemini failed. Using direct chunk fallback.")
    return _format_direct(chunks)


# ─────────────────────────────────────────────────────────────────────
# CLI — build only
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--build" in sys.argv or not os.path.exists(INDEX_PATH):
        build_vector_db()
    else:
        print(f"Index already exists at: {INDEX_PATH}")
        print("Use --build to force a rebuild.")
print("\nDEBUG")
print("_THIS_DIR =", _THIS_DIR)
print("PDF_DIR   =", PDF_DIR)
print("Exists?   =", os.path.exists(PDF_DIR))
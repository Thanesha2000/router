# pdf_loader.py
#
# WHAT THIS FILE DOES:
#   Extracts text from PDF files, page by page, and returns it in a
#   structured format that policy_rag.py can chunk and embed.
#
# WHY THIS IS A SEPARATE FILE (not inside policy_rag.py):
#   Separation of concerns. policy_rag.py's job is retrieval — embedding,
#   FAISS search, prompt construction. PDF parsing is a completely
#   different concern: it's about recovering clean text from a binary
#   file format that was designed for visual layout, not logical text
#   flow. Keeping them separate means:
#     1. You can test PDF extraction alone, without touching FAISS/Gemini.
#     2. If you ever swap PDF libraries (PyMuPDF -> pdfplumber), only
#        this file changes. policy_rag.py never knows the difference.
#     3. Anyone reading policy_rag.py doesn't need to understand PDF
#        internals to understand retrieval logic.
#
# LIBRARY CHOICE: PyMuPDF (imported as `fitz`)
#   Why PyMuPDF and not PyPDF2/pypdf or LangChain's PyPDFLoader:
#     - PyMuPDF is faster and has more reliable text extraction for
#       single-column government/circular-style PDFs (which is what
#       IRCTC/Railway Board documents are).
#     - It gives direct page-level access — page.get_text() — without
#       wrapping it in another library's Document abstraction. We build
#       our OWN lightweight dict format instead, matching what
#       policy_rag.py already expects from _load_policy_documents().
#     - It has zero dependency overlap with langchain-community (which
#       is sunset, per your existing architecture notes).

import os
import fitz  # PyMuPDF


def load_pdf_documents(data_dir: str) -> list[dict]:
    """
    Reads every .pdf file in data_dir and extracts text page-by-page.

    Args:
        data_dir (str): Folder containing .pdf files (e.g. raw_pdfs/)

    Returns:
        list[dict]: One dict PER PAGE, not per file. Each dict has:
            {
                "text": "<extracted page text>",
                "source": "refund_rules.pdf",
                "page": 3,
                "doc_type": "pdf"
            }

    WHY ONE DICT PER PAGE (not one dict per whole PDF file):
        This is the key design decision that enables page-level source
        attribution (Section 9 of the architecture doc). If we extracted
        a PDF as one giant text blob, we'd lose track of which page a
        chunk came from once we split it. By keeping text page-by-page
        from the start, every downstream chunk inherits its page number.

    WHAT HAPPENS IF EXTRACTION FAILS (scanned/image-only PDFs):
        Per Section 4 of the architecture doc, some Railway Board PDFs
        are scanned images, not real text. PyMuPDF will return an empty
        or near-empty string for these pages. We detect this (very short
        text) and skip the page with a warning — we do NOT add OCR in V1.
        This keeps the function honest: it never silently returns garbage.
    """
    page_records = []

    if not os.path.isdir(data_dir):
        print(f"[pdf_loader] WARNING: directory not found: {data_dir}")
        return page_records

    pdf_filenames = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".pdf"))

    if not pdf_filenames:
        print(f"[pdf_loader] No PDF files found in {data_dir}")
        return page_records

    for filename in pdf_filenames:
        filepath = os.path.join(data_dir, filename)

        try:
            doc = fitz.open(filepath)
        except Exception as e:
            # A corrupted or unreadable PDF should not crash the whole
            # build — skip it, log it, move on to the next file.
            print(f"[pdf_loader] ❌ Could not open {filename}: {str(e)[:100]}")
            continue

        pages_extracted = 0
        pages_skipped = 0

        for page_number, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            cleaned_text = _clean_pdf_text(raw_text)

            # Heuristic: a real text page in an IRCTC policy circular has
            # at least a few dozen characters. Near-empty extraction means
            # this page is likely a scanned image with no embedded text.
            if len(cleaned_text.strip()) < 30:
                pages_skipped += 1
                continue

            page_records.append({
                "text": cleaned_text,
                "source": filename,
                "page": page_number,
                "doc_type": "pdf",
            })
            pages_extracted += 1

        doc.close()

        if pages_skipped > 0:
            print(f"[pdf_loader] ⚠ {filename}: extracted {pages_extracted} pages, "
                  f"skipped {pages_skipped} page(s) with no readable text "
                  f"(likely scanned images — see Section 4 notes on OCR).")
        else:
            print(f"[pdf_loader] ✅ {filename}: extracted {pages_extracted} pages")

    return page_records


def _clean_pdf_text(raw_text: str) -> str:
    """
    Cleans common PDF extraction artifacts before chunking.

    WHY THIS IS NEEDED (per Section 5 of the architecture doc):
        PyMuPDF gives you the text that was laid out for PRINTING, not
        for READING as prose. Two specific problems show up constantly
        in government circular PDFs:

        1. Hyphenated line breaks: a word split across a line wraps as
           "can-\ncellation" instead of "cancellation". If we don't fix
           this, the chunker and embedder see two broken half-words
           instead of one real word, hurting retrieval quality.

        2. Excess blank lines / repeated whitespace: headers, footers,
           and page numbers often leave behind stray newlines. Left
           uncleaned, these inflate chunk size without adding real
           content, wasting your CHUNK_SIZE budget on whitespace.

    WHAT THIS FUNCTION DELIBERATELY DOES NOT DO:
        It does not try to fix multi-column layout interleaving. Per
        Section 5 of the architecture doc, this is acceptable because
        IRCTC/Railway Board circulars are predominantly single-column.
        If you later ingest a genuinely two-column PDF and see garbled
        sentences, that's the signal to add column-aware extraction —
        not before, because it adds complexity you don't yet need.
    """
    import re

    text = raw_text

    # Fix hyphenated line-break words: "can-\ncellation" -> "cancellation"
    text = re.sub(r"-\n", "", text)

    # Collapse 3+ newlines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs into one space
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE TEST BLOCK
#
#  WHY THIS EXISTS: per the "build in isolation, test in isolation"
#  principle from Step 0 above. Run this file directly, BEFORE wiring it
#  into policy_rag.py, to confirm extraction actually works on your real
#  PDFs. If this test passes, you know any future bug is in policy_rag.py
#  logic, not in PDF extraction.
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Allow running as: python pdf_loader.py /path/to/raw_pdfs
    test_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "data", "irctc_policies", "raw_pdfs"
    )

    print(f"Testing PDF loader on: {test_dir}")
    print("=" * 65)

    pages = load_pdf_documents(test_dir)

    print(f"\nTotal pages extracted: {len(pages)}")
    print("-" * 65)

    for record in pages[:5]:   # show first 5 pages as a sanity check
        preview = record["text"][:150].replace("\n", " ")
        print(f"\nSource : {record['source']} (page {record['page']})")
        print(f"Preview: {preview}...")

    if len(pages) > 5:
        print(f"\n... and {len(pages) - 5} more pages.")

    print("\n" + "=" * 65)
    if pages:
        print("✅ PDF extraction working. Safe to wire into policy_rag.py")
    else:
        print("⚠ No pages extracted. Check that raw_pdfs/ contains valid PDFs.")
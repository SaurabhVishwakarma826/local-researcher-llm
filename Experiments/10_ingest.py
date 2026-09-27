"""
Phase 3 / Topic 3.1 — Ingestion: getting clean text out of a PDF.

Usage (quote the path if it has spaces):
    python 10_ingest.py 1 "data\\pdfs\\report.pdf"          # inventory: what is in this PDF?
    python 10_ingest.py 2 "data\\pdfs\\report.pdf" 5        # look at page 5 exactly as extracted
    python 10_ingest.py 3 "data\\pdfs\\report.pdf" 5        # find headers/footers, clean page 5
    python 10_ingest.py 4 "data\\pdfs\\report.pdf"          # find tables

Page numbers are 1-based, as shown in a PDF viewer.
"""

import re
import sys
from collections import Counter
from pathlib import Path

# Let experiments import our own app/ package from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf  # noqa: E402

from app import tokens  # noqa: E402
from app.prompt import sanitize  # noqa: E402

PREFILL_TOK_S = 66      # measured in Phase 1


# ---------------------------------------------------------------------------
# 1 — Inventory: pages, text, tokens, and pages with no text layer
# ---------------------------------------------------------------------------
def exp1_inventory(doc, path, _page):
    meta = doc.metadata or {}
    print(f"File:   {path.name}")
    print(f"Title:  {meta.get('title') or '(none)'}")
    print(f"Pages:  {doc.page_count}\n")

    print(f"{'page':>4} {'chars':>7} {'tokens':>7} {'images':>7}  note")
    print("-" * 45)
    total_tok = 0
    empty = []
    for i, page in enumerate(doc, 1):
        text = page.get_text()
        n_chars = len(text.strip())
        n_tok = tokens.count_text(text) if text else 0
        n_img = len(page.get_images())
        total_tok += n_tok
        note = ""
        if n_chars < 50:
            note = "NO TEXT LAYER (scanned?)" if n_img else "blank"
            empty.append(i)
        if i <= 40:
            print(f"{i:>4} {n_chars:>7} {n_tok:>7} {n_img:>7}  {note}")
    if doc.page_count > 40:
        print(f"  ... {doc.page_count - 40} more pages not shown")

    print(f"\nTotal Qwen tokens in the whole PDF: {total_tok:,}")
    print(f"Pages with no usable text: {len(empty)}  {empty[:20]}")
    print(f"\nIf you pasted the WHOLE PDF into one prompt: {total_tok:,} tokens,")
    print(f"~{total_tok / PREFILL_TOK_S / 60:.1f} minutes of prefill on your CPU "
          f"(and it {'does NOT' if total_tok > 16384 else 'does'} fit in num_ctx 16384).")
    print("That is why RAG sends only the few relevant chunks instead.")


# ---------------------------------------------------------------------------
# 2 — One page, exactly as extracted
# ---------------------------------------------------------------------------
def exp2_page(doc, _path, page_no):
    page = doc[page_no - 1]
    text = page.get_text()
    print(f"=== Page {page_no}: raw text, first 1500 chars (repr shows every \\n) ===")
    print(repr(text[:1500]))

    print(f"\n=== Page {page_no}: text BLOCKS in extraction order ===")
    print("x = distance from left edge, y = distance from top. Two columns show up")
    print("as two groups of x values. Check the ORDER: does the left column finish")
    print("before the right one starts, or do they alternate?\n")
    print(f"{'#':>3} {'x':>6} {'y':>6}  text")
    for b in page.get_text("blocks"):
        x0, y0, _x1, _y1, btext, bno, btype = b
        if btype != 0:          # 1 = image block
            print(f"{bno:>3} {x0:>6.0f} {y0:>6.0f}  [image]")
            continue
        print(f"{bno:>3} {x0:>6.0f} {y0:>6.0f}  {btext.strip().replace(chr(10), ' ')[:70]!r}")


# ---------------------------------------------------------------------------
# 3 — Headers/footers, and a cleaned page
# ---------------------------------------------------------------------------
def _norm(line: str) -> str:
    """'Page 12 of 80' and 'Page 13 of 80' should count as the same line."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_furniture(doc, edge_lines: int = 3, min_share: float = 0.5) -> set:
    """Lines near the top/bottom of a page that repeat on at least half the pages."""
    counts = Counter()
    for page in doc:
        lines = [l for l in page.get_text().splitlines() if l.strip()]
        edges = set(_norm(l) for l in lines[:edge_lines] + lines[-edge_lines:])
        counts.update(edges)
    threshold = max(3, int(doc.page_count * min_share))
    return {line for line, c in counts.items() if c >= threshold}


def clean_page(text: str, furniture: set, edge_lines: int = 3) -> str:
    lines = text.splitlines()
    non_empty = [i for i, l in enumerate(lines) if l.strip()]
    edge_idx = set(non_empty[:edge_lines] + non_empty[-edge_lines:])
    kept = [l for i, l in enumerate(lines) if not (i in edge_idx and _norm(l) in furniture)]
    out = "\n".join(kept)
    out = re.sub(r"(\w)-\n(\w)", r"\1\2", out)      # join words split across lines: "main-\ntenance"
    out = re.sub(r"[ \t]+", " ", out)               # collapse runs of spaces
    out = re.sub(r"\n{3,}", "\n\n", out)            # at most one blank line
    return sanitize(out).strip()                    # 1.3: strip fake control markers


def exp3_clean(doc, _path, page_no):
    furniture = find_furniture(doc)
    print(f"Repeated header/footer lines found (digits shown as #): {len(furniture)}")
    for f in sorted(furniture):
        print(f"   {f!r}")
    if not furniture:
        print("   (none — this PDF may not have headers/footers, or has fewer than 3 pages)")

    raw = doc[page_no - 1].get_text()
    cleaned = clean_page(raw, furniture)
    print(f"\nPage {page_no}: {tokens.count_text(raw)} tokens raw -> {tokens.count_text(cleaned)} tokens cleaned")
    print(f"\n=== cleaned text, first 1200 chars ===\n{cleaned[:1200]}")

    joined = re.findall(r"(\w+)-\n(\w+)", raw)
    if joined:
        print("\nWords joined across line breaks (check these are right!):")
        for a, b in joined[:10]:
            print(f"   {a}-/{b}  ->  {a}{b}")
        print("A real hyphenated word split at a line end ('quarter-/car') gets")
        print("wrongly joined into 'quartercar'. Cleaning always trades one error for another.")


# ---------------------------------------------------------------------------
# 4 — Tables
# ---------------------------------------------------------------------------
def exp4_tables(doc, _path, _page):
    limit = min(doc.page_count, 30)
    print(f"Searching for tables in the first {limit} pages (this can be slow)...\n")
    found = 0
    for i in range(limit):
        page = doc[i]
        for t in page.find_tables().tables:
            found += 1
            md = t.to_markdown()
            print(f"=== Table on page {i + 1}: {t.row_count} rows x {t.col_count} cols, "
                  f"{tokens.count_text(md)} tokens as markdown ===")
            print(md[:600])
            if found >= 3:
                break
        if found >= 3:
            break

    if not found:
        print("No tables detected. Either this PDF has none, or they are images")
        print("(which need OCR), or they use no ruling lines and the detector missed them.")
        return

    page_no = next(i + 1 for i in range(limit) if doc[i].find_tables().tables)
    plain = doc[page_no - 1].get_text()
    print(f"\n=== Same page {page_no} as PLAIN text (first 600 chars) ===")
    print(plain[:600])
    print("\nCompare: in plain text, can you still tell which number belongs to which")
    print("column? If not, a chunk containing it will be meaningless to Qwen.")


EXPERIMENTS = {"1": exp1_inventory, "2": exp2_page, "3": exp3_clean, "4": exp4_tables}

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in EXPERIMENTS:
        sys.exit(__doc__)
    path = Path(sys.argv[2])
    if not path.exists():
        sys.exit(f"File not found: {path}")
    doc = pymupdf.open(path)
    page_no = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if not 1 <= page_no <= doc.page_count:
        sys.exit(f"Page must be between 1 and {doc.page_count}")
    EXPERIMENTS[sys.argv[1]](doc, path, page_no)

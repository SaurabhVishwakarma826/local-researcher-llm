"""
app/ingest.py — PDF -> clean page texts, with file and page kept for citations.

Every rule here comes from a measurement in 10_ingest.py (3.1):
  1. Pages with no text layer are SKIPPED AND REPORTED  (scanned book: 4/4 pages empty, no error)
  2. Repeated headers/footers are removed               (waste rules: 3 lines found)
  3. Symbol-font bullets (\uf0b7, \uf0a7) become "-"    (page 5)
  4. Line-end hyphens KEEP the hyphen                   ("co-processing" was wrongly joined)
  5. Real tables are extracted as markdown              (payslip: plain text loses the columns)
  6. Fake tables are rejected                           (page 1 layout: empty cells; resume: paragraph cells)
  7. All text goes through sanitize()                   (1.3: fake control markers)

Usage from the project root:
    python -m app.ingest                          # every PDF in config.DOCS_DIR
    python -m app.ingest "data\\pdfs\\file.pdf"     # one file
    python -m app.ingest "data\\pdfs\\file.pdf" --page 6   # also print cleaned page 6
"""

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app import config
from app.prompt import sanitize

EDGE_LINES = 3          # how many lines at the top/bottom of a page can be header/footer
FURNITURE_SHARE = 0.5   # a line is furniture if it appears on at least this share of pages


@dataclass
class Page:
    source: str         # file name, for citations
    page: int           # 1-based, as in a PDF viewer
    text: str           # cleaned body text, then any tables as markdown
    n_tables: int


@dataclass
class Report:
    source: str
    pages_total: int
    pages_kept: int = 0
    skipped_pages: list = field(default_factory=list)     # no text layer
    tables_kept: int = 0
    tables_rejected: int = 0
    furniture: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def _norm(line: str) -> str:
    """'Page 12' and 'Page 13' count as the same line."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_furniture(doc) -> set:
    counts = Counter()
    for page in doc:
        lines = [l for l in page.get_text().splitlines() if l.strip()]
        counts.update(set(_norm(l) for l in lines[:EDGE_LINES] + lines[-EDGE_LINES:]))
    threshold = max(3, int(doc.page_count * FURNITURE_SHARE))
    return {line for line, c in counts.items() if c >= threshold}


def clean_text(text: str, furniture: set = frozenset()) -> str:
    lines = text.splitlines()
    non_empty = [i for i, l in enumerate(lines) if l.strip()]
    edges = set(non_empty[:EDGE_LINES] + non_empty[-EDGE_LINES:])
    lines = [l for i, l in enumerate(lines) if not (i in edges and _norm(l) in furniture)]
    out = "\n".join(lines)

    # Symbol-font bullets live in Unicode's "private use" range. At a line start they
    # are list bullets -> "- ". Anywhere else they are junk -> removed.
    out = re.sub(r"(?m)^[ \t]*[\ue000-\uf8ff][ \t]*", "- ", out)
    out = re.sub(r"[\ue000-\uf8ff]", "", out)

    out = re.sub(r"(\w)-\n(\w)", r"\1-\2", out)     # keep the hyphen: "co-\nprocessing" -> "co-processing"
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return sanitize(out).strip()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _cell(c) -> str:
    # Joining a cell's lines with a SPACE avoids glued words like "methanegenerationpotential".
    return " ".join((c or "").split()).replace("|", "/")


def is_real_table(rows: list, header_names: list) -> bool:
    cells = [c for row in rows for c in row]
    if not cells:
        return False
    empty = sum(1 for c in cells if not (c or "").strip())
    if empty / len(cells) > config.TABLE_MAX_EMPTY_RATIO:
        return False
    placeholder = sum(1 for h in header_names if re.fullmatch(r"Col\d+", h or ""))
    if header_names and placeholder / len(header_names) >= 0.5:
        return False
    # A "table" whose cells are paragraphs is a page layout (two-column resume, author
    # box), not data. Real data cells are short: page 6's table averaged ~60 chars.
    filled = [len(c.strip()) for c in cells if (c or "").strip()]
    if filled and sum(filled) / len(filled) > config.TABLE_MAX_AVG_CELL_CHARS:
        return False
    return True


def table_to_markdown(rows: list) -> str:
    rows = [[_cell(c) for c in row] for row in rows]
    rows = [r for r in rows if any(r)]                      # drop fully empty rows
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _inside(block_rect, table_rect) -> bool:
    """A text block belongs to a table if its centre lies inside the table."""
    cx = (block_rect.x0 + block_rect.x1) / 2
    cy = (block_rect.y0 + block_rect.y1) / 2
    return table_rect.contains(pymupdf.Point(cx, cy))


# ---------------------------------------------------------------------------
# One page, one file, one folder
# ---------------------------------------------------------------------------
def extract_page(page, furniture: set, report: Report):
    tables_md, table_rects = [], []
    for t in page.find_tables().tables:
        rows = t.extract()
        if is_real_table(rows, list(t.header.names)):
            md = table_to_markdown(rows)
            if md:
                tables_md.append(md)
                table_rects.append(pymupdf.Rect(t.bbox))
                report.tables_kept += 1
        else:
            report.tables_rejected += 1

    # Body = every text block NOT inside a kept table (so table text isn't included twice).
    blocks = [b for b in page.get_text("blocks") if b[6] == 0]
    body_blocks = [b for b in blocks
                   if not any(_inside(pymupdf.Rect(b[:4]), r) for r in table_rects)]

    # Put each table where it sits on the page: right after the lowest text block that
    # ends above the table's top edge. (Appending tables at the end separated page 6's
    # table from its heading.) -1 means "before all text".
    after = {}
    for rect, md in zip(table_rects, tables_md):
        above = [i for i, b in enumerate(body_blocks) if b[3] <= rect.y0 + 2]
        idx = max(above, key=lambda i: body_blocks[i][3]) if above else -1
        after.setdefault(idx, []).append(md)

    def table_part(md):
        return f"\n[Table]\n{md}\n\n"

    parts = [table_part(md) for md in after.get(-1, [])]
    for i, b in enumerate(body_blocks):
        # Blocks already end with a line break. Joining them with ANOTHER one created
        # blank lines in the middle of sentences (page 5), so we join with nothing.
        parts.append(b[4] if b[4].endswith("\n") else b[4] + "\n")
        for md in after.get(i, []):
            parts.append(table_part(md))

    text = clean_text("".join(parts), furniture)
    if len(text) < config.MIN_PAGE_CHARS and not tables_md:
        return None                                   # no text layer (or blank): caller reports it
    return text, len(tables_md)


def ingest_pdf(path) -> tuple:
    path = Path(path)
    doc = pymupdf.open(path)
    report = Report(source=path.name, pages_total=doc.page_count)
    furniture = find_furniture(doc)
    report.furniture = sorted(furniture)

    pages = []
    for i, page in enumerate(doc, 1):
        result = extract_page(page, furniture, report)
        if result is None:
            report.skipped_pages.append(i)
            continue
        text, n_tables = result
        pages.append(Page(source=path.name, page=i, text=text, n_tables=n_tables))
    report.pages_kept = len(pages)
    doc.close()
    return pages, report


def ingest_folder(folder=None) -> tuple:
    folder = Path(folder or config.DOCS_DIR)
    all_pages, reports = [], []
    for pdf in sorted(folder.glob("*.pdf")):
        pages, report = ingest_pdf(pdf)
        all_pages += pages
        reports.append(report)
    return all_pages, reports


def print_report(r: Report):
    print(f"\n{r.source}")
    print(f"  pages kept {r.pages_kept}/{r.pages_total}   "
          f"tables kept {r.tables_kept}, rejected {r.tables_rejected}")
    if r.skipped_pages:
        print(f"  ! SKIPPED {len(r.skipped_pages)} page(s) with no text layer "
              f"(scanned? needs OCR): {r.skipped_pages[:20]}")
    if r.pages_kept == 0:
        print("  ! NOTHING from this file will be searchable.")
    for f in r.furniture:
        print(f"  removed header/footer: {f[:80]!r}")


if __name__ == "__main__":
    args = sys.argv[1:]
    show = None
    if "--page" in args:
        i = args.index("--page")
        show = int(args[i + 1])
        args = args[:i] + args[i + 2:]

    if args:
        pages, reports = ingest_pdf(args[0])
        reports = [reports]
    else:
        pages, reports = ingest_folder()
        print(f"Folder: {config.DOCS_DIR}")

    for r in reports:
        print_report(r)
    print(f"\nTotal pages ready for chunking: {len(pages)}")

    if show is not None:
        match = [p for p in pages if p.page == show]
        if match:
            p = match[0]
            print(f"\n=== {p.source} page {p.page} ({p.n_tables} table(s)) ===\n{p.text}")
        else:
            print(f"\nPage {show} was not kept (skipped, or out of range).")

"""
app/chunk.py — pages -> chunks.

Strategy chosen in 11_chunking.py (3.2): STRUCTURE, max 300 tokens.
  structure-300: 9/10 found, MRR 0.80, top-3 prefill ~6 s   (fixed-150: 6/10, MRR 0.50)

Rules:
  - Rejoin wrapped PDF lines into units: paragraphs, bullets, tables.
  - Pack whole units. Never cut a sentence unless one sentence alone is too big.
  - Tables stay whole; a big table is split by ROWS with its header repeated.
  - Every chunk starts with its section heading, so it still says what it is about.
  - ALL-CAPS lines are headings even with no blank line after them (q9).
  - Headings carry across page breaks (q6: page 6 started mid-section and lost
    its heading "…legacy waste remediation", so the question could not find it).
  - Oversized paragraphs are split by sentence, then by tokens, so no chunk can
    exceed the embedding limit.
"""

import re
from dataclasses import dataclass

from app import config, tokens


@dataclass
class Chunk:
    id: str             # "file.pdf#p6-c2": unique and readable
    source: str
    page: int
    heading: str
    text: str           # heading line + body: what gets embedded and shown to Qwen
    n_tokens: int


SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])")


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
def _is_caps_heading(line: str) -> bool:
    """'V ENVIRONMENTAL BENEFITS: …', 'EDUCATION'. These headings often have no blank
    line after them, so without this rule they merge into the next paragraph (q9)."""
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 5 and sum(c.isupper() for c in letters) / len(letters) > 0.9


def _units(text: str) -> list:
    """Rejoin wrapped lines: a bullet ('- '/'o '), a paragraph, or a [Table].
    A blank line starts a new section."""
    units, cur, section = [], None, 0
    lines = text.split("\n")

    def flush():
        nonlocal cur
        if cur and cur["text"].strip():
            units.append(cur)
        cur = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush()
            section += 1
        elif line == "[Table]":
            flush()
            rows = []
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1
                rows.append(lines[i].strip())
            units.append({"kind": "table", "text": "\n".join(rows), "section": section})
        elif _is_caps_heading(line):
            flush()
            section += 1                  # a caps heading always starts a new section
            units.append({"kind": "para", "text": line, "section": section})
        elif cur is None or line.startswith(("- ", "o ")):
            flush()
            cur = {"kind": "para", "text": line, "section": section}
        else:
            cur["text"] += " " + line
        i += 1
    flush()
    return units


def _is_heading(u) -> bool:
    return (u["kind"] == "para" and not u["text"].startswith(("- ", "o "))
            and tokens.count_text(u["text"]) <= 40)


# ---------------------------------------------------------------------------
# Splitting things that are too big
# ---------------------------------------------------------------------------
def _hard_split(text: str, limit: int) -> list:
    """Last resort: cut by tokens. Only used for a single sentence bigger than the limit."""
    t = tokens._tok()
    ids = t.encode(text, add_special_tokens=False)
    return [t.decode(ids[i:i + limit]) for i in range(0, len(ids), limit)]


def _split_long(text: str, limit: int) -> list:
    """A paragraph over the limit: split by sentences; a sentence still too big: by tokens."""
    if tokens.count_text(text) <= limit:
        return [text]
    parts, cur, n = [], [], 0
    for s in SENT_END.split(text):
        sn = tokens.count_text(s)
        if sn > limit:
            if cur:
                parts.append(" ".join(cur))
                cur, n = [], 0
            parts += _hard_split(s, limit)
            continue
        if cur and n + sn > limit:
            parts.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += sn
    if cur:
        parts.append(" ".join(cur))
    return parts


def _split_table(md: str, limit: int) -> list:
    rows = md.split("\n")
    header, body = rows[:2], rows[2:]
    base = tokens.count_text("\n".join(header))
    parts, cur, n = [], [], base
    for r in body:
        rn = tokens.count_text(r)
        if cur and n + rn > limit:
            parts.append("\n".join(header + cur))
            cur, n = [], base
        cur.append(r)
        n += rn
    parts.append("\n".join(header + cur))
    out = []
    for p in parts:                       # a single enormous row: fall back to token split
        out += [p] if tokens.count_text(p) <= limit else _hard_split(p, limit)
    return out


# ---------------------------------------------------------------------------
# One page
# ---------------------------------------------------------------------------
def _chunk_page(text: str, limit: int, carry: str) -> tuple:
    """Returns (list of (heading, body)), last heading seen (to carry to the next page)."""
    sections = {}
    for u in _units(text):
        sections.setdefault(u["section"], []).append(u)

    out, pending, last_heading, first = [], "", carry, True
    for sec in sections.values():
        head = pending
        if _is_heading(sec[0]):
            head = f"{pending} / {sec[0]['text']}" if pending else sec[0]["text"]
            sec = sec[1:]
        elif first and not pending:
            head = carry                  # page starts mid-section: use the previous page's heading
        first = False

        if not sec:                       # heading on its own (e.g. above a table): carry forward
            pending = head
            continue
        pending = ""
        if head:
            last_heading = head

        budget = limit - (tokens.count_text(head) + 1 if head else 0)
        cur, cur_n = [], 0
        for u in sec:
            if u["kind"] == "table":
                if cur:
                    out.append((head, "\n".join(cur)))
                    cur, cur_n = [], 0
                out += [(head, part) for part in _split_table(u["text"], budget)]
                continue
            for piece in _split_long(u["text"], budget):
                n = tokens.count_text(piece)
                if cur and cur_n + n > budget:
                    out.append((head, "\n".join(cur)))
                    cur, cur_n = [], 0
                cur.append(piece)
                cur_n += n
        if cur:
            out.append((head, "\n".join(cur)))

    if pending:                           # a heading at the very end of the page
        last_heading = pending
    return out, last_heading


# ---------------------------------------------------------------------------
# All pages
# ---------------------------------------------------------------------------
def chunk_pages(pages, limit: int = None) -> list:
    """pages: list of app.ingest.Page, in document order."""
    limit = limit or config.CHUNK_MAX_TOKENS
    chunks, carry, source = [], "", None
    for p in pages:
        if p.source != source:            # new file: never carry a heading between files
            source, carry = p.source, ""
        pieces, carry = _chunk_page(p.text, limit, carry)
        for i, (head, body) in enumerate(pieces):
            text = f"{head}\n{body}" if head else body
            chunks.append(Chunk(id=f"{p.source}#p{p.page}-c{i}", source=p.source, page=p.page,
                                heading=head, text=text, n_tokens=tokens.count_text(text)))
    return chunks


if __name__ == "__main__":
    import sys
    from app.ingest import ingest_folder, ingest_pdf

    pages = ingest_pdf(sys.argv[1])[0] if len(sys.argv) > 1 else ingest_folder()[0]
    chunks = chunk_pages(pages)
    sizes = [c.n_tokens for c in chunks]
    print(f"{len(chunks)} chunks from {len(pages)} pages   "
          f"avg {sum(sizes) / len(sizes):.0f} tokens, max {max(sizes)}")
    for c in chunks[:3]:
        print(f"\n--- {c.id}  ({c.n_tokens} tokens) ---\n{c.text[:300]}")

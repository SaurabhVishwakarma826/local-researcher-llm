"""
Phase 3 / Topic 3.2 — Chunking.

    python 11_chunking.py 1 [page]   # see where each strategy cuts one page (default page 6)
    python 11_chunking.py 2          # sizes: how many chunks, how big, prompt cost
    python 11_chunking.py 3          # retrieval score for every strategy and size
    python 11_chunking.py 4          # the final app/chunk.py, scored the same way

Uses the waste-rules PDF through app/ingest.py (3.1).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app import config, embed, tokens  # noqa: E402
from app.ingest import ingest_pdf  # noqa: E402

PDF = Path(config.DOCS_DIR) / "solid_waste_management_rules_2026.pdf"
PREFILL_TOK_S = 66
TOP_K = 3


def qtok(text: str) -> int:
    return tokens.count_text(text)


# ---------------------------------------------------------------------------
# Rebuild logical units (paragraphs, bullets, tables) from wrapped PDF lines
# ---------------------------------------------------------------------------
def to_units(text: str) -> list:
    """PDF lines are wrapped mid-sentence. Rejoin them into units:
    a bullet ('- ' or 'o '), a paragraph, or a [Table]. A blank line starts a new section."""
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
        elif cur is None or line.startswith(("- ", "o ")):
            flush()
            cur = {"kind": "para", "text": line, "section": section}
        else:
            cur["text"] += " " + line
        i += 1
    flush()
    return units


SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])")


def to_sentences(units: list) -> list:
    out = []
    for u in units:
        if u["kind"] == "table":
            out.append(u["text"])
        else:
            out += [s for s in SENT_END.split(u["text"]) if s.strip()]
    return out


# ---------------------------------------------------------------------------
# Strategy 1 — fixed: cut every `size` tokens
# ---------------------------------------------------------------------------
def chunk_fixed(text: str, size: int) -> list:
    t = tokens._tok()
    ids = t.encode(text, add_special_tokens=False)
    overlap = size // 8
    step = size - overlap
    return [t.decode(ids[i:i + size]) for i in range(0, max(len(ids) - overlap, 1), step)]


# ---------------------------------------------------------------------------
# Strategy 2 — sentence: pack whole sentences, repeat the last one as overlap
# ---------------------------------------------------------------------------
def chunk_sentences(text: str, size: int) -> list:
    pieces = to_sentences(to_units(text))
    counts = [qtok(p) for p in pieces]
    chunks, cur, cur_n = [], [], 0
    for p, n in zip(pieces, counts):
        if cur and cur_n + n > size:
            chunks.append("\n".join(cur))
            cur, cur_n = cur[-1:], qtok(cur[-1])      # overlap: one sentence
        cur.append(p)
        cur_n += n
    if cur:
        chunks.append("\n".join(cur))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3 — structure: paragraphs/bullets, whole tables, heading prefix
# ---------------------------------------------------------------------------
def split_table(md: str, budget: int) -> list:
    rows = md.split("\n")
    header, body = rows[:2], rows[2:]
    base = qtok("\n".join(header))
    parts, cur, n = [], [], base
    for r in body:
        rn = qtok(r)
        if cur and n + rn > budget:
            parts.append("\n".join(header + cur))
            cur, n = [], base
        cur.append(r)
        n += rn
    parts.append("\n".join(header + cur))
    return parts


def _is_heading(u) -> bool:
    return u["kind"] == "para" and not u["text"].startswith(("- ", "o ")) and qtok(u["text"]) <= 40


def chunk_structure(text: str, size: int) -> list:
    sections = {}
    for u in to_units(text):
        sections.setdefault(u["section"], []).append(u)

    chunks, carried = [], ""
    for sec in sections.values():
        head = carried
        if _is_heading(sec[0]):
            head = (carried + " / " + sec[0]["text"]) if carried else sec[0]["text"]
            sec = sec[1:]
        if not sec:                 # a heading on its own (e.g. above a table): carry it forward
            carried = head
            continue
        carried = ""

        prefix = f"{head}\n" if head else ""
        budget = size - qtok(prefix)
        cur, cur_n = [], 0
        for u in sec:
            if u["kind"] == "table":
                if cur:
                    chunks.append(prefix + "\n".join(cur))
                    cur, cur_n = [], 0
                chunks += [prefix + part for part in split_table(u["text"], budget)]
                continue
            n = qtok(u["text"])
            if cur and cur_n + n > budget:
                chunks.append(prefix + "\n".join(cur))
                cur, cur_n = [], 0
            cur.append(u["text"])
            cur_n += n
        if cur:
            chunks.append(prefix + "\n".join(cur))
    if carried:
        chunks.append(carried)
    return chunks


STRATEGIES = {"fixed": chunk_fixed, "sentence": chunk_sentences, "structure": chunk_structure}
SIZES = [150, 300]


def build(pages, strategy: str, size: int) -> list:
    """Chunks never cross a page boundary, so every chunk has exactly one page number."""
    out = []
    for p in pages:
        out += [(p.page, c) for c in STRATEGIES[strategy](p.text, size) if c.strip()]
    return out


def load_pages():
    if not PDF.exists():
        sys.exit(f"Not found: {PDF}")
    pages, _ = ingest_pdf(PDF)
    return pages


# ---------------------------------------------------------------------------
# 1 — Where does each strategy cut?
# ---------------------------------------------------------------------------
def exp1_show(page_no: int):
    page = next((p for p in load_pages() if p.page == page_no), None)
    if page is None:
        sys.exit(f"Page {page_no} not available")
    for name, fn in STRATEGIES.items():
        chunks = fn(page.text, 150)
        print(f"\n==================== {name.upper()}  (size 150) -> {len(chunks)} chunks ====================")
        for i, c in enumerate(chunks[:4], 1):
            print(f"\n--- chunk {i}  ({qtok(c)} tokens) ---")
            print(f"  starts: {c[:90]!r}")
            print(f"  ends:   {c[-90:]!r}")
        if len(chunks) > 4:
            print(f"\n  ... {len(chunks) - 4} more")
    print("\nLook at the starts and ends: which strategy cuts in the middle of a word,")
    print("a sentence, or the table? Which chunks still say what they are ABOUT?")


# ---------------------------------------------------------------------------
# 2 — Sizes and cost
# ---------------------------------------------------------------------------
def exp2_sizes():
    pages = load_pages()
    print(f"{'strategy':<10} {'size':>5} {'chunks':>7} {'avg tok':>8} {'max tok':>8} "
          f"{'max BGE tok':>12} {'top-3 prefill':>14}")
    print("-" * 72)
    for name in STRATEGIES:
        for size in SIZES:
            chunks = [c for _, c in build(pages, name, size)]
            q = [qtok(c) for c in chunks]
            b = max(embed.count_tokens(c) for c in chunks)
            avg = sum(q) / len(q)
            flag = "  <- over BGE limit!" if b > config.EMBED_MAX_TOKENS else ""
            print(f"{name:<10} {size:>5} {len(chunks):>7} {avg:>8.0f} {max(q):>8} "
                  f"{b:>12} {TOP_K * avg / PREFILL_TOK_S:>12.1f} s{flag}")
    print("\n'max tok' above the target size = a single sentence, bullet or table row")
    print("that could not be split further. 'max BGE tok' is the embedding model's own count.")


# ---------------------------------------------------------------------------
# 3 — Retrieval score
# ---------------------------------------------------------------------------
# Each question has a phrase that MUST appear in a retrieved chunk for the answer to be possible.
QUESTIONS = [
    ("What is the RDF substitution target for cement plants and Waste to Energy plants?", "from 5% to 15%"),
    ("Into how many streams must waste be segregated under SWM 2026?", "four-stream"),
    ("What can local authorities in hilly areas and islands charge tourists?", "user fees on tourists"),
    ("What should be reported to make RDF substitution targets enforceable?", "NCV bands"),
    ("How did the rules for bulk generators change between SWM 2016 and SWM 2026?", "Clear thresholds"),
    ("How should progress in legacy dumpsite remediation be measured?", "mass removed"),
    ("What goes wrong when mixed waste is shredded into RDF-like material?", "high moisture"),
    ("What fire prevention measures are recommended at MRFs?", "hotspot monitoring"),
    ("What is the expected environmental outcome of decentralised processing?", "Near-zero CH₄"),
    ("Why do hilly areas and islands need special waste provisions?", "seasonal load surges"),
]


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def score(texts: list, q_vecs: list) -> tuple:
    vecs = embed.embed_documents(texts)
    marks, rr = [], []
    for (q, phrase), qv in zip(QUESTIONS, q_vecs):
        order = np.argsort(-embed.similarity(qv, vecs))[:TOP_K]
        rank = next((r for r, j in enumerate(order, 1) if _norm(phrase) in _norm(texts[j])), None)
        marks.append(str(rank) if rank else ".")
        rr.append(1 / rank if rank else 0)
    return marks, sum(m != "." for m in marks), sum(rr) / len(rr), len(texts)


def exp3_retrieval():
    pages = load_pages()
    q_vecs = [embed.embed_query(q) for q, _ in QUESTIONS]
    results = {}
    for name in STRATEGIES:
        for size in SIZES:
            label = f"{name}-{size}"
            texts = [c for _, c in build(pages, name, size)]
            try:
                results[label] = score(texts, q_vecs)
            except ValueError as e:
                print(f"{label}: CANNOT EMBED — {e}")

    print(f"Each column is one question. Number = rank where the answer phrase was found "
          f"(1 = top), '.' = not in top {TOP_K}.\n")
    print(f"{'config':<16} {'chunks':>6}  {'q1 q2 q3 q4 q5 q6 q7 q8 q9 q10':<31} {'hits':>5} {'MRR':>6}")
    print("-" * 72)
    for label, (marks, hits, mrr, n) in sorted(results.items(), key=lambda kv: (-kv[1][1], -kv[1][2])):
        print(f"{label:<16} {n:>6}  {'  '.join(f'{m:<1}' for m in marks):<31} {hits:>3}/{len(QUESTIONS)} {mrr:>6.2f}")
    print("\nMRR (mean reciprocal rank): 1.0 = answer always ranked first, 0 = never found.")
    print("Caution: 10 questions, written by looking at the text. Real questions will be")
    print("phrased differently, so these scores are optimistic. Phase 4 fixes that.")


# ---------------------------------------------------------------------------
# 4 — The final app/chunk.py, scored the same way
# ---------------------------------------------------------------------------
def exp4_final():
    from app.chunk import chunk_pages
    pages = load_pages()
    q_vecs = [embed.embed_query(q) for q, _ in QUESTIONS]
    chunks = chunk_pages(pages)
    marks, hits, mrr, n = score([c.text for c in chunks], q_vecs)
    sizes = [c.n_tokens for c in chunks]
    print(f"app/chunk.py   {n} chunks, avg {sum(sizes) / len(sizes):.0f} tokens, max {max(sizes)}")
    print(f"               {'  '.join(marks)}   {hits}/{len(QUESTIONS)}   MRR {mrr:.2f}")
    print("               (structure-300 before the fixes: 1  2  1  1  1  .  2  1  1  1   9/10   MRR 0.80)")
    for label, phrase in (("q6", "mass removed"), ("q9", "Near-zero")):
        c = next((c for c in chunks if _norm(phrase) in _norm(c.text)), None)
        if c:
            print(f"\n{label} chunk: {c.id}\n  heading: {c.heading!r}")


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1"
    if key == "1":
        exp1_show(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif key == "2":
        exp2_sizes()
    elif key == "3":
        exp3_retrieval()
    elif key == "4":
        exp4_final()
    else:
        sys.exit(__doc__)

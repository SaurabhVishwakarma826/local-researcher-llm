"""
app/store.py — the ONLY file in the project that talks to the vector database (Chroma).

Design (3.3):
  - WE embed (app/embed.py, BGE). Chroma only stores and searches. The collection is
    created with NO embedding function, so Chroma can never silently embed text with
    its own default model (which would make every score meaningless, with no error).
  - Re-indexing is safe: unchanged files are skipped (content hash), a changed file's
    old chunks are deleted before the new ones go in, removed files are removed.
  - search() returns cosine similarity, the same number used in Phase 2.

Usage from the project root:
    python -m app.store index                 # index everything in config.DOCS_DIR
    python -m app.store index --force         # re-index even unchanged files
    python -m app.store search "your question"
    python -m app.store search "your question" --source file.pdf
    python -m app.store list                  # what is in the store
"""

import hashlib
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings

from app import config, embed
from app.chunk import chunk_pages
from app.ingest import ingest_pdf, print_report


@dataclass
class Hit:
    id: str
    text: str
    source: str
    page: int
    heading: str
    score: float        # cosine similarity: for RANKING only (Phase 2)


@lru_cache(maxsize=1)
def collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR,
                                       settings=Settings(anonymized_telemetry=False))
    args = dict(name=config.CHROMA_COLLECTION, embedding_function=None)
    try:                                  # Chroma 1.x
        return client.get_or_create_collection(**args, configuration={"hnsw": {"space": "cosine"}})
    except TypeError:                     # Chroma 0.x
        return client.get_or_create_collection(**args, metadata={"hnsw:space": "cosine"})


def _file_hash(path: Path) -> str:
    """Fingerprint of the file AND the processing pipeline version. Without the version, a
    change to ingest.py or chunk.py would leave old chunks in the store, silently, because
    the PDF itself did not change. Bump config.INDEX_VERSION whenever those files change."""
    version = str(getattr(config, "INDEX_VERSION", 1)).encode()
    return hashlib.sha256(path.read_bytes() + b"|v" + version).hexdigest()[:16]


def indexed_files() -> dict:
    """source file name -> content hash, for everything currently stored."""
    metas = collection().get(include=["metadatas"])["metadatas"]
    return {m["source"]: m["file_hash"] for m in metas}


def remove_file(source: str):
    collection().delete(where={"source": source})


def index_file(path, force: bool = False) -> dict:
    path = Path(path)
    h = _file_hash(path)
    if not force and indexed_files().get(path.name) == h:
        return {"source": path.name, "status": "unchanged", "chunks": 0, "seconds": 0.0}

    t0 = time.time()
    pages, report = ingest_pdf(path)
    chunks = chunk_pages(pages)
    remove_file(path.name)                 # never leave stale chunks from an older version
    if chunks:
        vecs = embed.embed_documents([c.text for c in chunks])
        collection().add(
            ids=[c.id for c in chunks],
            embeddings=vecs.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page, "heading": c.heading,
                        "n_tokens": c.n_tokens, "file_hash": h} for c in chunks],
        )
    return {"source": path.name, "status": "indexed", "chunks": len(chunks),
            "seconds": time.time() - t0, "report": report}


def index_folder(folder=None, force: bool = False) -> list:
    folder = Path(folder or config.DOCS_DIR)
    present = {p.name for p in folder.glob("*.pdf")}
    for gone in set(indexed_files()) - present:
        remove_file(gone)                  # file deleted from the folder -> delete from the store
    return [index_file(p, force) for p in sorted(folder.glob("*.pdf"))]


def search(question: str, k: int = None, source: str = None) -> list:
    qv = embed.embed_query(question)
    res = collection().query(
        query_embeddings=[qv.tolist()],
        n_results=k or config.TOP_K,
        where={"source": source} if source else None,
        include=["documents", "metadatas", "distances"],
    )
    return [Hit(id=i, text=d, source=m["source"], page=m["page"], heading=m["heading"],
                score=1.0 - dist)          # cosine distance -> cosine similarity
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0])]


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "list"

    if cmd == "index":
        for r in index_folder(force="--force" in args):
            if r["status"] == "unchanged":
                print(f"{r['source']}: unchanged, skipped")
            else:
                print_report(r["report"])
                print(f"  -> {r['chunks']} chunks stored in {r['seconds']:.1f}s")
        print(f"\nStore now holds {collection().count()} chunks.")

    elif cmd == "search" and len(args) > 1:
        src = args[args.index("--source") + 1] if "--source" in args else None
        for rank, h in enumerate(search(args[1], source=src), 1):
            print(f"\n{rank}. score {h.score:.3f}   {h.source}  p{h.page}")
            print(f"   heading: {h.heading[:90]!r}")
            print(f"   {h.text[:250]!r}")

    else:
        files = indexed_files()
        print(f"{collection().count()} chunks from {len(files)} file(s):")
        for f in sorted(files):
            n = len(collection().get(where={"source": f}, include=[])["ids"])
            print(f"  {n:>4}  {f}")

"""
app/retrieve.py — retrieval strategies on top of the vector store (Phase 5).

  dense   embedding similarity (store.search): compares MEANING. Blurs exact words
          and numbers (2.2: NH44-KM-123 vs NH48-KM-321 scored 0.93).
  bm25    keyword matching: rewards exact words that are RARE across chunks, in SHORT
          chunks. Cannot see meaning or synonyms.
  hybrid  both, combined with Reciprocal Rank Fusion: each chunk gets 1/(RRF_K + rank)
          from each list. Ranks, not scores, because the two score scales differ.

  rerank  (5.2) optional second stage on any mode: a CROSS-encoder reads the question and
          each candidate TOGETHER and re-sorts the top RERANK_POOL. More precise than
          embeddings (which encode question and chunk separately), but nothing can be
          precomputed, so it only runs on a shortlist.

Note: Hit.score means different things per mode (cosine / BM25 / RRF / reranker). Rank only.

Usage from the project root:
    python -m app.retrieve "your question"                   # dense, bm25, hybrid side by side
    python -m app.retrieve "your question" --rerank minilm   # ... plus hybrid reranked
    python -m app.retrieve "your question" --source file.pdf
"""

import re
import sys
import time
from dataclasses import replace
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app import config, store

MODES = ("dense", "bm25", "hybrid")

# Words so common they match every chunk. BM25 already down-weights common words,
# but with only ~60 chunks its "how rare is this word" estimate is noisy.
STOPWORDS = frozenset("""a an and are as at be been by can could did do does for from had has have
how i in is it its of on or should that the their them there these they this to was were what
when where which who whom why will with would you your""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """Lowercase words and numbers. '5%' -> '5', 'four-stream' -> 'four', 'stream'."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


_cache = {"key": None}


def _index() -> dict:
    """BM25 index over every chunk in the store, rebuilt only when the store changes."""
    got = store.collection().get(include=["documents", "metadatas"])
    key = hash(tuple(zip(got["ids"], (m["file_hash"] for m in got["metadatas"]))))
    if _cache["key"] != key:
        hits = [store.Hit(id=i, text=d, source=m["source"], page=m["page"],
                          heading=m["heading"], score=0.0)
                for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]
        _cache.update(key=key, hits=hits,
                      bm25=BM25Okapi([tokenize(h.text) for h in hits]) if hits else None)
    return _cache


def bm25_search(question: str, k: int, source: str = None) -> list:
    idx = _index()
    if idx["bm25"] is None:
        return []
    scores = idx["bm25"].get_scores(tokenize(question))
    out = []
    for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
        if scores[i] <= 0:
            break                        # no shared words at all: not a keyword match
        h = idx["hits"][i]
        if source and h.source != source:
            continue
        out.append(replace(h, score=float(scores[i])))
        if len(out) == k:
            break
    return out


def hybrid_search(question: str, k: int, source: str = None) -> list:
    pool = min(config.HYBRID_POOL, max(store.collection().count(), 1))
    fused = {}
    for ranked in (store.search(question, k=pool, source=source),
                   bm25_search(question, pool, source)):
        for rank, h in enumerate(ranked, 1):
            entry = fused.setdefault(h.id, [h, 0.0])
            entry[1] += 1.0 / (config.RRF_K + rank)
    best = sorted(fused.values(), key=lambda e: -e[1])[:k]
    return [replace(h, score=s) for h, s in best]


@lru_cache(maxsize=2)
def _reranker(name: str):
    from sentence_transformers import CrossEncoder
    model_id = config.RERANKERS[name]
    try:                                   # local cache first: no network after the first download
        return CrossEncoder(model_id, device="cpu", max_length=512, local_files_only=True)
    except Exception:
        return CrossEncoder(model_id, device="cpu", max_length=512)


def rerank(question: str, hits: list, k: int, name: str) -> list:
    if not hits:
        return []
    scores = _reranker(name).predict([(question, h.text) for h in hits])
    best = sorted(zip(hits, scores), key=lambda hs: -float(hs[1]))[:k]
    return [replace(h, score=float(s)) for h, s in best]


def _first_stage(question: str, k: int, source: str, mode: str) -> list:
    if mode == "dense":
        return store.search(question, k=k, source=source)
    if mode == "bm25":
        return bm25_search(question, k, source)
    if mode == "hybrid":
        return hybrid_search(question, k, source)
    raise ValueError(f"mode must be one of {MODES}, not {mode!r}")


def search(question: str, k: int = None, source: str = None, mode: str = None,
           reranker: str = "config") -> list:
    """reranker: a key of config.RERANKERS, None for no reranking, or "config" to use
    config.RERANKER."""
    k = k or config.TOP_K
    mode = mode or config.RETRIEVAL_MODE
    name = config.RERANKER if reranker == "config" else reranker
    if not name:
        return _first_stage(question, k, source, mode)
    pool = _first_stage(question, max(config.RERANK_POOL, k), source, mode)
    return rerank(question, pool, k, name)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    q = args[0]
    src = args[args.index("--source") + 1] if "--source" in args else None
    rr = args[args.index("--rerank") + 1] if "--rerank" in args else None
    print(f"Q: {q}\nkeywords BM25 matches on: {tokenize(q)}\n")
    runs = [(m, m, None) for m in MODES] + ([(f"hybrid + rerank ({rr})", "hybrid", rr)] if rr else [])
    for label, mode, name in runs:
        t0 = time.time()
        hits = search(q, k=5, source=src, mode=mode, reranker=name)
        print(f"--- {label}  ({time.time() - t0:.2f}s) ---")
        for r, h in enumerate(hits, 1):
            print(f"  {r}. {h.score:8.4f}  {h.id:<50} {h.heading[:45]!r}")
        print()

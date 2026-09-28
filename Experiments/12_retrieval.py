"""
Phase 3 / Topic 3.3 — Storage and retrieval.

    python 12_retrieval.py 1                     # index the folder, then index again (should skip)
    python 12_retrieval.py 2                     # is Chroma's search exact? + the 10 questions
    python 12_retrieval.py 3 "question" [file]   # ask anything; optionally only one file
    python 12_retrieval.py 4                     # off-topic questions: does retrieval ever say "nothing"?
"""

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import config, embed, store  # noqa: E402

# Reuse the 10 questions from 3.2 instead of copying them.
_spec = importlib.util.spec_from_file_location("chunking", Path(__file__).parent / "11_chunking.py")
_chunking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chunking)
QUESTIONS, _norm = _chunking.QUESTIONS, _chunking._norm


def _dir_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


# ---------------------------------------------------------------------------
# 1 — Index once, then again
# ---------------------------------------------------------------------------
def exp1_index():
    t0 = time.time()
    first = store.index_folder(force=True)
    t_first = time.time() - t0
    for r in first:
        print(f"  {r['source']:<45} {r['chunks']:>4} chunks  {r['seconds']:>5.1f}s")
    print(f"First index (forced): {t_first:.1f}s, {store.collection().count()} chunks stored")

    t0 = time.time()
    second = store.index_folder()
    print(f"\nSecond run: {time.time() - t0:.2f}s  -> "
          f"{sum(r['status'] == 'unchanged' for r in second)}/{len(second)} files skipped as unchanged")
    print(f"On disk: {_dir_mb(ROOT / config.CHROMA_DIR):.1f} MB in {config.CHROMA_DIR}")
    print("\nThe scanned book shows 0 chunks and is re-checked every run (nothing is stored")
    print("for it, so there is no fingerprint to compare). That is expected.")


# ---------------------------------------------------------------------------
# 2 — Exactness check + the 10 questions, now with ALL files in the store
# ---------------------------------------------------------------------------
def exp2_check():
    col = store.collection()
    everything = col.get(include=["embeddings", "documents", "metadatas"])
    all_vecs = np.array(everything["embeddings"])
    print(f"Store: {len(everything['ids'])} chunks from "
          f"{len({m['source'] for m in everything['metadatas']})} files\n")

    same, worst, marks, rr = 0, 0.0, [], []
    for q, phrase in QUESTIONS:
        hits = store.search(q)
        exact = np.argsort(-(all_vecs @ embed.embed_query(q)))[:config.TOP_K]
        same += [h.id for h in hits] == [everything["ids"][i] for i in exact]
        for h in hits:
            i = everything["ids"].index(h.id)
            worst = max(worst, abs(h.score - float(all_vecs[i] @ embed.embed_query(q))))
        rank = next((r for r, h in enumerate(hits, 1) if _norm(phrase) in _norm(h.text)), None)
        marks.append(str(rank) if rank else ".")
        rr.append(1 / rank if rank else 0)

    print(f"Chroma top-{config.TOP_K} identical to exact search: {same}/{len(QUESTIONS)} questions")
    print(f"Largest score difference vs our own cosine:  {worst:.6f}")
    print(f"\n10 questions through the store:  {'  '.join(marks)}   "
          f"{sum(m != '.' for m in marks)}/10   MRR {sum(rr) / len(rr):.2f}")
    print("(3.2 result, waste-rules PDF only: 1  2  1  1  1  1  1  1  1  1   10/10   MRR 0.95)")
    print("\nIf the score dropped, chunks from ANOTHER file (the resume) pushed a")
    print("correct chunk out of the top 3. More documents = more competition.")


# ---------------------------------------------------------------------------
# 3 — Ask anything
# ---------------------------------------------------------------------------
def exp3_ask(question: str, source: str = None):
    where = f" (only {source})" if source else ""
    print(f"Q: {question}{where}")
    for rank, h in enumerate(store.search(question, source=source), 1):
        print(f"\n{rank}. score {h.score:.3f}   {h.source}  page {h.page}")
        print(f"   heading: {h.heading[:90]!r}")
        print(f"   {h.text[:220]!r}")


# ---------------------------------------------------------------------------
# 4 — Off-topic questions
# ---------------------------------------------------------------------------
OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Who won the cricket world cup in 2011?",
    "What is the boiling point of water?",
]


def exp4_off_topic():
    print(f"{'best score':>10}  {'from':<42} question")
    print("-" * 95)
    on = [store.search(q)[0].score for q, _ in QUESTIONS]
    for q in OFF_TOPIC:
        h = store.search(q)[0]
        print(f"{h.score:>10.3f}  {h.source[:40]:<42} {q}")
    print(f"\nOn-topic questions (the 10 from 3.2): best scores {min(on):.3f} to {max(on):.3f}")
    print("\nRetrieval ALWAYS returns 3 chunks, even for questions no document can answer.")
    print("Compare the ranges: is there a clean gap between on-topic and off-topic scores?")
    print("If not, a score threshold cannot reliably say 'no answer here', and 3.4's")
    print("prompt has to make Qwen say 'I don't know' instead.")


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1"
    if key == "1":
        exp1_index()
    elif key == "2":
        exp2_check()
    elif key == "3" and len(sys.argv) > 2:
        exp3_ask(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif key == "4":
        exp4_off_topic()
    else:
        sys.exit(__doc__)

"""
app/embed.py — the ONLY file in the project that turns text into vectors.

Lessons applied (09_embeddings.py):
  - exp 5: BGE silently drops everything past 512 tokens. So embed_documents()
    REFUSES over-long text instead of letting it be cut off.
  - exp 4: BGE expects a prefix on questions only. embed_query() adds it,
    embed_documents() never does, so the two can't get mixed up.
  - Vectors are normalised, so a dot product equals cosine similarity.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app import config


@lru_cache(maxsize=1)
def _model():
    # Local cache first: no network, no Hugging Face warnings after the first download.
    try:
        return SentenceTransformer(config.EMBED_MODEL_ID, device="cpu", local_files_only=True)
    except Exception:
        return SentenceTransformer(config.EMBED_MODEL_ID, device="cpu")


def count_tokens(text: str) -> int:
    """Tokens according to the EMBEDDING model's tokenizer (not Qwen's — see app/tokens.py)."""
    return len(_model().tokenizer(text, truncation=False, verbose=False)["input_ids"])


def embed_documents(texts: list, batch_size: int = 32) -> np.ndarray:
    """Vectors for chunks of documents. Raises if any text would be silently truncated."""
    for i, text in enumerate(texts):
        n = count_tokens(text)
        if n > config.EMBED_MAX_TOKENS:
            raise ValueError(
                f"Document {i} is {n} embedding tokens; the limit is {config.EMBED_MAX_TOKENS}. "
                "Split it smaller: the embedding model would silently ignore everything past the limit."
            )
    return _model().encode(texts, normalize_embeddings=True, batch_size=batch_size)


def embed_query(question: str) -> np.ndarray:
    """Vector for a user question, with BGE's query prefix added."""
    return _model().encode([config.EMBED_QUERY_PREFIX + question], normalize_embeddings=True)[0]


def similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query against many documents. Use for RANKING only:
    the absolute numbers mean little (unrelated text still scored 0.3-0.4)."""
    return doc_vecs @ query_vec

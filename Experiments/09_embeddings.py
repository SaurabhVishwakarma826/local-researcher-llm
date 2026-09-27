"""
Phase 2 / Topic 2.2 — Embeddings.

Run one experiment at a time:
    python 09_embeddings.py 1   # look inside a vector + how fast your CPU embeds
    python 09_embeddings.py 2   # meaning vs shared words
    python 09_embeddings.py 3   # where embeddings fail (the Phase 2 exit test)
    python 09_embeddings.py 4   # ranking passages for a question, with/without prefix
    python 09_embeddings.py 5   # the embedding model's own silent truncation

First run downloads BAAI/bge-small-en-v1.5 (~130 MB).
"""

import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_ID = "BAAI/bge-small-en-v1.5"
# BGE expects this in front of QUESTIONS only, never in front of documents.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_ID, device="cpu")
    return _model


def embed(texts):
    # normalize_embeddings=True makes every vector length 1,
    # so a plain dot product IS the cosine similarity.
    return model().encode(texts, normalize_embeddings=True)


def sim(a: str, b: str) -> float:
    v = embed([a, b])
    return float(v[0] @ v[1])


def n_tokens(text: str) -> int:
    """Tokens according to BGE's OWN tokenizer (not Qwen's — different vocabulary)."""
    return len(model().tokenizer(text)["input_ids"])


# ---------------------------------------------------------------------------
# 1 — What a vector looks like, and how fast your CPU makes them
# ---------------------------------------------------------------------------
def exp1_inside():
    v = embed(["The pothole on NH-44 was 30 cm deep."])[0]
    print(f"Dimensions: {v.shape[0]}")
    print(f"First 8 numbers: {np.round(v[:8], 4).tolist()}")
    print(f"Vector length: {np.linalg.norm(v):.4f}  (1.0 because we normalised it)")
    print(f"Max input length: {model().max_seq_length} tokens (BGE's own tokenizer)\n")

    sentences = [f"Section {i} of the highway shows moderate cracking and some rutting." for i in range(200)]
    embed(sentences[:8])                         # warm-up
    t0 = time.time()
    embed(sentences)
    dt = time.time() - t0
    print(f"Embedded 200 sentences in {dt:.2f}s  ->  {200 / dt:.0f} sentences/sec on your CPU")
    print("Keep this number: in Phase 3 it decides how long it takes to ingest a PDF.")


# ---------------------------------------------------------------------------
# 2 — Meaning vs shared words
# ---------------------------------------------------------------------------
def exp2_meaning():
    pairs = [
        ("same meaning, different words", "The road has a deep hole.", "There is a pothole on the highway."),
        ("same meaning, different words", "The survey vehicle measures roughness.", "A car with sensors records how bumpy the road is."),
        ("same word, different meaning", "The bank of the river flooded.", "The bank approved my loan."),
        ("same word, different meaning", "The model predicted rain.", "The model walked on the runway."),
        ("related topic", "Potholes damage car tyres.", "Rutting forms in wheel paths."),
        ("unrelated", "Potholes damage car tyres.", "My cat likes to sleep in the sun."),
    ]
    rows = sorted(((sim(a, b), label, a, b) for label, a, b in pairs), reverse=True)
    print(f"{'score':>6}  {'type':<31} pair")
    print("-" * 100)
    for s, label, a, b in rows:
        print(f"{s:>6.3f}  {label:<31} {a!r}  vs  {b!r}")
    print("\nIf 'same meaning, different words' ranks ABOVE 'same word, different meaning',")
    print("the model is comparing meaning, not just matching words.")
    print("Notice even 'unrelated' is not near 0. Scores are only useful for RANKING.")


# ---------------------------------------------------------------------------
# 3 — Where embeddings fail
# ---------------------------------------------------------------------------
def exp3_failures():
    pairs = [
        ("negation", "The road has potholes.", "The road has no potholes."),
        ("negation", "The bridge is safe to use.", "The bridge is not safe to use."),
        ("numbers", "IRI is 2.1 m/km on this section.", "IRI is 9.8 m/km on this section."),
        ("numbers", "Repair cost: 4 lakh rupees.", "Repair cost: 400 lakh rupees."),
        ("identifiers", "Section NH44-KM-123 needs repair.", "Section NH48-KM-321 needs repair."),
        ("direction", "Rutting was found in the left lane.", "Rutting was found in the right lane."),
        ("baseline paraphrase", "The road has potholes.", "There are holes in the road surface."),
    ]
    print(f"{'score':>6}  {'failure type':<20} pair")
    print("-" * 100)
    for label, a, b in pairs:
        s = sim(a, b)
        flag = "  <-- opposite meaning, very high score" if s > 0.85 and label != "baseline paraphrase" else ""
        print(f"{s:>6.3f}  {label:<20} {a!r}  vs  {b!r}{flag}")
    print("\nPhase 2 exit test: find a pair that means the OPPOSITE and still scores > 0.85.")
    print("Compare each row with the paraphrase baseline: is 'no potholes' closer to")
    print("'has potholes' than a real paraphrase is?")


# ---------------------------------------------------------------------------
# 4 — Ranking passages for a question
# ---------------------------------------------------------------------------
QUESTION = "How is road roughness measured?"
PASSAGES = [
    ("relevant",   "IRI is computed by simulating a quarter-car model over the measured longitudinal road profile."),
    ("relevant",   "Roughness surveys use laser profilometers mounted on survey vehicles."),
    ("road, not the answer", "Potholes form when water weakens the base layer under traffic loads."),
    ("tangential", "The survey vehicle must be calibrated before every trip."),
    ("word trap",  "Roughness of the skin can be treated with a good moisturiser."),
    ("unrelated",  "The office canteen serves lunch from 1 pm."),
]


def exp4_ranking():
    doc_vecs = embed([p for _, p in PASSAGES])        # documents: NO prefix
    for label, q in (("WITHOUT query prefix", QUESTION), ("WITH query prefix", QUERY_PREFIX + QUESTION)):
        qv = embed([q])[0]
        scores = doc_vecs @ qv
        order = np.argsort(-scores)
        print(f"\n{label}:  {QUESTION!r}")
        for rank, i in enumerate(order, 1):
            kind, text = PASSAGES[i]
            print(f"  {rank}. {scores[i]:.3f}  [{kind:<20}] {text[:70]}")
    print("\nThings to check:")
    print("  - Are both 'relevant' passages at ranks 1-2?")
    print("  - Where did the 'word trap' (skin roughness) land?")
    print("  - Did the prefix change the ORDER, or only the numbers?")
    print("  - The 'unrelated' score is still well above 0 — which is why you can")
    print("    never say 'score above 0.5 means relevant'.")


# ---------------------------------------------------------------------------
# 5 — The embedding model silently truncates too
# ---------------------------------------------------------------------------
def exp5_truncation():
    q = QUERY_PREFIX + "Why was the bridge closed?"
    key = "The bridge was closed because inspectors found cracked welds in the main girders."
    filler = ("The committee reviewed the annual budget and discussed staffing levels "
              "for the next quarter in detail. ") * 40

    docs = [
        ("key sentence alone", key),
        ("key at START of long text", key + " " + filler),
        ("key at END of long text", filler + " " + key),
        ("filler only (no answer)", filler),
    ]
    qv = embed([q])[0]
    limit = model().max_seq_length
    print(f"BGE reads at most {limit} tokens. Anything after that is dropped, with no warning.\n")
    print(f"{'score':>6}  {'BGE tokens':>10}  document")
    print("-" * 60)
    for label, d in docs:
        s = float(embed([d])[0] @ qv)
        t = n_tokens(d)
        cut = "  (cut off!)" if t > limit else ""
        print(f"{s:>6.3f}  {t:>10}  {label}{cut}")
    print("\nTwo lessons:")
    print("  1. Key at END: the answer was beyond the limit, so it was never embedded.")
    print("     It scores like the filler — this chunk would never be retrieved.")
    print("  2. Key at START: it was read, but mixed with lots of unrelated text,")
    print("     so the score is diluted. Big chunks blur meaning.")
    print("Both push Phase 3 toward SMALL chunks, measured in tokens.")


EXPERIMENTS = {"1": exp1_inside, "2": exp2_meaning, "3": exp3_failures,
               "4": exp4_ranking, "5": exp5_truncation}

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1"
    if key not in EXPERIMENTS:
        sys.exit(f"Pick one of: {', '.join(EXPERIMENTS)}")
    EXPERIMENTS[key]()

"""
eval/run_eval.py — the evaluation harness.

    python eval/run_eval.py retrieval                 # retrieval only, test split (seconds)
    python eval/run_eval.py retrieval --split dev
    python eval/run_eval.py retrieval --label chunk300   # name this run in the results file

Every run is saved to eval/results/<timestamp>_<kind>_<label>.json so any future
change can be compared against today's numbers.
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config, store  # noqa: E402

GOLDEN = ROOT / "eval" / "golden.jsonl"
RESULTS = ROOT / "eval" / "results"
KS = [1, 3, 5, 10]


def norm(s: str) -> str:
    """Same normalisation as check_golden.py: lowercase, hyphens as spaces, one space."""
    return " ".join(s.lower().replace("-", " ").split())


def load_golden(split: str) -> list:
    qs = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    return qs if split == "all" else [q for q in qs if q["split"] == split]


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieval_rows(qs: list) -> list:
    everything = store.collection().get(include=["documents", "metadatas"])
    chunks = [(i, norm(d), m) for i, d, m in
              zip(everything["ids"], everything["documents"], everything["metadatas"])]
    rows = []
    for q in qs:
        hits = store.search(q["question"], k=max(KS), source=q.get("only_source"))
        row = {"id": q["id"], "by": q.get("by", "?"), "type": q["type"], "tags": q.get("tags", []),
               "question": q["question"], "top_score": round(hits[0].score, 4) if hits else None,
               "retrieved": [{"id": h.id, "score": round(h.score, 4)} for h in hits[:5]]}
        if q["type"] == "answer":
            ev = norm(q["evidence"])
            holders = [cid for cid, text, m in chunks if m["source"] == q["source"] and ev in text]
            rank = next((r for r, h in enumerate(hits, 1)
                         if h.source == q["source"] and ev in norm(h.text)), None)
            row.update(rank=rank, findable=bool(holders), evidence_chunks=holders)
        rows.append(row)
    return rows


def retrieval_summary(rows: list) -> dict:
    ans = [r for r in rows if r["type"] == "answer"]
    ref = [r for r in rows if r["type"] == "refuse"]

    def block(rs):
        return {"n": len(rs),
                **{f"hit@{k}": round(_mean(bool(r["rank"]) and r["rank"] <= k for r in rs), 3) for k in KS},
                "mrr": round(_mean(1 / r["rank"] if r["rank"] else 0 for r in rs), 3)}

    by_tag, by_author = defaultdict(list), defaultdict(list)
    for r in ans:
        by_author[r["by"]].append(r)
        for t in r["tags"] or ["(untagged)"]:
            by_tag[t].append(r)
    found1 = [r["top_score"] for r in ans if r["rank"] == 1]
    return {
        "answerable": block(ans),
        "unfindable": [r["id"] for r in ans if not r["findable"]],
        "by_author": {a: block(rs) for a, rs in sorted(by_author.items())},
        "by_tag": {t: block(rs) for t, rs in sorted(by_tag.items())},
        "threshold": {
            "answerable_found_at_1": [min(found1), max(found1)] if found1 else None,
            "refuse_best_scores": [min(r["top_score"] for r in ref), max(r["top_score"] for r in ref)] if ref else None,
            "refuse_above_lowest_answerable": sum(r["top_score"] >= min(found1) for r in ref) if found1 and ref else None,
            "n_refuse": len(ref),
        },
    }


def print_retrieval(s: dict, rows: list):
    a = s["answerable"]
    print(f"\nANSWERABLE QUESTIONS: {a['n']}")
    print("  " + "   ".join(f"hit@{k} {a[f'hit@{k}']:.2f}" for k in KS) + f"   MRR {a['mrr']:.2f}")
    print(f"  (we currently send top {config.TOP_K} chunks to Qwen)")

    print("\nBY AUTHOR")
    for who, b in s["by_author"].items():
        print(f"  {who:<9} n={b['n']:<3} hit@3 {b['hit@3']:.2f}   MRR {b['mrr']:.2f}")

    print("\nBY TAG (hit@3; small n = weak evidence)")
    for tag, b in sorted(s["by_tag"].items(), key=lambda kv: kv[1]["hit@3"]):
        print(f"  {tag:<16} n={b['n']:<3} hit@3 {b['hit@3']:.2f}   MRR {b['mrr']:.2f}")

    if s["unfindable"]:
        print(f"\nUNFINDABLE (evidence is in NO single chunk: a chunking problem, not ranking): {s['unfindable']}")

    t = s["threshold"]
    if t["answerable_found_at_1"] and t["refuse_best_scores"]:
        print("\nSCORE CUT-OFF CHECK (3.3's question, with more data)")
        print(f"  answerable, correct chunk at rank 1: best scores {t['answerable_found_at_1'][0]:.3f} - {t['answerable_found_at_1'][1]:.3f}")
        print(f"  unanswerable questions:              best scores {t['refuse_best_scores'][0]:.3f} - {t['refuse_best_scores'][1]:.3f}")
        print(f"  {t['refuse_above_lowest_answerable']}/{t['n_refuse']} unanswerable questions score at or above "
              f"the LOWEST correct answerable one (0 = a clean cut-off is possible)")

    misses = [r for r in rows if r["type"] == "answer" and (not r["rank"] or r["rank"] > config.TOP_K)]
    if misses:
        print(f"\nNOT IN TOP {config.TOP_K} ({len(misses)}): what was retrieved instead")
        for r in misses:
            where = f"rank {r['rank']}" if r["rank"] else "not in top 10"
            why = "" if r["findable"] else "  [UNFINDABLE]"
            print(f"  {r['id']} ({where}){why}  {r['question'][:70]!r}")
            print(f"       evidence lives in: {', '.join(r['evidence_chunks'][:3]) or '(no chunk)'}")
            print(f"       got instead:       {', '.join(x['id'] for x in r['retrieved'][:3])}")


# ---------------------------------------------------------------------------
def save(kind: str, label: str, split: str, summary: dict, rows: list, seconds: float) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"{stamp}_{kind}{'_' + label if label else ''}.json"
    payload = {
        "timestamp": stamp, "kind": kind, "label": label, "split": split, "seconds": round(seconds, 1),
        "config": {"embed_model": config.EMBED_MODEL_ID, "chunk_max_tokens": config.CHUNK_MAX_TOKENS,
                   "top_k": config.TOP_K, "n_chunks": store.collection().count(),
                   "llm": config.LLM_MODEL, "prompt_style": getattr(config, "RAG_PROMPT_STYLE", None)},
        "summary": summary, "per_question": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    args = sys.argv[1:]
    kind = args[0] if args else ""
    split = args[args.index("--split") + 1] if "--split" in args else "test"
    label = args[args.index("--label") + 1] if "--label" in args else ""
    if kind != "retrieval":
        sys.exit(__doc__)

    print("Updating the store (unchanged files are skipped)...")
    store.index_folder()
    qs = load_golden(split)
    print(f"{len(qs)} questions ({split} split), {store.collection().count()} chunks in the store")

    t0 = time.time()
    rows = retrieval_rows(qs)
    summary = retrieval_summary(rows)
    dt = time.time() - t0
    print_retrieval(summary, rows)
    print(f"\n{dt:.1f}s.  Saved: {save('retrieval', label, split, summary, rows, dt).relative_to(ROOT)}")


if __name__ == "__main__":
    main()

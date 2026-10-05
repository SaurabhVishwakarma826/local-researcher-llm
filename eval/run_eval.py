"""
eval/run_eval.py — the evaluation harness.

    python eval/run_eval.py retrieval [--mode hybrid] [--rerank bge] [--split dev] [--label NAME]
    python eval/run_eval.py answer --style v3 [--mode hybrid] [--rerank bge] [--k 5] [--label NAME]
      --rerank none   turns reranking off even if config.RERANKER is set
    python eval/run_eval.py rescore NAME_OR_FILE                            # seconds, no Ollama
    python eval/run_eval.py compare A B                                     # seconds

NAME is the --label of an earlier run (the latest file with that label and split is used),
or a path to a file in eval/results/. Every run is saved to eval/results/.

Rules (Phase 4):
  - Choose between versions using the test split; never edit code until specific test questions pass.
  - Only widen a check to accept the SAME fact in other words, never a different fact.
"""

import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config, llm, rag, retrieve, store  # noqa: E402

GOLDEN = ROOT / "eval" / "golden.jsonl"
RESULTS = ROOT / "eval" / "results"
KS = [1, 3, 5, 10]
SUCCESS = {"correct", "correct_without_evidence", "refused"}


def norm(s: str) -> str:
    """Same normalisation as check_golden.py: lowercase, hyphens as spaces, one space."""
    return " ".join(s.lower().replace("-", " ").split())


def load_golden(split: str = "all") -> list:
    qs = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    return qs if split == "all" else [q for q in qs if q["split"] == split]


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs, p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))] if xs else 0.0


# ===========================================================================
# Retrieval (4.2)
# ===========================================================================
def retrieval_rows(qs: list, mode: str, reranker) -> list:
    everything = store.collection().get(include=["documents", "metadatas"])
    chunks = [(i, norm(d), m) for i, d, m in
              zip(everything["ids"], everything["documents"], everything["metadatas"])]
    rows = []
    for q in qs:
        t0 = time.time()
        hits = retrieve.search(q["question"], k=max(KS), source=q.get("only_source"),
                               mode=mode, reranker=reranker)
        row = {"id": q["id"], "by": q.get("by", "?"), "type": q["type"], "tags": q.get("tags", []),
               "question": q["question"], "seconds": round(time.time() - t0, 3),
               "top_score": round(hits[0].score, 4) if hits else None,
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
    secs = [r.get("seconds", 0) for r in rows]
    return {
        "answerable": block(ans),
        "seconds_per_question": {"mean": round(_mean(secs), 3), "max": round(max(secs), 3) if secs else 0},
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
    sp = s.get("seconds_per_question")
    if sp:
        print(f"  retrieval time per question: mean {sp['mean']:.2f}s, max {sp['max']:.2f}s")
    print("\nBY AUTHOR")
    for who, b in s["by_author"].items():
        print(f"  {who:<9} n={b['n']:<3} hit@3 {b['hit@3']:.2f}   MRR {b['mrr']:.2f}")
    print("\nBY TAG (hit@3; small n = weak evidence)")
    for tag, b in sorted(s["by_tag"].items(), key=lambda kv: kv[1]["hit@3"]):
        print(f"  {tag:<16} n={b['n']:<3} hit@3 {b['hit@3']:.2f}   MRR {b['mrr']:.2f}")
    if s["unfindable"]:
        print(f"\nUNFINDABLE (evidence is in NO single chunk: a chunking problem): {s['unfindable']}")
    t = s["threshold"]
    if t["answerable_found_at_1"] and t["refuse_best_scores"]:
        print("\nSCORE CUT-OFF CHECK")
        print(f"  answerable, correct chunk at rank 1: best scores {t['answerable_found_at_1'][0]:.3f} - {t['answerable_found_at_1'][1]:.3f}")
        print(f"  unanswerable questions:              best scores {t['refuse_best_scores'][0]:.3f} - {t['refuse_best_scores'][1]:.3f}")
        print(f"  {t['refuse_above_lowest_answerable']}/{t['n_refuse']} unanswerable questions score at or above "
              f"the LOWEST correct answerable one (0 = a clean cut-off is possible)")
    misses = [r for r in rows if r["type"] == "answer" and (not r["rank"] or r["rank"] > config.TOP_K)]
    if misses:
        print(f"\nNOT IN TOP {config.TOP_K} ({len(misses)}): what was retrieved instead")
        for r in misses:
            where = f"rank {r['rank']}" if r["rank"] else "not in top 10"
            print(f"  {r['id']} ({where}){'' if r['findable'] else '  [UNFINDABLE]'}  {r['question'][:70]!r}")
            print(f"       evidence lives in: {', '.join(r['evidence_chunks'][:3]) or '(no chunk)'}")
            print(f"       got instead:       {', '.join(x['id'] for x in r['retrieved'][:3])}")


# ===========================================================================
# Answers (4.3)
# ===========================================================================
REFUSAL_PATTERNS = ("i don't know", "i do not know", "do not specify", "does not specify", "not specified",
                    "do not mention", "does not mention", "not mentioned", "do not contain",
                    "does not contain", "no information", "not provided", "do not provide",
                    "does not provide", "cannot find", "can't find", "not stated")


def is_refusal(text: str) -> bool:
    t = text.lower().replace("\u2019", "'")
    return any(p in t for p in REFUSAL_PATTERNS)


def has(t: str, x: str) -> bool:
    """Match at a token start on normalised text: '5%' does NOT match inside '15%'."""
    return re.search(r"(?<![\w.])" + re.escape(norm(x)), t) is not None


def classify(q: dict, text: str, evidence_in_prompt: bool, n_cited: int) -> dict:
    """Everything that depends only on the answer text and the golden set, so saved
    answers can be RE-scored after the golden set is corrected (rescore)."""
    t = norm(text)
    refused = is_refusal(text)
    groups = q.get("must_include", [])
    met = [any(has(t, x) for x in alts) for alts in groups]
    violations = [w for w in q.get("must_not_include", []) if has(t, w)]
    if q["type"] == "answer":
        if all(met) and not violations:
            cat = "correct" if evidence_in_prompt else "correct_without_evidence"
        elif not evidence_in_prompt:
            cat = "retrieval_miss"
        elif any(met) and not violations:
            cat = "partial"
        elif refused:
            cat = "wrongly_refused"
        else:
            cat = "generation_miss"
        contradictory = refused and (all(met) or n_cited > 0)
    else:
        cat = ("refused" if refused and not violations
               else "refused_but_leaked" if refused else "answered_instead")
        contradictory = refused and n_cited > 0
    return {"category": cat, "refused": refused, "violations": violations,
            "facts_met": f"{sum(met)}/{len(met)}", "contradictory": contradictory}


def answer_rows(qs: list, style: str, k: int, mode: str, reranker) -> list:
    rows = []
    for n, q in enumerate(qs, 1):
        a = rag.answer(q["question"], k=k, source=q.get("only_source"), style=style,
                       mode=mode, reranker=reranker)
        row = {"id": q["id"], "by": q.get("by", "?"), "type": q["type"], "tags": q.get("tags", []),
               "question": q["question"], "answer": a.text,
               "prompt_ids": [h.id for h in a.hits], "cited": [h.id for h in a.cited],
               "bad_citations": a.bad_citations, "malformed": a.malformed,
               "prompt_tokens": a.prompt_tokens, "output_tokens": a.output_tokens,
               "seconds": round(a.seconds, 1)}
        in_prompt = True
        if q["type"] == "answer":
            ev = norm(q["evidence"])
            in_prompt = any(h.source == q["source"] and ev in norm(h.text) for h in a.hits)
            row.update(evidence_in_prompt=in_prompt,
                       cite_right_doc=any(h.source == q["source"] for h in a.cited),
                       cite_exact=any(h.source == q["source"] and ev in norm(h.text) for h in a.cited))
        row.update(classify(q, a.text, in_prompt, len(a.cited)))
        rows.append(row)
        flag = "  (contradictory)" if row["contradictory"] else ""
        print(f"  {n:>2}/{len(qs)} {q['id']:<4} {row['category']:<25} {a.seconds:>5.1f}s{flag}", flush=True)
    return rows


def answer_summary(rows: list) -> dict:
    ans = [r for r in rows if r["type"] == "answer"]
    ref = [r for r in rows if r["type"] == "refuse"]
    good = [r for r in ans if r["category"] in SUCCESS]
    by_tag, by_author = defaultdict(list), defaultdict(list)
    for r in rows:
        ok = r["category"] in SUCCESS
        by_author[(r["by"], r["type"])].append(ok)
        for t in r["tags"] or ["(untagged)"]:
            by_tag[(t, r["type"])].append(ok)
    secs = [r["seconds"] for r in rows]
    outs = [r["output_tokens"] for r in rows if r.get("output_tokens")]
    return {
        "answerable": {"n": len(ans), "correct": round(_mean(r in good for r in ans), 3),
                       "categories": dict(Counter(r["category"] for r in ans))},
        "refuse": {"n": len(ref), "refused_correctly": round(_mean(r["category"] == "refused" for r in ref), 3),
                   "categories": dict(Counter(r["category"] for r in ref))},
        "contradictory": sum(r.get("contradictory", False) for r in rows),
        "citations_on_correct_answers": {
            "n": len(good),
            "valid": round(_mean(bool(r["cited"]) for r in good), 3),
            "right_document": round(_mean(r.get("cite_right_doc", False) for r in good), 3),
            "exact_passage": round(_mean(r.get("cite_exact", False) for r in good), 3),
            "answers_with_invented_format": sum(bool(r["malformed"]) for r in rows)},
        "by_author": {f"{a} ({t})": {"n": len(v), "score": round(_mean(v), 3)} for (a, t), v in sorted(by_author.items())},
        "by_tag": {f"{g} ({t})": {"n": len(v), "score": round(_mean(v), 3)} for (g, t), v in sorted(by_tag.items())},
        "cost": {"mean_seconds": round(_mean(secs), 1), "median_seconds": _pct(secs, 0.5),
                 "p90_seconds": _pct(secs, 0.9), "max_seconds": max(secs) if secs else 0,
                 "mean_prompt_tokens": round(_mean(r["prompt_tokens"] for r in rows)),
                 "mean_output_tokens": round(_mean(outs)) if outs else None},
    }


def print_answers(s: dict, rows: list):
    a, r, c, co = s["answerable"], s["refuse"], s["citations_on_correct_answers"], s["cost"]
    print(f"\nANSWERABLE  {a['n']}:  correct {a['correct']:.2f}      {a['categories']}")
    print(f"REFUSE      {r['n']}:  refused correctly {r['refused_correctly']:.2f}      {r['categories']}")
    print(f"CONTRADICTORY replies (answer AND refuse, or refuse while citing): {s['contradictory']}")
    print(f"\nCITATIONS on the {c['n']} correct answers:  valid {c['valid']:.2f}   "
          f"right document {c['right_document']:.2f}   exact passage {c['exact_passage']:.2f}")
    print(f"  answers using an invented citation format: {c['answers_with_invented_format']}")
    print("\nBY AUTHOR (score = correct for answerable, refused for refuse)")
    for k_, v in s["by_author"].items():
        print(f"  {k_:<22} n={v['n']:<3} {v['score']:.2f}")
    print("\nBY TAG (lowest first; small n = weak evidence)")
    for k_, v in sorted(s["by_tag"].items(), key=lambda kv: kv[1]["score"]):
        print(f"  {k_:<28} n={v['n']:<3} {v['score']:.2f}")
    outs = f", {co['mean_output_tokens']} answer tokens" if co.get("mean_output_tokens") else ""
    print(f"\nCOST per question: mean {co['mean_seconds']}s, median {co['median_seconds']}s, "
          f"90% under {co['p90_seconds']}s, max {co['max_seconds']}s | "
          f"{co['mean_prompt_tokens']} prompt tokens{outs}")
    bad = [x for x in rows if x["category"] not in ("correct", "refused") or x.get("contradictory")]
    if bad:
        print(f"\nREAD THESE ({len(bad)}): not a plain success, or contradictory")
        for x in bad:
            extra = f"  violates {x['violations']}" if x["violations"] else ""
            flag = " (contradictory)" if x.get("contradictory") else ""
            print(f"\n  {x['id']} [{x['category']}{flag}] facts {x.get('facts_met', '?')}{extra}  {x['question'][:70]!r}")
            print(f"     -> {x['answer'][:260]!r}")


# ===========================================================================
# Saving, finding, rescoring, comparing (4.4)
# ===========================================================================
def save(kind: str, label: str, split: str, summary: dict, rows: list, seconds: float,
         cfg: dict = None) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"{stamp}_{kind}{'_' + label if label else ''}.json"
    base = {"embed_model": config.EMBED_MODEL_ID, "chunk_max_tokens": config.CHUNK_MAX_TOKENS,
            "top_k": config.TOP_K, "n_chunks": store.collection().count(),
            "llm": config.LLM_MODEL, "prompt_style": getattr(config, "RAG_PROMPT_STYLE", None),
            "retrieval_mode": getattr(config, "RETRIEVAL_MODE", "dense"),
            "reranker": getattr(config, "RERANKER", None)}
    base.update(cfg or {})
    payload = {"timestamp": stamp, "kind": kind, "label": label, "split": split,
               "seconds": round(seconds, 1), "config": base, "summary": summary, "per_question": rows}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def find_run(ref: str, split: str = "test") -> Path:
    p = Path(ref)
    for cand in (p, ROOT / p, RESULTS / p):
        if cand.is_file():
            return cand
    matches = []
    for f in sorted(RESULTS.glob(f"*_{ref}.json")):
        try:
            if json.loads(f.read_text(encoding="utf-8")).get("split") == split:
                matches.append(f)
        except json.JSONDecodeError:
            pass
    if not matches:
        sys.exit(f"No saved run with label {ref!r} and split {split!r} in {RESULTS}")
    return matches[-1]


def load_run(ref: str, split: str = "test") -> dict:
    path = find_run(ref, split)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_file"] = path.name
    return data


def reclassify(rows: list) -> tuple:
    """Re-score saved answers against the CURRENT golden set. evidence_in_prompt and
    cite_* depend on retrieval, not on must_include, so they are kept as saved."""
    golden = {q["id"]: q for q in load_golden()}
    out, changed = [], []
    for r in rows:
        q = golden.get(r["id"])
        if q is None:
            continue
        new = dict(r)
        new.update(classify(q, r["answer"], r.get("evidence_in_prompt", True), len(r["cited"])))
        if new["category"] != r["category"]:
            changed.append((r["id"], r["category"], new["category"]))
        out.append(new)
    return out, changed


def rescore(ref: str, split: str):
    old = load_run(ref, split)
    if old["kind"] != "answer":
        sys.exit("rescore only applies to 'answer' runs (retrieval runs are fast; just rerun them)")
    rows, changed = reclassify(old["per_question"])
    print(f"Rescored {old['_file']} against the current golden set (no Qwen calls).")
    print(f"{len(changed)} question(s) changed outcome:" if changed else "No outcomes changed.")
    for i, a, b in changed:
        print(f"  {i}: {a} -> {b}")
    summary = answer_summary(rows)
    print_answers(summary, rows)
    cfg = dict(old["config"], rescored_from=old["_file"])
    path = save("answer", f"{old['label']}_rescored" if old["label"] else "rescored",
                old["split"], summary, rows, 0.0, cfg)
    print(f"\nSaved: {path.relative_to(ROOT)}")


def compare(ref_a: str, ref_b: str, split: str):
    A, B = load_run(ref_a, split), load_run(ref_b, split)
    if A["kind"] != B["kind"]:
        sys.exit(f"Cannot compare a {A['kind']} run with a {B['kind']} run")
    print(f"A = {A['_file']}\n    {A['config']}\nB = {B['_file']}\n    {B['config']}\n")
    # One variable at a time: if several settings differ, the comparison cannot say which
    # one caused the change (6.2: a run labelled 'v1_7b' was really v3 + 7B).
    ignore = {"n_chunks", "rescored_from"}
    diffs = sorted(k for k in set(A["config"]) | set(B["config"])
                   if k not in ignore and A["config"].get(k) != B["config"].get(k))
    if len(diffs) > 1:
        print(f"!! WARNING: these runs differ in {len(diffs)} settings: "
              + ", ".join(f"{k} ({A['config'].get(k)} -> {B['config'].get(k)})" for k in diffs))
        print("!! Any change below cannot be attributed to one of them.\n")
    elif diffs:
        print(f"Only difference: {diffs[0]} ({A['config'].get(diffs[0])} -> {B['config'].get(diffs[0])})\n")
    if A["kind"] == "answer":
        # Both sides scored with the SAME checks, or golden-set edits show up as fake flips.
        for run in (A, B):
            run["per_question"], _ = reclassify(run["per_question"])
            run["summary"] = answer_summary(run["per_question"])
        print("(both runs re-scored against the current golden.jsonl)\n")

    def line(name, a, b, fmt="{:.2f}", lower_is_better=False):
        if a is None or b is None:
            print(f"  {name:<32} {str(a):>8} {str(b):>8}")
            return
        d = b - a
        better = (d < 0) if lower_is_better else (d > 0)
        mark = "" if abs(d) < 1e-9 else ("  better" if better else "  worse")
        print(f"  {name:<32} {fmt.format(a):>8} {fmt.format(b):>8}   {'+' if d > 0 else ''}{fmt.format(d)}{mark}")

    sa, sb = A["summary"], B["summary"]
    print(f"  {'':<32} {'A':>8} {'B':>8}   change")
    if A["kind"] == "retrieval":
        for k in KS:
            line(f"hit@{k}", sa["answerable"][f"hit@{k}"], sb["answerable"][f"hit@{k}"])
        line("MRR", sa["answerable"]["mrr"], sb["answerable"]["mrr"])
        ra = {r["id"]: r for r in A["per_question"] if r["type"] == "answer"}
        rb = {r["id"]: r for r in B["per_question"] if r["type"] == "answer"}
        moved = [(i, ra[i]["rank"], rb[i]["rank"]) for i in ra if i in rb and ra[i]["rank"] != rb[i]["rank"]]
        print(f"\nRANK CHANGES ({len(moved)})  (None = not in top 10)")
        for i, x, y in sorted(moved, key=lambda t: (t[2] or 99) - (t[1] or 99)):
            print(f"  {i}: {x} -> {y}")
        return

    ca, cb = sa["citations_on_correct_answers"], sb["citations_on_correct_answers"]
    line("answerable correct", sa["answerable"]["correct"], sb["answerable"]["correct"])
    line("refused correctly", sa["refuse"]["refused_correctly"], sb["refuse"]["refused_correctly"])
    line("contradictory replies", sa.get("contradictory"), sb.get("contradictory"), "{:.0f}", True)
    line("citations: right document", ca["right_document"], cb["right_document"])
    line("citations: exact passage", ca["exact_passage"], cb["exact_passage"])
    line("invented citation formats", ca["answers_with_invented_format"], cb["answers_with_invented_format"], "{:.0f}", True)
    line("mean seconds", sa["cost"]["mean_seconds"], sb["cost"]["mean_seconds"], "{:.1f}", True)
    line("mean prompt tokens", sa["cost"]["mean_prompt_tokens"], sb["cost"]["mean_prompt_tokens"], "{:.0f}", True)

    ra = {r["id"]: r for r in A["per_question"]}
    rb = {r["id"]: r for r in B["per_question"]}
    common = [i for i in ra if i in rb]
    improved = [i for i in common if ra[i]["category"] not in SUCCESS and rb[i]["category"] in SUCCESS]
    regressed = [i for i in common if ra[i]["category"] in SUCCESS and rb[i]["category"] not in SUCCESS]
    other = [i for i in common if i not in improved + regressed and ra[i]["category"] != rb[i]["category"]]
    print(f"\nFIXED in B ({len(improved)}):")
    for i in improved:
        print(f"  {i}: {ra[i]['category']} -> {rb[i]['category']}   {rb[i]['question'][:60]!r}")
    print(f"\nBROKEN in B ({len(regressed)}):  read these answers")
    for i in regressed:
        print(f"  {i}: {ra[i]['category']} -> {rb[i]['category']}   {rb[i]['question'][:60]!r}")
        print(f"       B said: {rb[i]['answer'][:200]!r}")
    if other:
        print(f"\nOTHER CHANGES ({len(other)}):")
        for i in other:
            print(f"  {i}: {ra[i]['category']} -> {rb[i]['category']}")
    print(f"\nNet: {len(improved) - len(regressed):+d} questions. "
          "Answers are deterministic (temperature 0), but with 44 questions a\n"
          "difference of 1-2 is weak evidence: it could depend on which questions we happened to write.")


# ===========================================================================
def main():
    args = sys.argv[1:]
    kind = args[0] if args else ""
    split = args[args.index("--split") + 1] if "--split" in args else "test"
    label = args[args.index("--label") + 1] if "--label" in args else ""

    if kind == "rescore" and len(args) > 1:
        return rescore(args[1], split)
    if kind == "compare" and len(args) > 2:
        return compare(args[1], args[2], split)
    if kind not in ("retrieval", "answer"):
        sys.exit(__doc__)

    print("Updating the store (unchanged files are skipped)...")
    store.index_folder()
    qs = load_golden(split)
    print(f"{len(qs)} questions ({split} split), {store.collection().count()} chunks in the store")

    mode = args[args.index("--mode") + 1] if "--mode" in args else config.RETRIEVAL_MODE
    reranker = args[args.index("--rerank") + 1] if "--rerank" in args else getattr(config, "RERANKER", None)
    reranker = None if reranker in (None, "none") else reranker
    print(f"retrieval mode: {mode}, reranker: {reranker or 'off'}")
    t0 = time.time()
    if kind == "retrieval":
        if reranker:
            print(f"loading reranker {reranker!r} (the first time downloads it)...")
            retrieve.rerank("warm up", [store.Hit("x", "warm up", "x", 0, "", 0.0)], 1, reranker)
        rows = retrieval_rows(qs, mode, reranker)
        summary = retrieval_summary(rows)
        print_retrieval(summary, rows)
        cfg = {"retrieval_mode": mode, "reranker": reranker}
    else:
        style = args[args.index("--style") + 1] if "--style" in args else config.RAG_PROMPT_STYLE
        k = int(args[args.index("--k") + 1]) if "--k" in args else config.TOP_K
        if not llm.is_up():
            sys.exit(f"Ollama is not running at {config.OLLAMA_HOST}. Start it with: ollama serve")
        llm.chat([{"role": "user", "content": "hi"}], num_predict=1)       # warm up
        print(f"style {style}, top-{k} chunks. About {len(qs) * 12 // 60} min. Progress:")
        rows = answer_rows(qs, style, k, mode, reranker)
        summary = answer_summary(rows)
        print_answers(summary, rows)
        cfg = {"prompt_style": style, "top_k": k, "retrieval_mode": mode, "reranker": reranker}
    dt = time.time() - t0
    path = save(kind, label, split, summary, rows, dt, cfg)
    print(f"\n{dt:.0f}s.  Saved: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

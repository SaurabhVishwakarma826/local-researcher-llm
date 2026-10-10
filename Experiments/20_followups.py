"""
Phase 7 / Topic 7.3 — Follow-up questions: raw vs concat vs rewrite (retrieval only, ~3 min).
Part A: follow-ups on the same topic. Part B: follow-ups right after a topic switch.
Part C: the topic-switch question itself (added after concat failed it in the live chat).

    python 20_followups.py

For each follow-up, the evidence phrase must reach the top 3 retrieved chunks (as in 4.2).
Uses the retrieval settings in config.py (hybrid + reranker).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, llm, retrieve  # noqa: E402
from app.rewrite import concat, rewrite, union_search  # noqa: E402

# (first question, [(follow-up, evidence phrase in the PDF), ...])
CONVERSATIONS = [
    ("What is the RDF substitution target in SWM 2026?",
     [("and what is the starting value?", "increasing substitution from 5% to 15%"),
      ("which industries does it apply to?", "including cement plants")]),
    ("Who counts as a bulk waste generator?",
     [("what do they have to do with their wet waste?", "Bulk generators are mandated to process biodegradable"),
      ("and if they don't have space?", "off-site processing only through the formal issuance")]),
    ("What are MRFs?",
     [("what fire safety measures does the paper suggest for them?", "hotspot monitoring")]),
    ("What does SWM 2026 say about hilly areas?",
     [("what can they charge tourists?", "levy user fees on tourists"),
      ("and what must hotels do there?", "Hotels and restaurants must undertake decentralised wet-waste processing")]),
    ("What projects has Saurabh built?",
     [("which one used Flask?", "URL Shortener API using Flask and MongoDB"),
      ("and what database did it use?", "URL Shortener API using Flask and MongoDB")]),
    ("Tell me about legacy dumpsites in the rules",
     [("how often must progress be reported?", "quarterly reporting")]),
]


# Topic switches: the follow-up refers to the MOST RECENT topic, not the first one.
# Concat-all keeps the old topic in the query; concat-last and rewrite should not.
TOPIC_SWITCHES = [
    (["What does SWM 2026 say about hilly areas?", "What is the RDF substitution target?"],
     "and what is the starting value?", "increasing substitution from 5% to 15%"),
    (["What projects has Saurabh built?", "What are MRFs?"],
     "what fire safety measures are suggested for them?", "hotspot monitoring"),
    (["Who counts as a bulk waste generator?", "Tell me about legacy dumpsites in the rules"],
     "how often must progress be reported?", "quarterly reporting"),
    (["What are MRFs?", "What projects has Saurabh built?"],
     "which one used Flask?", "URL Shortener API using Flask and MongoDB"),
    (["Tell me about legacy dumpsites in the rules", "What does SWM 2026 say about hilly areas?"],
     "and what must hotels do there?", "Hotels and restaurants must undertake decentralised wet-waste processing"),
]
# Part C (added after the live chat test): the topic-switch QUESTION ITSELF. Parts A and B
# never searched this, and concat failed it in real use ("hilly areas" after RDF).
SWITCH_QUESTIONS = [
    (["I am Saurabh from Mumbai, working as an AI engineer in Hyderabad",
      "What is the RDF substitution target in SWM 2026?"],
     "what does it say about hilly areas?", "hilly areas and islands"),
    (["What are MRFs?", "what fire safety measures does the paper suggest for them?"],
     "who counts as a bulk waste generator?", "three measurable and verifiable thresholds"),
    (["What projects has Saurabh built?", "which one used Flask?"],
     "how often must legacy dumpsite progress be reported?", "quarterly reporting"),
    (["What does SWM 2026 say about hilly areas?", "and what must hotels do there?"],
     "what was Saurabh's CGPA?", "CGPA : 8.53"),
    (["Who counts as a bulk waste generator?", "and if they don't have space?"],
     "what are the RDF substitution targets?", "increasing substitution from 5% to 15%"),
]
METHODS = ("raw", "concat-last", "concat-all", "rewrite", "union")


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("-", " ").split())


def hit(hits: list, evidence: str) -> bool:
    ev = _norm(evidence)
    return any(ev in _norm(h.text) for h in hits)


def queries(previous: list, msg: str) -> dict:
    """The retrieved chunks for each method (union searches twice, so it returns hits directly)."""
    t0 = time.time()
    rq = rewrite(previous, msg)
    secs = time.time() - t0
    k = config.TOP_K
    return {"raw": retrieve.search(msg, k=k), "concat-last": retrieve.search(concat(previous, msg, n=1), k=k),
            "concat-all": retrieve.search(concat(previous, msg), k=k), "rewrite": retrieve.search(rq, k=k),
            "union": union_search(previous, msg, k), "_rewritten": rq, "_secs": secs}


def score(qs: dict, evidence: str, totals: dict) -> str:
    marks = {k: hit(qs[k], evidence) for k in METHODS}
    for k, v in marks.items():
        totals[k] += v
    return "  ".join(f"{k} {'OK' if v else '--'}" for k, v in marks.items())


def main():
    if not llm.is_up():
        sys.exit("Ollama is not running. Start it with: ollama serve")
    print(f"model {config.LLM_MODEL}, retrieval {config.RETRIEVAL_MODE} + {config.RERANKER}, top {config.TOP_K}\n")
    secs = []

    print("=== A. Follow-ups on the SAME topic")
    same = dict.fromkeys(METHODS, 0)
    n_same = 0
    for first, followups in CONVERSATIONS:
        previous = [first]
        for msg, evidence in followups:
            qs = queries(previous, msg)
            secs.append(qs["_secs"])
            print(f"  {msg!r:<58} {score(qs, evidence, same)}")
            previous.append(msg)
            n_same += 1

    print("\n=== B. Follow-ups right after a TOPIC SWITCH")
    switch = dict.fromkeys(METHODS, 0)
    for previous, msg, evidence in TOPIC_SWITCHES:
        qs = queries(previous, msg)
        secs.append(qs["_secs"])
        print(f"  after {previous[-1]!r}\n    {msg!r:<56} {score(qs, evidence, switch)}")
        print(f"      rewritten as: {qs['_rewritten']!r}")

    print("\n=== C. The topic-switch QUESTION ITSELF (the case the live chat failed)")
    sw_q = dict.fromkeys(METHODS, 0)
    for previous, msg, evidence in SWITCH_QUESTIONS:
        qs = queries(previous, msg)
        secs.append(qs["_secs"])
        print(f"  after {previous[-1]!r}\n    {msg!r:<56} {score(qs, evidence, sw_q)}")

    print(f"\n{'':<16}" + "".join(f"{m:>13}" for m in METHODS))
    print(f"  A same topic  " + "".join(f"{same[m]:>10}/{n_same}" for m in METHODS))
    print(f"  B after switch" + "".join(f"{switch[m]:>10}/{len(TOPIC_SWITCHES)}" for m in METHODS))
    print(f"  C the switch  " + "".join(f"{sw_q[m]:>10}/{len(SWITCH_QUESTIONS)}" for m in METHODS))
    total = n_same + len(TOPIC_SWITCHES) + len(SWITCH_QUESTIONS)
    print(f"  TOTAL         " + "".join(f"{same[m] + switch[m] + sw_q[m]:>10}/{total}" for m in METHODS))
    print(f"\nRewrite cost: {sum(secs) / len(secs):.1f}s per question. union = two searches (~0.3 s extra).")
    print("Small store caveat: with only 2 PDFs, one distinctive word ('Flask') can rescue a vague")
    print("query. With many documents, raw and concat would both do worse than here.")


if __name__ == "__main__":
    main()

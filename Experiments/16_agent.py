"""
Phase 6 / Topic 6.2 — The agent loop, tested against what goes wrong.

    python 16_agent.py 1   # a question that needs search AND calculation
    python 16_agent.py 2   # a tool that fails once: does the agent retry?
    python 16_agent.py 3   # a tool that returns nothing: does it invent an answer?
    python 16_agent.py 4   # a tool that always fails: does the loop stop?
    python 16_agent.py 5   # dates: current_date then days_between

Needs Ollama running.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import agent, llm  # noqa: E402
from app.tools import REGISTRY, Tool, TransientToolError  # noqa: E402


def show(r: agent.AgentResult, expect: str):
    print(f"\n-> {r.stopped}, {r.model_calls} model calls, {r.seconds:.1f}s, "
          f"largest prompt {r.max_prompt_tokens} tokens")
    print(f"   unverified numbers flagged: {r.unverified or 'none'}")
    print(f"   expected: {expect}")


# ---------------------------------------------------------------------------
def exp1_multi_step():
    q = ("What RDF substitution percentage do the SWM 2026 rules reach? If a cement plant "
         "burns 2,400 tonnes of fuel a year, how many tonnes would be RDF at that level?")
    print(f"Q: {q}\n")
    r = agent.run(q, on_step=agent.print_step)
    show(r, "search finds 15%, calculator computes 0.15 * 2400 = 360 tonnes")


# ---------------------------------------------------------------------------
def exp2_flaky():
    """The calculator fails on its FIRST call only, like a network blip."""
    state = {"calls": 0}
    real = REGISTRY["calculator"]

    def flaky(args):
        state["calls"] += 1
        if state["calls"] == 1:
            raise TransientToolError("calculator service temporarily unavailable")
        return real.fn(args)

    reg = dict(REGISTRY, calculator=Tool(real.schema, flaky))
    q = "What is 17.5% of 48,317?"
    print(f"Q: {q}   (exact: 8455.475; the calculator will fail once)\n")
    r = agent.run(q, registry=reg, on_step=agent.print_step)
    show(r, "our code retries silently; the model sees 8455.475 and never knows it failed once")


# ---------------------------------------------------------------------------
def exp3_empty():
    """Search returns nothing at all. The honest answer is 'I could not find it'."""
    real = REGISTRY["search_documents"]
    reg = dict(REGISTRY, search_documents=Tool(real.schema, lambda args: "No results."))
    q = "According to the SWM 2026 paper, what is the RDF substitution target for cement plants?"
    print(f"Q: {q}   (search is rigged to return 'No results.')\n")
    r = agent.run(q, registry=reg, on_step=agent.print_step)
    show(r, "says it could not find the answer. If it says '15%' or any number, that came "
            "from MEMORY or invention, not from the documents")


# ---------------------------------------------------------------------------
def exp4_always_fails():
    real = REGISTRY["calculator"]

    def broken(args):
        raise TransientToolError("calculator is down")

    reg = dict(REGISTRY, calculator=Tool(real.schema, broken))
    q = "Calculate (3847 * 29) - 1562 using the calculator."
    print(f"Q: {q}   (the calculator ALWAYS fails; exact answer 110001)\n")
    r = agent.run(q, registry=reg, on_step=agent.print_step)
    show(r, "our code retries, then reports failure; an honest answer that the tool failed. "
            "If the model computes a number anyway, it must be FLAGGED as unverified")


def exp5_dates():
    q = "How many days are there between today and 1 April 2027?"
    print(f"Q: {q}\n")
    r = agent.run(q, on_step=agent.print_step)
    show(r, "current_date -> today, THEN days_between(today, 2027-04-01) -> one exact number")


EXPERIMENTS = {"1": exp1_multi_step, "2": exp2_flaky, "3": exp3_empty, "4": exp4_always_fails,
               "5": exp5_dates}

if __name__ == "__main__":
    if not llm.is_up():
        sys.exit("Ollama is not running. Start it with: ollama serve")
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    if key not in EXPERIMENTS:
        sys.exit(__doc__)
    EXPERIMENTS[key]()

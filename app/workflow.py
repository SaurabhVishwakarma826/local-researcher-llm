"""
app/workflow.py — a fixed WORKFLOW in LangGraph (Phase 6.2, option B).

Why not an agent: on 11 multi-step cases, Qwen 3B as a free agent got 1/8 multi-step
right, with or without a planning step. It failed at choosing and ORDERING steps.
Here our code fixes the order; the model only does small single jobs:

    START -> router -> [retrieve] -> [dates] -> [math] -> answer -> END
                         (each optional step is skipped if the router says so)

  router    model answers 3 yes/no questions, forced into JSON by Ollama's `format`
  retrieve  CODE only: Phase 5 hybrid + reranker search. ALWAYS runs when
            config.WORKFLOW_ALWAYS_RETRIEVE is True (first run: the router said "math only"
            for 5 document questions, so the facts came from memory). Searching needlessly
            costs ~0.3 s; skipping a needed search means answering from memory.
  dates     model names two dates (may write TODAY); CODE counts the days
  math      model writes ONE arithmetic expression; CODE calculates (1 retry on error)
  answer    model writes the reply, told to copy computed numbers exactly

Returns an agent.AgentResult so eval/agent_eval.py can score it like the agent.

    python -m app.workflow "your question"
    python -m app.workflow --graph          # print the graph as a Mermaid diagram
"""

import json
import sys
import time
from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app import config, llm, retrieve
from app.agent import AgentResult, Step, unverified_numbers
from app.prompt import sanitize
from app.tools import safe_eval

ORDER = ["retrieve", "dates", "math"]          # fixed: search always before any calculation


class State(TypedDict, total=False):
    question: str
    needs_documents: bool
    needs_math: bool
    needs_dates: bool
    context: str
    hits: list
    today: str
    date_start: str
    date_end: str
    days: int
    expression: str
    calc_result: str
    answer: str
    steps: list           # Step records, for the trace and the evaluation
    model_calls: int


# ---------------------------------------------------------------------------
def _ask_json(system: str, user: str, schema: dict) -> tuple:
    """One model call whose output MUST match the JSON schema (Ollama `format`, 1.4)."""
    text, s = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                       fmt=schema, num_predict=200)
    try:
        return json.loads(text), s
    except json.JSONDecodeError:
        return {}, s


def _add(state: State, step: Step, calls: int = 0) -> dict:
    return {"steps": state.get("steps", []) + [step],
            "model_calls": state.get("model_calls", 0) + calls}


def _facts(state: State) -> str:
    parts = []
    if state.get("context"):
        parts.append(f"Facts from the documents:\n{state['context']}")
    computed = []
    if state.get("days") is not None and state.get("date_start"):
        computed.append(f"- days from {state['date_start']} to {state['date_end']}: {state['days']}")
    if state.get("calc_result") and not state["calc_result"].startswith("ERROR"):
        computed.append(f"- {state['expression']} = {state['calc_result']}")
    if computed:
        parts.append("Computed results (exact):\n" + "\n".join(computed))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
ROUTER_SYSTEM = ("Decide what is needed to answer the user's question. "
                 "needs_documents: facts from the user's PDFs (the resume of Saurabh Vishwakarma, "
                 "or a paper on India's Solid Waste Management Rules 2026). "
                 "needs_math: any arithmetic, percentage or ratio. "
                 "needs_dates: today's date or counting days between dates.")
ROUTER_SCHEMA = {"type": "object",
                 "properties": {"needs_documents": {"type": "boolean"},
                                "needs_math": {"type": "boolean"},
                                "needs_dates": {"type": "boolean"}},
                 "required": ["needs_documents", "needs_math", "needs_dates"]}


def router(state: State) -> dict:
    t0 = time.time()
    out, s = _ask_json(ROUTER_SYSTEM, state["question"], ROUTER_SCHEMA)
    route = {k: bool(out.get(k, False)) for k in ("needs_documents", "needs_math", "needs_dates")}
    if getattr(config, "WORKFLOW_ALWAYS_RETRIEVE", False):
        route["needs_documents"] = True
    summary = ", ".join(k.replace("needs_", "") for k, v in route.items() if v) or "nothing"
    return {**route, "today": date.today().isoformat(),
            **_add(state, Step("plan", result=f"route: {summary}", seconds=time.time() - t0,
                               prompt_tokens=s["prompt_tokens"]), calls=1)}


def node_retrieve(state: State) -> dict:
    t0 = time.time()
    hits = retrieve.search(state["question"], k=config.TOP_K)
    context = "\n\n".join(f"[{i}] {h.source}, page {h.page}\n{sanitize(h.text)}"
                          for i, h in enumerate(hits, 1))
    return {"hits": hits, "context": context,
            **_add(state, Step("tool", name="search_documents", args={"query": state["question"]},
                               result=context or "No results.", seconds=time.time() - t0))}


DATES_SYSTEM = ("Give the two dates needed to count the days in the user's question, as YYYY-MM-DD. "
                "Write TODAY for today's date. Use dates from the facts if the question refers to them. "
                "Reply only with the JSON.")
DATES_SCHEMA = {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
                "required": ["start", "end"]}


def node_dates(state: State) -> dict:
    t0 = time.time()
    today = state["today"]
    user = f"Today is {today}.\n\n{_facts(state)}\n\nQuestion: {state['question']}"
    out, s = _ask_json(DATES_SYSTEM, user, DATES_SCHEMA)

    def parse(v):
        v = str(v or "").strip()
        return date.fromisoformat(today if v.upper() == "TODAY" else v)

    steps = state.get("steps", []) + [Step("tool", name="current_date", result=today)]
    try:
        a, b = parse(out.get("start")), parse(out.get("end"))
        days = abs((b - a).days)
        steps.append(Step("tool", name="days_between", args=out, result=str(days),
                          seconds=time.time() - t0, prompt_tokens=s["prompt_tokens"]))
        return {"date_start": a.isoformat(), "date_end": b.isoformat(), "days": days, "steps": steps,
                "model_calls": state.get("model_calls", 0) + 1}
    except ValueError:
        steps.append(Step("tool", name="days_between", args=out,
                          result=f"ERROR: could not read dates {out}", seconds=time.time() - t0))
        return {"steps": steps, "model_calls": state.get("model_calls", 0) + 1}


MATH_SYSTEM = ("Write ONE arithmetic expression that computes what the user's question asks, using "
               "numbers from the question and the facts. Use only numbers and + - * / ( ). Write "
               "percentages as decimals (15% -> 0.15) or divide by 100. Reply only with the JSON.")
MATH_SCHEMA = {"type": "object", "properties": {"expression": {"type": "string"}},
               "required": ["expression"]}


def node_math(state: State) -> dict:
    t0 = time.time()
    base = f"{_facts(state)}\n\nQuestion: {state['question']}"
    out, s = _ask_json(MATH_SYSTEM, base, MATH_SCHEMA)
    calls, expr = 1, str(out.get("expression", ""))
    for attempt in range(2):                     # one retry, with the error shown
        try:
            v = safe_eval(expr)
            result = str(round(v, 10)) if isinstance(v, float) else str(v)
            break
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
            result = f"ERROR: {e}"
            if attempt == 0:
                out, s = _ask_json(MATH_SYSTEM, f"{base}\n\nYour expression {expr!r} failed: {e}. "
                                               "Write a corrected one.", MATH_SCHEMA)
                calls, expr = calls + 1, str(out.get("expression", ""))
    return {"expression": expr, "calc_result": result,
            **_add(state, Step("tool", name="calculator", args={"expression": expr}, result=result,
                               seconds=time.time() - t0, prompt_tokens=s["prompt_tokens"]), calls=calls)}


ANSWER_SYSTEM = ("Answer the user's question briefly. If facts or computed results are given, use ONLY "
                 "them, and copy computed numbers exactly: do not recalculate. If they do not contain "
                 "the answer, say so. If nothing is given, just reply helpfully.")


def node_answer(state: State) -> dict:
    facts = _facts(state)
    user = f"{facts}\n\nQuestion: {state['question']}" if facts else state["question"]
    text, s = llm.chat([{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
                       num_predict=config.AGENT_MAX_TOKENS)
    return {"answer": text.strip(),
            **_add(state, Step("answer", result=text.strip(), seconds=s["total_s"],
                               prompt_tokens=s["prompt_tokens"]), calls=1)}


# ---------------------------------------------------------------------------
def _next_after(current: str):
    """Conditional edge: the next ENABLED step after `current`, in the fixed ORDER."""
    flag = {"retrieve": "needs_documents", "dates": "needs_dates", "math": "needs_math"}

    def pick(state: State) -> str:
        later = ORDER[ORDER.index(current) + 1:] if current in ORDER else ORDER
        return next((n for n in later if state.get(flag[n])), "answer")
    return pick


def build():
    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("retrieve", node_retrieve)
    g.add_node("dates", node_dates)
    g.add_node("math", node_math)
    g.add_node("answer", node_answer)
    g.add_edge(START, "router")
    for node in ["router"] + ORDER:
        # Only FORWARD destinations: the drawing must show what can actually happen.
        later = ORDER[ORDER.index(node) + 1:] if node in ORDER else ORDER
        g.add_conditional_edges(node, _next_after(node), {n: n for n in later + ["answer"]})
    g.add_edge("answer", END)
    return g.compile()


_graph = None


def run(question: str, on_step=None) -> AgentResult:
    global _graph
    _graph = _graph or build()
    t0 = time.time()
    final = _graph.invoke({"question": question, "steps": [], "model_calls": 0})
    res = AgentResult(question=question, answer=final.get("answer", ""), stopped="answered",
                      steps=final["steps"], model_calls=final["model_calls"],
                      max_prompt_tokens=max((s.prompt_tokens for s in final["steps"]), default=0))
    res.plan = next((s.result for s in final["steps"] if s.kind == "plan"), "")
    # Check against everything the answer step was shown: documents AND computed lines
    # (expression, dates), not just tool results (0.15 in '0.15 * 2400 = 360' was flagged).
    sources = [question, _facts(final),
               *[s.result for s in final["steps"] if s.kind == "tool" and not s.result.startswith("ERROR")]]
    res.unverified = unverified_numbers(res.answer, sources)
    if res.unverified:
        nums = ", ".join(f"{n:g}" for n in res.unverified)
        res.answer += (f"\n\n[Warning: {nums} did not come from the documents, a computed result "
                       "or the question. Treat as unverified.]")
    res.seconds = time.time() - t0
    if on_step:
        for s in res.steps:
            on_step(s)
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--graph" in args:
        print(build().get_graph().draw_mermaid())
        sys.exit()
    if not args:
        sys.exit(__doc__)
    if not llm.is_up():
        sys.exit("Ollama is not running. Start it with: ollama serve")
    from app.agent import print_step
    r = run(args[0], on_step=print_step)
    print(f"\n{r.model_calls} model call(s), {r.seconds:.1f}s, largest prompt {r.max_prompt_tokens} tokens")
    if r.unverified:
        print(f"UNVERIFIED numbers in the answer: {r.unverified}")

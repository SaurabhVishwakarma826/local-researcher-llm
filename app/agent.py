"""
app/agent.py — a hand-built agent loop (Phase 6.2, ReAct pattern).

    model -> tool call -> our code runs it -> observation -> model -> ... -> answer

Safeguards, each against a failure SEEN in 16_agent.py:
  - one tool call per step: the model once asked for search AND calculator in the same
                   step, and calculated with a number it had not seen yet (24 instead of 360).
  - number check:  numbers in the final answer must appear in the question or a successful
                   tool result; otherwise the answer is flagged. After a tool failure the model
                   did the maths itself and got it wrong (8685.525, 109901) with no warning.
  - temporary tool failures are retried by our code (app/tools.py), not left to the model.
  - step limit:    a confused model cannot loop forever. When reached, ONE final
                   call WITHOUT tools forces an answer from what it has.
  - repeat guard:  the same tool + same arguments again gets an error observation.
  - errors are observations (app/tools.run_tool), never crashes.
  - token counts come from OLLAMA per call: 6.1 showed our own counter is ~55
    tokens off once tool schemas are in the prompt.
"""

import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

from app import config, llm
from app.tools import REGISTRY, run_tool

SYSTEM = ("You are a research assistant with tools. Use a tool only when you need it, "
          "and call ONE tool at a time. Never calculate numbers yourself: use the calculator. "
          "If a tool fails, say so instead of guessing. "
          "When you have enough information, answer the user directly and briefly.")

PLAN_REQUEST = ("Before using any tools, write a short numbered plan: which tool to call, in which "
                "order, and which earlier result each step needs. If no tool is needed, write "
                "'No tools needed.' Do NOT answer the question yet.\n\nAvailable tools:\n{tools}")

FOLLOW_PLAN = ("\n\nFollow this plan, one tool call at a time, using each result in the next step. "
               "Skip it if the plan says no tools are needed.\n\nPlan:\n{plan}")

ONE_AT_A_TIME = ("SKIPPED: call one tool at a time. Wait for the result of the first call, "
                 "then decide the next call using that result.")

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list:
    """(value, decimals, is_percent) for each number: '2,333.27' -> (2333.27, 2, False)."""
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group()
        try:
            pct = text[m.end():m.end() + 2].lstrip().startswith("%")
            out.append((float(raw.replace(",", "")), len(raw.split(".")[1]) if "." in raw else 0, pct))
        except ValueError:
            pass
    return out


def unverified_numbers(answer: str, sources: list) -> list:
    """Numbers in the answer that appear in no source (question or successful tool result).
    A number written to d decimals counts as verified if a source value ROUNDS to it
    (6.2: the calculator said 2333.265, the model wrote 2333.27, and that was flagged).
    Integers below 10 are ignored (list markers, 'one tool', etc.)."""
    known = [v for src in sources for v, _, _ in _numbers(src)]
    out = []
    for n, d, pct in _numbers(answer):
        if n < 10 and n == int(n):
            continue
        # A percentage may restate a fraction: '65%' is verified by 0.65 (6.2, a04).
        candidates = known + ([k * 100 for k in known] if pct else [])
        # Within half a unit of the last decimal written: 2333.27 covers 2333.265...2333.275.
        # (Not round(): 2333.265 is stored as 2333.26499..., so round() gives 2333.26.)
        if not any(abs(n - k) <= 0.5 * 10 ** -d + 1e-9 for k in candidates):
            out.append(n)
    return sorted(set(out))


FINAL_NUDGE = ("You have used all available tool steps. Do not call any more tools. "
               "Answer the original question now using only the information above. "
               "If it is not enough, say so.")


@dataclass
class Step:
    kind: str               # "plan" | "tool" | "answer"
    name: str = ""
    args: dict = None
    result: str = ""
    seconds: float = 0.0
    prompt_tokens: int = 0


@dataclass
class AgentResult:
    question: str
    answer: str
    stopped: str            # "answered" | "step_limit"
    steps: list = field(default_factory=list)
    model_calls: int = 0
    seconds: float = 0.0
    max_prompt_tokens: int = 0
    unverified: list = field(default_factory=list)   # numbers with no tool/question source
    plan: str = ""                                   # plan-then-execute only


def run(question: str, registry: dict = None, max_steps: int = None, on_step=None,
        plan: bool = False) -> AgentResult:
    """plan=True: plan-then-execute. One extra model call, WITHOUT tools, writes the steps
    first; the loop then follows them (6.2: the plain loop called the calculator with an
    invented 50% before searching for the real 15%)."""
    registry = registry if registry is not None else REGISTRY
    max_steps = max_steps or config.AGENT_MAX_STEPS
    schemas = [t.schema for t in registry.values()]
    res = AgentResult(question=question, answer="", stopped="answered")
    seen = Counter()
    t0 = time.time()

    system = SYSTEM
    if plan:
        tool_list = "\n".join(f"- {n}: {t.schema['function']['description']}" for n, t in registry.items())
        m, s = llm.chat_message([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": f"{question}\n\n" + PLAN_REQUEST.format(tools=tool_list)}],
                                num_predict=config.AGENT_MAX_TOKENS)
        res.model_calls += 1
        res.plan = (m.get("content") or "").strip()
        system = SYSTEM + FOLLOW_PLAN.format(plan=res.plan)
        if on_step:
            on_step(Step("plan", result=res.plan, seconds=s["total_s"], prompt_tokens=s["prompt_tokens"]))
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]

    def call(with_tools: bool, extra=None):
        m, s = llm.chat_message(msgs + (extra or []), tools=schemas if with_tools else None,
                                num_predict=config.AGENT_MAX_TOKENS)
        res.model_calls += 1
        res.max_prompt_tokens = max(res.max_prompt_tokens, s["prompt_tokens"])
        return m, s

    for _ in range(max_steps):
        msg, s = call(with_tools=True)
        calls = msg.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": calls})
        if not calls:
            res.answer = (msg.get("content") or "").strip()
            step = Step("answer", result=res.answer, seconds=s["total_s"], prompt_tokens=s["prompt_tokens"])
            res.steps.append(step)
            if on_step:
                on_step(step)
            break
        for i, c in enumerate(calls):
            name = c["function"]["name"]
            args = c["function"].get("arguments") or {}
            if i > 0:
                result = ONE_AT_A_TIME            # only the first call of a step is run
            else:
                key = (name, json.dumps(args, sort_keys=True))
                seen[key] += 1
                if seen[key] > config.AGENT_MAX_REPEATS:
                    result = ("ERROR: you already called this tool with exactly these arguments. "
                              "Use the earlier result, try something different, or answer.")
                else:
                    result = run_tool(registry, name, args)
            msgs.append({"role": "tool", "content": result, "tool_name": name})
            step = Step("tool", name=name, args=args, result=result,
                        seconds=s["total_s"], prompt_tokens=s["prompt_tokens"])
            res.steps.append(step)
            if on_step:
                on_step(step)
    else:
        res.stopped = "step_limit"
        msg, s = call(with_tools=False, extra=[{"role": "user", "content": FINAL_NUDGE}])
        res.answer = (msg.get("content") or "").strip()
        step = Step("answer", result=res.answer, seconds=s["total_s"], prompt_tokens=s["prompt_tokens"])
        res.steps.append(step)
        if on_step:
            on_step(step)

    sources = [question] + [st.result for st in res.steps
                            if st.kind == "tool" and not st.result.startswith(("ERROR", "SKIPPED"))]
    res.unverified = unverified_numbers(res.answer, sources)
    if res.unverified:
        nums = ", ".join(f"{n:g}" for n in res.unverified)
        res.answer += (f"\n\n[Warning: {nums} did not come from any tool result or the question. "
                       "Treat as unverified.]")
    res.seconds = time.time() - t0
    return res


def print_step(step: Step):
    if step.kind == "plan":
        print(f"  [{step.seconds:4.1f}s, {step.prompt_tokens} tok]  PLAN: {step.result[:300]!r}")
    elif step.kind == "tool":
        r = step.result.replace("\n", " ")
        print(f"  [{step.seconds:4.1f}s, {step.prompt_tokens} tok]  -> {step.name}({step.args})")
        print(f"                     observed: {r[:160]!r}{'...' if len(r) > 160 else ''}")
    else:
        print(f"  [{step.seconds:4.1f}s, {step.prompt_tokens} tok]  ANSWER: {step.result[:400]!r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m app.agent "your question" [--plan] [--python]')
    if not llm.is_up():
        sys.exit("Ollama is not running. Start it with: ollama serve")
    from app.tools import with_python
    reg = with_python() if "--python" in sys.argv else None     # opt in: least privilege
    r = run(sys.argv[1], registry=reg, on_step=print_step, plan="--plan" in sys.argv)
    print(f"\n{r.stopped}: {r.model_calls} model call(s), {r.seconds:.1f}s, "
          f"largest prompt {r.max_prompt_tokens} tokens")
    if r.unverified:
        print(f"UNVERIFIED numbers in the answer: {r.unverified}")

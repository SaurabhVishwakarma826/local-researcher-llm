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

ONE_AT_A_TIME = ("SKIPPED: call one tool at a time. Wait for the result of the first call, "
                 "then decide the next call using that result.")

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set:
    out = set()
    for m in _NUMBER.findall(text):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def unverified_numbers(answer: str, sources: list) -> list:
    """Numbers in the answer that appear in no source (question or successful tool result).
    Integers below 10 are ignored (list markers, 'one tool', etc.)."""
    known = set().union(*(_numbers(s) for s in sources)) if sources else set()
    out = []
    for n in sorted(_numbers(answer)):
        if n < 10 and n == int(n):
            continue
        if not any(abs(n - k) <= 1e-6 * max(1.0, abs(k)) for k in known):
            out.append(n)
    return out

FINAL_NUDGE = ("You have used all available tool steps. Do not call any more tools. "
               "Answer the original question now using only the information above. "
               "If it is not enough, say so.")


@dataclass
class Step:
    kind: str               # "tool" | "answer"
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


def run(question: str, registry: dict = None, max_steps: int = None, on_step=None) -> AgentResult:
    registry = registry if registry is not None else REGISTRY
    max_steps = max_steps or config.AGENT_MAX_STEPS
    schemas = [t.schema for t in registry.values()]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    res = AgentResult(question=question, answer="", stopped="answered")
    seen = Counter()
    t0 = time.time()

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
    if step.kind == "tool":
        r = step.result.replace("\n", " ")
        print(f"  [{step.seconds:4.1f}s, {step.prompt_tokens} tok]  -> {step.name}({step.args})")
        print(f"                     observed: {r[:160]!r}{'...' if len(r) > 160 else ''}")
    else:
        print(f"  [{step.seconds:4.1f}s, {step.prompt_tokens} tok]  ANSWER: {step.result[:400]!r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m app.agent "your question"')
    if not llm.is_up():
        sys.exit("Ollama is not running. Start it with: ollama serve")
    r = run(sys.argv[1], on_step=print_step)
    print(f"\n{r.stopped}: {r.model_calls} model call(s), {r.seconds:.1f}s, "
          f"largest prompt {r.max_prompt_tokens} tokens")
    if r.unverified:
        print(f"UNVERIFIED numbers in the answer: {r.unverified}")

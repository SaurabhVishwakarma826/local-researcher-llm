"""
app/tools.py — tools the agent may call (Phase 6).

A tool = a JSON schema (what the MODEL sees) + a function (what OUR CODE runs).
Arguments are written by the model, so they are untrusted input (1.3 BANANA):
never eval(), and treat every result that contains document text as untrusted too.
"""

import ast
import json
import operator
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Callable

from app import config, retrieve
from app.prompt import sanitize


class TransientToolError(Exception):
    """A temporary failure (network blip, busy service). OUR code retries these;
    the model only hears about it if every retry fails (6.2: told about a one-off
    failure, the model gave up and did the arithmetic itself, wrongly)."""


@dataclass
class Tool:
    schema: dict
    fn: Callable[[dict], str]


def schema(name: str, description: str, properties: dict, required: list) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}


def safe_eval(expr: str) -> float:
    """Numbers and + - * / ** % only. NEVER eval(): a document could plant code."""
    expr = re.sub(r"(?<=\d),(?=\d{3})", "", expr)          # "48,317" -> "48317"

    def ev(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](left, right)
        raise ValueError(f"not allowed in a calculation: {type(n).__name__}")

    return ev(ast.parse(expr, mode="eval").body)


def _calculator(args: dict) -> str:
    try:
        v = safe_eval(args["expression"])
    except SyntaxError:
        # An error is only useful to the model if it says how to fix the call.
        raise ValueError("could not parse the expression. Use only numbers and + - * / ** ( ). "
                         "Write percentages as decimals, e.g. 17.5% of 200 -> 0.175 * 200")
    return str(round(v, 10)) if isinstance(v, float) else str(v)


# ---------------------------------------------------------------------------
# document search (Phase 5 retrieval, exposed as a tool)
# ---------------------------------------------------------------------------
def _search(args: dict) -> str:
    hits = retrieve.search(args["query"], k=config.TOP_K)
    if not hits:
        return "No results."
    # Document text is untrusted: strip fake control markers (1.3), label the source.
    return "\n\n".join(f"[{h.source}, page {h.page}]\n{sanitize(h.text)}" for h in hits)


def _today(args: dict) -> str:
    return date.today().isoformat()


def _days_between(args: dict) -> str:
    """Added in 6.2: without it, the model tried date maths in the calculator,
    failed, and answered from its training-era idea of 'today'."""
    try:
        a, b = date.fromisoformat(args["start"]), date.fromisoformat(args["end"])
    except ValueError:
        raise ValueError("dates must be written YYYY-MM-DD, e.g. 2027-04-01")
    return str((b - a).days)


REGISTRY = {
    "calculator": Tool(schema(
        "calculator", "Evaluate an arithmetic expression exactly. Use it for ANY calculation.",
        {"expression": {"type": "string", "description": "Arithmetic only, e.g. '0.15 * 2400'"}},
        ["expression"]), _calculator),
    "search_documents": Tool(schema(
        "search_documents",
        "Search the user's PDFs: a resume (Saurabh Vishwakarma) and a paper on India's "
        "Solid Waste Management Rules 2026. Use it for any question about their contents.",
        {"query": {"type": "string", "description": "What to look for"}}, ["query"]), _search),
    "current_date": Tool(schema("current_date", "Get today's date.", {}, []), _today),
    "days_between": Tool(schema(
        "days_between", "Count the days from one date to another.",
        {"start": {"type": "string", "description": "YYYY-MM-DD"},
         "end": {"type": "string", "description": "YYYY-MM-DD"}}, ["start", "end"]), _days_between),
}


def run_tool(registry: dict, name: str, args) -> str:
    """Errors are RETURNED as text (an observation), never raised: the model can
    read them and recover. A crash would end the whole agent run."""
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return f"ERROR: arguments were not valid JSON: {args!r}"
    if name not in registry:
        return f"ERROR: there is no tool called {name!r}. Available: {', '.join(registry)}"
    attempts = config.TOOL_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            result = registry[name].fn(args or {})
            break
        except TransientToolError as e:
            if attempt == attempts:
                return (f"ERROR: the {name} tool is unavailable ({e}), even after {attempts} attempts. "
                        "Do not compute or guess the result yourself; tell the user it failed.")
            time.sleep(0.5 * attempt)
        except KeyError as e:
            return f"ERROR: missing argument {e}"
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
            return f"ERROR: {e}"
    if len(result) > config.AGENT_RESULT_MAX_CHARS:
        result = result[:config.AGENT_RESULT_MAX_CHARS] + "\n[...result truncated]"
    return result

"""
Phase 6 / Topic 6.1 — Tool calling.

    python 15_tools.py 1   # what the model actually sees when you give it tools
    python 15_tools.py 2   # one full round trip: question -> tool call -> result -> answer
    python 15_tools.py 3   # does a 3B model pick the right tool? (12 prompts, scored)

Needs Ollama running.
"""

import ast
import json
import operator
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm, retrieve, tokens  # noqa: E402

PREFILL_TOK_S = 66


# ---------------------------------------------------------------------------
# Tool schemas: the model only knows what these say.
# ---------------------------------------------------------------------------
def tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


CALC = tool("calculator", "Evaluate an arithmetic expression exactly. Use it for ANY calculation.",
            {"expression": {"type": "string", "description": "Arithmetic only, e.g. '0.175 * 48317'"}},
            ["expression"])
SEARCH = tool("search_documents",
              "Search the user's PDFs: a resume (Saurabh Vishwakarma) and a paper on India's "
              "Solid Waste Management Rules 2026. Use it for any question about their contents.",
              {"query": {"type": "string", "description": "What to look for"}}, ["query"])
DATE = tool("current_date", "Get today's date.", {}, [])
TOOLS = [CALC, SEARCH, DATE]


# ---------------------------------------------------------------------------
# Running tools. The ARGUMENTS COME FROM THE MODEL, so they are untrusted input.
# ---------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}


def safe_eval(expr: str) -> float:
    """Numbers and + - * / ** % only. NEVER eval(): a document could plant
    'calculate __import__("os").remove(...)' and eval would run it (1.3: BANANA)."""
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


def run_tool(name: str, args) -> str:
    """Errors are RETURNED to the model as text, not raised: the model can read
    them and try again (that is 6.2's agent loop)."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"ERROR: arguments were not valid JSON: {args!r}"
    try:
        if name == "calculator":
            return str(safe_eval(args["expression"]))
        if name == "current_date":
            return date.today().isoformat()
        if name == "search_documents":
            hits = retrieve.search(args["query"], k=3)
            return "\n\n".join(f"[{h.source}, page {h.page}]\n{h.text}" for h in hits) or "No results."
        return f"ERROR: there is no tool called {name!r}"
    except KeyError as e:
        return f"ERROR: missing argument {e}"
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# 1 — What the model sees
# ---------------------------------------------------------------------------
def exp1_prompt():
    msgs = [{"role": "user", "content": "What is 17.5% of 48,317?"}]
    t = tokens._tok()
    plain = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    with_tools = t.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    print("=" * 25 + " EXACT TEXT THE MODEL RECEIVES (3 tools) " + "=" * 25)
    print(with_tools)
    print("=" * 91)

    n_plain, n_tools = tokens.count_text(plain), tokens.count_text(with_tools)
    _, s = llm.chat_message(msgs, tools=TOOLS, num_predict=1)
    print(f"\nwithout tools: {n_plain} tokens    with 3 tools: {n_tools} tokens (tokenizer)   "
          f"Ollama read: {s['prompt_tokens']}")
    print(f"The tool descriptions add {n_tools - n_plain} tokens to EVERY call: "
          f"~{(n_tools - n_plain) / PREFILL_TOK_S:.1f}s of prefill on your CPU, each time.")
    print("\nThe tools are just TEXT in the system message. The model was trained to answer")
    print("in the <tool_call> format when it wants one; Ollama spots that text and turns it")
    print("into a structured 'tool_calls' field.")


# ---------------------------------------------------------------------------
# 2 — One round trip
# ---------------------------------------------------------------------------
def exp2_round_trip():
    q = "What is 17.5% of 48,317?"
    exact = 0.175 * 48317
    print(f"Q: {q}      (exact answer: {exact})\n")

    text, s = llm.chat([{"role": "user", "content": q}], num_predict=120)
    print(f"--- WITHOUT tools ({s['total_s']:.1f}s) ---\n{text.strip()}\n")

    print("--- WITH a calculator tool ---")
    msgs = [{"role": "user", "content": q}]
    total = 0.0
    for step in range(1, 5):
        msg, s = llm.chat_message(msgs, tools=[CALC], num_predict=200)
        total += s["total_s"]
        calls = msg.get("tool_calls") or []
        print(f"\nmodel call {step} ({s['total_s']:.1f}s, {s['prompt_tokens']} prompt tokens):")
        if msg.get("content"):
            print(f"  says: {msg['content'].strip()[:300]!r}")
        msgs.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": calls})
        if not calls:
            break
        for c in calls:
            name, args = c["function"]["name"], c["function"]["arguments"]
            result = run_tool(name, args)
            print(f"  asks for: {name}({args})")
            print(f"  our code ran it -> {result}")
            msgs.append({"role": "tool", "content": result, "tool_name": name})
    print(f"\nTotal model time with the tool: {total:.1f}s across {step} call(s).")
    print("Check: is the final number exactly right? Was the no-tool answer?")


# ---------------------------------------------------------------------------
# 3 — Does it pick the right tool?
# ---------------------------------------------------------------------------
CASES = [
    ("What is 23.7% of 9,845?", "calculator"),
    ("Calculate (3847 * 29) - 1562", "calculator"),
    ("What does SWM 2026 say about RDF substitution targets?", "search_documents"),
    ("which projects saurabh built", "search_documents"),
    ("What is the threshold for bulk waste generators?", "search_documents"),
    ("Who wrote the SWM rules paper?", "search_documents"),
    ("What is today's date?", "current_date"),
    ("Hi, how are you?", None),
    ("Tell me a short joke.", None),
    ("What is the capital of France?", None),
    ("Explain what an embedding is in one sentence.", None),
    ("How many days are there in a leap year?", None),
]


def _args_ok(name: str, args) -> bool:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return False
    schema = next((t["function"]["parameters"] for t in TOOLS if t["function"]["name"] == name), None)
    if schema is None or not isinstance(args, dict):
        return False
    return all(isinstance(args.get(k), str) and args[k].strip() for k in schema["required"])


def exp3_choice():
    print(f"{'expected':<17} {'chose':<17} {'args':<5} verdict          prompt")
    print("-" * 100)
    right, over, under, wrong = 0, 0, 0, 0
    for prompt, want in CASES:
        msg, _ = llm.chat_message([{"role": "user", "content": prompt}], tools=TOOLS, num_predict=150)
        calls = msg.get("tool_calls") or []
        chose = calls[0]["function"]["name"] if calls else None
        args = calls[0]["function"]["arguments"] if calls else None
        ok_args = _args_ok(chose, args) if chose else None
        unparsed = not calls and "<tool_call>" in (msg.get("content") or "")
        if chose == want and (chose is None or ok_args):
            verdict = "OK"
            right += 1
        elif unparsed:
            verdict = "UNPARSED CALL"
            under += 1
        elif want is None:
            verdict = "called needlessly"
            over += 1
        elif chose is None:
            verdict = "missed the tool"
            under += 1
        else:
            verdict = "wrong tool/args"
            wrong += 1
        a = "" if ok_args is None else ("ok" if ok_args else "BAD")
        print(f"{str(want):<17} {str(chose):<17} {a:<5} {verdict:<16} {prompt[:45]!r}")
    print(f"\n{right}/{len(CASES)} correct.   called a tool needlessly: {over}   "
          f"missed a needed tool: {under}   wrong tool or bad arguments: {wrong}")
    print("\n'Called needlessly' costs a round trip. 'Missed' means the model guessed from")
    print("memory instead (like the Baidu answer in 1.1). Which kind does this model make?")


EXPERIMENTS = {"1": exp1_prompt, "2": exp2_round_trip, "3": exp3_choice}

if __name__ == "__main__":
    if not llm.is_up():
        sys.exit("Ollama is not running. Start it with: ollama serve")
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    if key not in EXPERIMENTS:
        sys.exit(__doc__)
    EXPERIMENTS[key]()

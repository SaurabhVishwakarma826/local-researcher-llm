"""
app/sandbox.py — run model-written Python with layered restrictions (Phase 6.3).

!! This is NOT a security boundary. Python is too flexible: AST filters like this one
!! have well-known bypasses. It stops ACCIDENTS and NAIVE injection attempts. Real
!! isolation = a container/VM with no network and no access to your files (Phase 11).
!! The real safeguard until then is layer 5: a human reading the code before it runs.

Layers:
  1. AST check BEFORE running: imports only from an allowlist; no open/eval/exec/getattr/...;
     no __dunder__ names, attributes or strings (the classic escape routes).
  2. Separate process, isolated mode (-I), stripped environment, empty temp folder as cwd.
  3. Timeout: the process is killed (infinite loops).
  4. Output cap (output floods). Output goes to a file, not memory.
  5. Human approval, when config.PYTHON_TOOL_CONFIRM is True and a person is at the keyboard.

Known gaps on Windows: no memory limit (needs job objects or a container); a very fast
output flood can still write a large temporary file before the timeout.
"""

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from app import config

ALLOWED_IMPORTS = {"math", "statistics", "datetime", "decimal", "fractions",
                   "itertools", "collections", "re", "json", "random"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "hasattr", "input", "breakpoint", "help",
                "exit", "quit", "memoryview", "dir", "type", "object", "super", "classmethod",
                "staticmethod", "property"}


def check_code(code: str):
    """Return None if the code passes layer 1, else the reason it was rejected."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error on line {e.lineno}: {e.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of '{a.name}' is not allowed (allowed: {', '.join(sorted(ALLOWED_IMPORTS))})"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not allowed"
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id.startswith("__"):
                return f"use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return f"access to private attribute '.{node.attr}' is not allowed"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:          # blocks '{0.__class__}'.format(x) style tricks
                return "strings containing '__' are not allowed"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) \
                    and node.right.value > 12:
                return "large powers are not allowed (memory/CPU safety)"
    return None


def _approved(code: str) -> bool:
    if not getattr(config, "PYTHON_TOOL_CONFIRM", True) or not sys.stdin.isatty():
        return True
    print("\n" + "-" * 20 + " the model wants to run this Python code " + "-" * 20)
    print(code)
    print("-" * 81)
    return input("Run it? [y/N] ").strip().lower() == "y"


def run_python(code: str) -> str:
    reason = check_code(code)
    if reason:
        return f"BLOCKED (layer 1, code check): {reason}"
    if not _approved(code):
        return "BLOCKED (layer 5): the user declined to run this code."

    limit = config.PYTHON_MAX_OUTPUT_CHARS
    with tempfile.TemporaryDirectory() as d:
        script, out_path = Path(d) / "snippet.py", Path(d) / "out.txt"
        script.write_text(code, encoding="utf-8")
        env = {"PYTHONIOENCODING": "utf-8", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        try:
            with open(out_path, "w", encoding="utf-8") as out:
                p = subprocess.run([sys.executable, "-I", str(script)], cwd=d, env=env,
                                   stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                   timeout=config.PYTHON_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return (f"ERROR (layer 3): stopped after {config.PYTHON_TIMEOUT_S}s. "
                    "Infinite loop, or too slow? Simplify the code.")
        text = out_path.read_text(encoding="utf-8", errors="replace")
    clipped = text[:limit] + (f"\n[...output cut at {limit} characters (layer 4)]" if len(text) > limit else "")
    if p.returncode != 0:
        last = "\n".join(clipped.strip().splitlines()[-3:])
        return f"ERROR: the code raised an exception:\n{last}"
    return clipped.strip() or "(nothing was printed: use print() to output the result)"

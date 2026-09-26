# app/prompt.py — prompt assembly helpers.

import re

# Any Qwen-style control marker: <|im_start|>, <|im_end|>, <|endoftext|>, etc.
# Measured in 06_templates.py exp 4: Ollama converts these strings into REAL
# special tokens even inside message content, so untrusted text must be cleaned.
_SPECIAL_MARKER = re.compile(r"<\|[^|<>\s]{1,40}\|>")


def sanitize(untrusted: str) -> str:
    """Remove control markers from text we did not write (PDFs, web pages, user uploads)."""
    return _SPECIAL_MARKER.sub("", untrusted)
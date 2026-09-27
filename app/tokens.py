"""
app/tokens.py — exact token counting with the model's own tokenizer.

Measured in 08_tokens.py exp 3: count_messages() matched Ollama's
prompt_eval_count exactly (29/29, 41/41, 96/96). Replaces the old
"characters / 3.5" guess, which was off by -69% on numbers and Hindi.
"""

from functools import lru_cache

from transformers import AutoTokenizer

from app import config


@lru_cache(maxsize=1)
def _tok():
    # Load from the local cache first (no network, no HF warning).
    # Only go online if the files have never been downloaded.
    try:
        return AutoTokenizer.from_pretrained(config.TOKENIZER_ID, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(config.TOKENIZER_ID)


def count_text(text: str) -> int:
    """Tokens in a plain piece of text (a chunk, a document, a question)."""
    return len(_tok().encode(text, add_special_tokens=False))


def count_messages(messages) -> int:
    """Exact prompt size Ollama will report for this message list:
    template markers, hidden default system message, and the open assistant turn."""
    text = _tok().apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return count_text(text)

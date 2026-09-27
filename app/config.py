# app/config.py — every tunable lives here. Measured values noted with source.

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b"

# Measured: 03_threads.py — 12 best prefill (66 tok/s), decode flat 10-12
NUM_THREAD = 12

# Measured: 04_numctx.py — 16384 prefills as fast as 4096 (within noise).
# Changing num_ctx between requests forces a ~3s model reload. Never vary it.
# KV cache at 16384: ~590 MB on top of 1.93 GB weights.
NUM_CTX = 16384

KEEP_ALIVE = "10m"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42

# Health check: normal prefill on this machine is ~60-70 tok/s.
# If logged prefill drops below ~40, restart the Ollama server first
# before debugging anything else (see the unexplained 19.6 tok/s run).
PREFILL_TOK_S_FLOOR = 40

# Sampling presets (1.2).
# Always set every cut-off explicitly. Ollama's defaults (top_k 40, top_p 0.9)
# silently limit what temperature can do, so leaving them implicit hides changes.
SAMPLING_DETERMINISTIC = {"temperature": 0.0, "seed": 42}          # RAG answers, eval, query rewriting
SAMPLING_VARIED = {"temperature": 0.7, "top_k": 1000,              # multi-query expansion only
                   "top_p": 1.0, "min_p": 0.1}


# Chat (1.4)
SYSTEM_PROMPT = ("You are a concise technical assistant. "
                 "If you are not sure of a fact, say so instead of guessing.")

# Exact token counting (2.1). Same tokenizer as the Ollama model; verified
# against prompt_eval_count 3/3. chars/3.5 was off by up to -69% on numbers.
TOKENIZER_ID = "Qwen/Qwen2.5-3B-Instruct"

# Why 4000 and not most of NUM_CTX: when old messages are dropped, the start of
# the prompt changes, so Ollama's prompt cache is lost and the WHOLE history is
# re-read. At ~66 tok/s, 4000 tokens = ~60 s for that one turn; 12000 = ~3 min.
CHAT_HISTORY_BUDGET = 4000

# Embeddings (2.2)
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
# BGE expects this in front of QUESTIONS only, never documents.
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Measured in 09_embeddings.py exp 5: text past this is silently ignored.
# embed.py refuses longer text instead of letting that happen.
EMBED_MAX_TOKENS = 512

DOCS_DIR = "data/pdfs"          # only this folder is ingested. Personal files go in data/private.
MIN_PAGE_CHARS = 50             # below this, a page is treated as having no text layer
TABLE_MAX_EMPTY_RATIO = 0.5     # page 1's fake "table" was mostly empty cells
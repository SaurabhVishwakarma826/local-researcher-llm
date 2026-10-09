# app/config.py — every tunable lives here. Measured values noted with source.

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"

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
# Health check: below this, suspect a degraded Ollama server. Set for the CURRENT model:
# 3B normally prefills ~66 tok/s, 7B ~25-32 tok/s. (7.1: 40 fired constantly on 7B.)
PREFILL_TOK_S_FLOOR = 15

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

# Chunking (3.2)
# Measured in 11_chunking.py: structure-300 scored 9/10, MRR 0.80, ~6 s top-3 prefill.
CHUNK_MAX_TOKENS = 300

# A "table" whose cells average more than this many characters is a page layout
# (two-column resume, author box), not data. Page 6's real table averaged ~60.
TABLE_MAX_AVG_CELL_CHARS = 200

# Vector store (3.3)
CHROMA_DIR = "data/chroma"      # inside data/, so it is already git-ignored
CHROMA_COLLECTION = "docs"

TOP_K = 3   # Phase 4: top 5 fixed 4, broke 4, +64% time, caused the first hallucination (c32)

# RAG answers (3.4)
RAG_MAX_PROMPT_TOKENS = 3000    # ~45 s of prefill at worst; chunks are dropped beyond this
RAG_ANSWER_MAX_TOKENS = 300     # ~25 s of decode at worst

# Prompt version (3.4). v1 = the original that scored answerable 3/4, refusals 4/4.
# Change only after 13_rag.py experiment 3 shows a version that beats it.
RAG_PROMPT_STYLE = "v3"   # Phase 4 (44 q): v3 0.61 correct, citations 0.90 exact, 0 invented
                          #                 v1 0.79 correct, citations 0.61 exact, 6 invented

# Retrieval mode (Phase 5). "dense" = embeddings only (the Phase 4 baseline).
# Change only after the harness shows a better hit@3.
RETRIEVAL_MODE = "hybrid"
HYBRID_POOL = 20    # how many results each method contributes before fusing
RRF_K = 60          # standard RRF constant: 1/(60 + rank)

# Reranking (5.2). None = off. Turn on only after the harness shows a better hit@3
# that is worth the extra time per question.
RERANKER = "minilm"   # 5.2: hit@3 0.88 -> 0.94, +0.3 s/question. bge: same hit@3, 2.5 s, hurt tables
RERANK_POOL = 10    # hybrid's hit@10 is 1.00: every answer is somewhere in its top 10
RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",   # ~90 MB, fast
    "bge": "BAAI/bge-reranker-base",                    # ~1.1 GB, slower
}

# Bump this whenever ingest.py or chunk.py changes, so the store re-indexes automatically.
# 1 = up to Phase 5.1; 2 = line-level table exclusion + captions as table headings (5.2)
INDEX_VERSION = 2

# Agent (6.2)
AGENT_MAX_STEPS = 5           # model calls with tools before a forced final answer
AGENT_MAX_REPEATS = 1         # identical tool call allowed once; again -> error observation
AGENT_MAX_TOKENS = 300        # answer length per model call
AGENT_RESULT_MAX_CHARS = 4000 # longer tool results are cut, so one search can't flood num_ctx

TOOL_RETRIES = 2   # 6.2: temporary tool failures are retried by our code before the model hears

# Workflow (6.2 option B). The router said "math only" for 5 document questions, so facts
# came from memory. Searching needlessly costs ~0.3 s; skipping a needed search does not.
WORKFLOW_ALWAYS_RETRIEVE = True

# Python tool (6.3). NOT a security boundary: stops accidents and naive injection only.
PYTHON_TIMEOUT_S = 5            # layer 3: kill runaway code
PYTHON_MAX_OUTPUT_CHARS = 2000  # layer 4: cap output
PYTHON_TOOL_CONFIRM = True      # layer 5: show the code and ask y/N before running (interactive only)

# Short-term memory (7.1)
MEMORY_BUDGET_TOKENS = 4000      # system + history + new question (same as CHAT_HISTORY_BUDGET)
MEMORY_WINDOW_TURNS = 4          # exchanges kept word for word by window / summary
MEMORY_SUMMARY_MAX_WORDS = 120   # running summary length cap
MEMORY_MAX_FACTS = 30   # fact memory cap; oldest facts drop first beyond this
MEMORY_STRATEGY = "facts"   # 7.1: only strategy to recall turns 2 AND 10 of 20 within budget;
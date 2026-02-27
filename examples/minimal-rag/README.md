# minimal-rag

A complete RAG app in 6 files. Clone it, run it, break it, fix it.

## Quick Start

```bash
cd examples/minimal-rag
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# 1. index the sample docs
python ingest.py

# 2. run the app
uvicorn app:app --reload

# 3. open http://localhost:8000 and ask a question
```

## What's in Here

```
app.py              FastAPI server with /chat and /chat/stream endpoints
                    Includes: safety filter, cost logging, similarity threshold,
                    model routing, streaming SSE, built-in web UI

ingest.py           Document ingestion → chunking → FAISS vector store
                    Run once, or re-run when you add new documents

safety.py           Prompt injection detection (13 patterns)
                    PII detection and masking (5 patterns)

eval.py             RAG quality evaluation with RAGAS
                    Runs your golden test set, checks faithfulness

test_rag.py         5 minimum viable tests (pytest)
                    Retrieval, safety, PII masking — runs in seconds

golden_qa.json      8 question/answer pairs for evaluation
sample_docs/        Sample FAQ document for testing
```

## Run the Tests

```bash
# safety + retrieval tests (free, seconds)
pytest test_rag.py -v

# RAG quality evaluation (costs ~$0.05, takes ~30s)
python eval.py --quick     # first 3 questions only
python eval.py             # all 8 questions
```

## Use Your Own Documents

```bash
# put your .txt files in a folder
python ingest.py /path/to/your/docs

# update golden_qa.json with questions about YOUR docs
python eval.py
```

## What This Demonstrates

Every pattern from the guide docs, working together:

| Guide Section | Implementation |
|--------------|----------------|
| Hallucination detection | `eval.py` — RAGAS faithfulness check |
| Safety filter | `safety.py` — `check_input()` + `mask_pii_in_output()` |
| Cost tracking | `app.py` — `log_cost()` prints per-request cost |
| Retrieval debugging | `app.py` — `rag_logger` logs every retrieved chunk with scores |
| Similarity threshold | `app.py` — `MIN_SIMILARITY_SCORE = 0.3` filters garbage chunks |
| Chunking | `ingest.py` — 800 chars, 200 overlap, recursive splitting |
| Streaming | `app.py` — `/chat/stream` endpoint with SSE |
| Error handling | `app.py` — safety check before LLM call |
| Model routing | `app.py` — `pick_model()` routes by query complexity |
| Testing | `test_rag.py` — the 5 minimum tests from the guide |

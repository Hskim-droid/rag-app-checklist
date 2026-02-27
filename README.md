# RAG App Checklist

Production-ready patterns for RAG applications. Two guides with copy-paste code.

## [RAG App Production Guide](docs/cross-cutting-concerns.md)

Common production issues and their solutions:

- **Hallucination detection** — automated grounding check with RAGAS
- **Input safety** — prompt injection and PII protection
- **Cost monitoring** — per-request cost logging and model routing
- **Retrieval debugging** — logging, similarity thresholds, diagnostics
- **Chunking** — recommended settings and quality verification
- **Streaming** — SSE with FastAPI and vanilla JS
- **Error handling** — retry logic with rate limit handling

## [RAG App Testing Guide](docs/testing-strategy.md)

Testing strategies for non-deterministic LLM outputs:

- **Retrieval testing** — verify correct context is found (deterministic, free)
- **Generation testing** — grounding and hallucination detection (RAGAS)
- **Safety testing** — prompt injection and PII, parametrized
- **Agent testing** — tool selection accuracy, step efficiency, hallucination rate
- **LLM mocking** — fast, cost-free testing

## Try It

```bash
cd examples/minimal-rag
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python ingest.py
uvicorn app:app --reload
# open http://localhost:8000
```

A complete RAG app in 6 files — every pattern from the guides, actually running. See [`examples/minimal-rag/`](examples/minimal-rag/) for details.

## Origin

Extracted from a production AI platform running RAG + Knowledge Graph + agent tool calling. Patterns stripped of proprietary details and rewritten as practical recipes.

## License

MIT

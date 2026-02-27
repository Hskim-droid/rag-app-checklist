# RAG App Survival Guide

You built a RAG app. It works on your laptop. Before you share that link, read these two docs.

## [Ship Your RAG App Without Embarrassing Yourself](docs/cross-cutting-concerns.md)

Copy-paste solutions for the things that will break in production:

- **Hallucination detection** — 10 lines to check if your RAG is making stuff up
- **Safety filter** — block prompt injection and PII leaks (one function, drop it in)
- **Cost tracking** — find out you're spending $0.08 per chat before the bill arrives
- **Retrieval debugging** — why your chunks are returning garbage and how to fix it
- **Chunking** — the settings that actually work (and how to verify)
- **Streaming** — stop making users stare at a spinner for 5 seconds
- **Error handling** — your LLM API will fail, handle it

## [How to Test a RAG App](docs/testing-strategy.md)

You can't `assert answer == "exact string"` with an LLM. Here's what you do instead:

- **Test retrieval** — is the right context being found? (deterministic, free)
- **Test generation** — is the answer grounded in context, or hallucinated? (RAGAS)
- **Test safety** — prompt injection and PII, parametrized
- **Test agents** — tool selection accuracy, step efficiency, hallucination rate
- **Mock the LLM** — so your tests don't cost money

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

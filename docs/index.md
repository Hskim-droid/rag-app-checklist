# RAG App Checklist

Two guides for taking a RAG app from local prototype to production-ready.

---

## [RAG App Production Guide](cross-cutting-concerns.md)

Copy-paste solutions for common production issues:

- **Hallucination detection** — automated check with RAGAS in 10 lines
- **Input safety** — prompt injection and PII protection in one function
- **Cost monitoring** — per-request cost logging and model routing
- **Retrieval debugging** — logging, similarity thresholds, and diagnostics
- **Chunking** — recommended settings and quality verification
- **Streaming** — SSE implementation with FastAPI and vanilla JS
- **Error handling** — retry logic with rate limit and context overflow handling

## [RAG App Testing Guide](testing-strategy.md)

LLM outputs are non-deterministic. Standard assertions don't apply. Here's what works:

- **Retrieval testing** — verify correct context is found (deterministic, free)
- **Generation testing** — check grounding and detect hallucination (RAGAS)
- **Safety testing** — prompt injection and PII, parametrized
- **Agent testing** — tool selection accuracy, step efficiency, hallucination rate
- **LLM mocking** — keep tests fast and cost-free

---

## Quick Start

| Goal | Recommended Reading |
|------|---------------------|
| Deploy a RAG app this week | [Production Guide](cross-cutting-concerns.md) — start with the checklist at the bottom |
| Reduce hallucination | [Production Guide, Section 1](cross-cutting-concerns.md#1-hallucination-detection) |
| Add input safety | [Production Guide, Section 2](cross-cutting-concerns.md#2-input-safety-prompt-injection-and-pii) |
| Write tests for a RAG app | [Testing Guide](testing-strategy.md) — start with the 5 essential tests at the bottom |
| Test agent tool calls | [Testing Guide, Section 5](testing-strategy.md#5-agent-tool-call-testing-if-applicable) |

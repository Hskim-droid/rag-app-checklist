# RAG App Checklist

You built a RAG app. It works on your laptop. Before you share that link, go through these two guides.

---

## [Ship Your RAG App Without Embarrassing Yourself](cross-cutting-concerns.md)

Copy-paste solutions for the things that will break in production:

- **Hallucination detection** — 10 lines to check if your RAG is making stuff up
- **Safety filter** — block prompt injection and PII leaks (one function, drop it in)
- **Cost tracking** — find out you're spending $0.08 per chat before the bill arrives
- **Retrieval debugging** — why your chunks are returning garbage and how to fix it
- **Chunking** — the settings that actually work (and how to verify)
- **Streaming** — stop making users stare at a spinner for 5 seconds
- **Error handling** — your LLM API will fail, handle it

## [How to Test a RAG App](testing-strategy.md)

You can't `assert answer == "exact string"` with an LLM. Here's what you do instead:

- **Test retrieval** — is the right context being found? (deterministic, free)
- **Test generation** — is the answer grounded in context, or hallucinated? (RAGAS)
- **Test safety** — prompt injection and PII, parametrized
- **Test agents** — tool selection accuracy, step efficiency, hallucination rate
- **Mock the LLM** — so your tests don't cost money

---

## Quick Start

Pick your situation:

| I need to... | Read this |
|-------------|-----------|
| Ship a RAG app this week | [Ship Your RAG App](cross-cutting-concerns.md) — start with the checklist at the bottom |
| Stop my RAG from hallucinating | [Ship Your RAG App, Section 1](cross-cutting-concerns.md#1-your-rag-is-hallucinating-and-you-dont-know-it) |
| Add safety before someone jailbreaks it | [Ship Your RAG App, Section 2](cross-cutting-concerns.md#2-someone-will-try-to-jailbreak-your-app-on-day-one) |
| Write tests for a RAG app | [How to Test a RAG App](testing-strategy.md) — start with the 5 minimum tests at the bottom |
| Test my agent's tool calls | [Testing Strategy, Section 5](testing-strategy.md#5-test-your-agents-tool-calls-if-you-have-agents) |

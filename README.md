# LLM Platform Engineering Reference

Practical architectural guides for building production LLM applications — extracted from a real system running on-premises with local models + cloud fallback.

## Documents

### [Cross-Cutting Concerns](docs/cross-cutting-concerns.md)

How to handle the infrastructure that every LLM platform needs but nobody documents well:

- **Observability** — OpenTelemetry + Logfire for tracing LLM calls, agent steps, and tool invocations
- **RAG Evaluation** — RAGAS pipeline with daily batch evaluation and alerting
- **E2E Type Safety** — OpenAPI codegen to eliminate manual FE-BE synchronization
- **CI/CD** — Three-pipeline structure: correctness (PR), delivery (deploy), quality monitoring (daily eval)
- **Safety Pipeline** — 6-stage input + 3-stage output filtering
- **Streaming** — Why you should standardize on one SSE protocol

### [Testing Strategy](docs/testing-strategy.md)

How to test applications where the core logic involves non-deterministic LLM calls:

- **Test Pyramid for AI Apps** — Classic pyramid + RAG evaluation + agent behavioral testing
- **Hexagonal Architecture for Testability** — Port-based DI makes LLM mocking trivial
- **RAG Evaluation (RAGAS)** — Golden datasets, pass/fail thresholds, daily batch execution
- **Agent Evaluation** — Four metrics: Tool Selection Accuracy, Task Completion Rate, Step Efficiency, Hallucination Rate
- **Integration Tests** — Real databases, mocked LLMs

## Context

These documents were generalized from the architecture of a production AI platform that includes:

- Multi-model LLM gateway (local vLLM + cloud API fallback)
- Knowledge Graph with hybrid search (vector + graph + reranking)
- RAG-powered chat with adaptive model routing
- Agent framework with tool calling
- Real-time collaborative document editing

The patterns described are technology-agnostic where possible, with concrete examples using FastAPI, PydanticAI, LangGraph, and pytest.

## License

MIT

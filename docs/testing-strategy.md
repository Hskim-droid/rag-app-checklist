# Testing Strategy for LLM & Agent Applications

> A practical reference for building a test suite around LLM-powered applications — from unit tests through RAG evaluation to agent behavioral testing.

---

## 1. The Test Pyramid for AI Applications

The classic test pyramid still applies, but AI applications add two new layers: **RAG evaluation** and **agent behavioral testing**.

```
           ╱╲
          ╱  ╲           E2E Tests (Playwright)
         ╱ 5% ╲          - Critical user flows only
        ╱──────╲          - Login → Chat → RAG → File Upload
       ╱        ╲
      ╱   15%    ╲       Integration Tests (TestClient + Docker)
     ╱────────────╲      - API endpoints end-to-end
    ╱              ╲     - Real DB, mocked LLM
   ╱                ╲
  ╱      80%         ╲   Unit Tests (pytest)
 ╱────────────────────╲  - Domain services, utilities
                          - Port mocking for isolation

  +  RAG Evaluation (RAGAS)         — daily batch, not per-test
  +  Agent Evaluation (custom)      — tool selection, task completion
```

---

## 2. Unit Tests (pytest)

### Directory Structure

```
tests/unit/
├── domain/
│   ├── test_chat_service.py          # Chat business logic
│   ├── test_retrieval_service.py     # RAG search and ingestion
│   ├── test_news_service.py          # Content generation logic
│   ├── test_auth_service.py          # JWT issuance and verification
│   └── test_admin_service.py         # Permission management
├── infra/
│   ├── test_llm_gateway.py           # LLM calls (mocked)
│   ├── test_semantic_router.py       # Model routing decisions
│   └── test_structured_output.py     # Pydantic output validation
└── middleware/
    ├── test_rate_limiter.py          # Rate limiting logic
    ├── test_content_filter.py        # Content filter patterns
    └── test_safety_filter.py         # Injection detection, PII masking
```

### Why Hexagonal Architecture Matters for Testing

The single biggest testing win for LLM applications is **port-based dependency injection**. When your domain services depend on abstract interfaces (ports) rather than concrete implementations, you can swap in mocks trivially:

```python
# Define a mock that replaces the real LLM gateway
class MockLLMPort(LLMPort):
    async def chat(self, messages, model, options=None):
        return "mocked response"

class MockSearchPort(SearchPort):
    async def search(self, query, top_k=5):
        return [Document(content="test doc", score=0.95)]

# Test domain logic in complete isolation — no LLM, no DB, no network
async def test_chat_service_processes_query():
    service = ChatService(
        llm=MockLLMPort(),
        search=MockSearchPort(),
        # ... other mocked ports
    )
    result = await service.process_query("test question")
    assert result.success is True
    assert len(result.sources) > 0
```

Without hexagonal architecture, testing an LLM application means either:
- Calling real LLMs (slow, expensive, non-deterministic)
- Monkeypatching internals (fragile, couples tests to implementation)

With ports, you test business logic deterministically in milliseconds.

---

## 3. API Integration Tests (TestClient)

### Directory Structure

```
tests/integration/
├── test_chat_api.py
├── test_retrieval_api.py
├── test_news_api.py
├── test_documents_api.py
├── test_agent_api.py
├── test_admin_api.py
└── test_auth_api.py
```

### Example: Testing a Streaming Chat Endpoint

```python
from httpx import AsyncClient

@pytest.fixture
async def auth_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        token = create_test_jwt(user_id="test-user", roles=["admin"])
        client.cookies.set("access_token", token)
        yield client

async def test_chat_stream_returns_chunks(auth_client):
    async with auth_client.stream("POST", "/v1/chat/stream", json={
        "query": "Hello, how are you?",
        "model_preference": "fast"
    }) as response:
        assert response.status_code == 200
        chunks = []
        async for chunk in response.aiter_text():
            chunks.append(chunk)
        assert len(chunks) > 0
```

### Test Infrastructure

Use a dedicated Docker Compose for integration tests:

```yaml
# docker-compose.test.yml
services:
  postgres-test:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: test_db
    tmpfs: /var/lib/postgresql/data    # RAM disk for speed

  neo4j-test:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: none                 # No auth in test

  redis-test:
    image: redis:7-alpine

  llm-mock:
    image: your-llm-mock:latest        # Or use LiteLLM in mock mode
```

Key principle: **real databases, mocked LLM**. Database behavior is hard to mock correctly (transactions, constraints, indexes). LLM behavior is hard to test with real models (non-deterministic, slow, expensive). Mock the expensive non-deterministic part, keep the cheap deterministic part real.

---

## 4. RAG Evaluation (RAGAS)

### Directory Structure

```
tests/eval/
├── test_rag_quality.py
├── datasets/
│   ├── domain_a_qa.json              # Golden Q&A set (50+ pairs)
│   └── domain_b_qa.json              # Golden Q&A set (30+ pairs)
└── conftest.py                        # RAGAS configuration
```

### Golden Dataset Format

```json
{
  "question": "What are the active ingredients in Product X?",
  "ground_truth": "The active ingredients are Compound A and Compound B.",
  "contexts": ["Product X contains Compound A (0.5%) and Compound B (0.1%)..."]
}
```

### Pass/Fail Thresholds

| Metric | Minimum Threshold |
|--------|-------------------|
| Faithfulness | ≥ 0.8 |
| Answer Relevancy | ≥ 0.7 |
| Context Precision | ≥ 0.6 |
| Context Recall | ≥ 0.6 |

### Execution Strategy

```
Schedule:     Daily at 06:00 via CI (not on every PR — too slow and expensive)
Manual:       pytest tests/eval/ -v
On failure:   Alert via Slack/email, block deployment pipeline
```

Why not per-PR? A full RAGAS evaluation requires LLM calls to judge faithfulness and relevancy. Running it on every PR would be prohibitively expensive. Daily batch evaluation catches regressions within 24 hours, which is fast enough for most teams.

---

## 5. E2E Tests (Playwright)

### Directory Structure

```
tests/e2e/
├── auth.spec.ts                      # Login/logout flows
├── chat.spec.ts                      # Chat → streaming → RAG sources
├── retrieval.spec.ts                 # Knowledge search → answer
├── documents.spec.ts                 # Upload → edit → share
├── agent.spec.ts                     # Agent execution → tool calls
└── admin.spec.ts                     # Admin panel functions
```

### Critical Scenarios (Keep These Minimal)

E2E tests are expensive to maintain. Only test flows that, if broken, would be immediately visible to users:

1. **Login → Chat → RAG verification** — User logs in, asks a question, gets an answer with source citations, sources are clickable
2. **File upload → search** — User uploads a document, searches for its content, finds it in results
3. **Agent execution** — User triggers an agent task, sees progress, gets results with tool call evidence

---

## 6. Agent Evaluation Framework

This is the newest and least-established testing layer. Standard test frameworks don't cover agent-specific failure modes.

### Directory Structure

```
tests/eval/agents/
├── test_chat_agent.py
├── test_retrieval_agent.py
├── test_news_agent.py
└── datasets/
    ├── agent_scenarios.json           # Input scenarios
    └── expected_tool_calls.json       # Expected tool sequences
```

### The Four Agent Metrics

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Tool Selection Accuracy** | % of cases where the agent chose the correct tool | Wrong tool = wrong answer, even if reasoning is correct |
| **Task Completion Rate** | % of tasks the agent completed successfully | Measures end-to-end reliability |
| **Step Efficiency** | Actual steps / minimum possible steps | Detects unnecessary loops and redundant tool calls |
| **Hallucination Rate** | % of answers generated without retrieval evidence | The most dangerous failure mode in RAG agents |

### Example: Testing an Agent's Tool Selection

```python
async def test_retrieval_agent_selects_correct_tools():
    agent = RetrievalAgent(deps=test_deps)
    result = await agent.run("What is the price of Product X?")

    # 1. Verify answer quality
    assert result.data.confidence == "high"
    assert "price" in result.data.answer.lower() or "$" in result.data.answer

    # 2. Verify source attribution
    assert any(s.type == "kg_direct" for s in result.data.sources)

    # 3. Verify tool selection — the agent should have searched the KG
    tool_calls = [m for m in result.all_messages() if m.kind == "tool-call"]
    assert tool_calls[0].tool_name == "kg_search"

    # 4. Verify step efficiency — should not need more than 3 steps
    assert len(tool_calls) <= 3
```

### What Makes Agent Testing Different

Traditional tests verify **outputs**. Agent tests must also verify **process**:

- Did the agent use the right tools in the right order?
- Did it stop when it had enough information, or did it loop unnecessarily?
- When retrieval returned no results, did it say "I don't know" or hallucinate?
- When given a dangerous tool (file deletion, API call), did it request human approval?

These behavioral properties can't be tested with simple input/output assertions. You need to inspect the agent's message history, tool call sequence, and reasoning trace.

---

## Summary

| Layer | What | When | Cost |
|-------|------|------|------|
| Unit Tests | Domain logic with mocked ports | Every PR | Seconds |
| Integration Tests | API endpoints with real DBs, mocked LLM | Every PR | Minutes |
| E2E Tests | Critical user flows in browser | Every PR | Minutes |
| RAG Evaluation | Retrieval + generation quality (RAGAS) | Daily batch | ~$5-20/run |
| Agent Evaluation | Tool selection, completion, hallucination | Daily batch | ~$10-30/run |

### Key Principles

1. **Hexagonal architecture is a testing prerequisite** — Without port-based DI, you can't isolate domain logic from LLM calls
2. **Mock the LLM, keep the DB real** — Databases are deterministic and cheap; LLMs are neither
3. **RAG evaluation is not optional** — If you ship RAG without measuring faithfulness, you're shipping blind
4. **Agent tests verify process, not just output** — Tool selection accuracy and hallucination rate matter as much as answer correctness
5. **Daily batch for expensive evals** — Don't run RAGAS on every PR; daily is fast enough to catch regressions

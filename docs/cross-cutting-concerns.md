# Cross-Cutting Concerns for LLM Platforms

> A practical reference for observability, RAG evaluation, type safety, CI/CD, safety filtering, and streaming standardization in production LLM applications.

---

## 1. Observability (OpenTelemetry + Logfire)

LLM applications have unique observability needs beyond traditional web services: token usage tracking, per-model cost attribution, RAG retrieval quality, and agent step tracing.

### Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Metrics | OpenTelemetry Metrics | Request latency, token counts, costs |
| Tracing | OpenTelemetry Distributed Tracing | End-to-end request flow |
| LLM Tracking | Pydantic Logfire | LLM call details, agent execution |
| Dashboard | Grafana | Visualization and alerting |

### Auto-Instrumentation

```
FastAPI           ──OTel SDK──→  [OTel Collector]  ──→  [Prometheus]  ──→  [Grafana]
PydanticAI Agent  ──Logfire───→  [Logfire Cloud / Self-hosted]
```

Instrument these layers automatically:
- **HTTP requests/responses** — FastAPI Instrumentor
- **Database queries** — SQLAlchemy, Neo4j, Redis Instrumentors
- **LLM calls** — Logfire PydanticAI integration
- **Agent execution** — Logfire LangGraph integration

### Custom Metrics

Define domain-specific metrics that matter for LLM operations:

```
app.chat.latency_ms              // End-to-end chat response latency
app.chat.tokens_used             // Token consumption per request
app.rag.search_latency_ms        // RAG retrieval latency
app.rag.faithfulness_score       // RAG faithfulness (from RAGAS)
app.agent.steps_count            // Agent reasoning steps
app.agent.tool_calls_count       // Tool invocations per agent run
app.llm.cost_usd                 // Per-request LLM cost
app.pipeline.cycle_duration_ms   // Background pipeline cycle time
```

### Trace Anatomy — What a Single Chat Request Looks Like

```
[Chat Request]                                  total: 4.15s
  ├── [Preflight Check]           (1.2ms)       rate limit + content filter + prompt limits
  ├── [System Prompt Build]       (0.5ms)       role instructions + context injection
  ├── [Agent Run]                 (2,340ms)     core reasoning loop
  │   ├── [LLM Call #1]           (model=fast, tokens=1200, cost=$0.001)
  │   ├── [Tool: kg_search]       (45ms)        knowledge graph lookup
  │   ├── [Tool: vector_search]   (120ms)       embedding similarity search
  │   └── [LLM Call #2]           (model=fast, tokens=2400, cost=$0.002)
  ├── [Stream Response]           (1,800ms)     SSE token delivery to client
  └── [Background Tasks]
      ├── [KG Update]             (300ms)       extract entities from conversation
      └── [Metrics Record]        (5ms)         log usage data
```

This trace structure lets you answer questions like:
- Which tool call is the bottleneck?
- What percentage of latency is LLM inference vs. retrieval?
- How much does each request cost?

---

## 2. RAG Evaluation (RAGAS Pipeline)

Most teams ship RAG without measuring quality. Integrate [RAGAS](https://docs.ragas.io/) into your pipeline to catch retrieval and generation regressions automatically.

### Metrics

| Metric | What It Measures | Range |
|--------|-----------------|-------|
| **Faithfulness** | Does the answer stay grounded in retrieved context? | 0–1 |
| **Answer Relevancy** | Is the answer actually relevant to the question? | 0–1 |
| **Context Precision** | How much of the retrieved context is useful? | 0–1 |
| **Context Recall** | Was all necessary information retrieved? | 0–1 |

### Pipeline Architecture

```
[Request Logging]
  Log: query, retrieved_contexts, generated_answer, ground_truth (if available)
      │
      ▼
[RAGAS Evaluate]  (async batch — run daily, not per-request)
      │
      ├─ Faithfulness:  LLM verifies each answer sentence against context
      ├─ Relevancy:     LLM generates reverse-questions from answer → compare to original
      ├─ Precision:     Ratio of useful context chunks to total retrieved
      └─ Recall:        Ratio of needed information that was actually retrieved
      │
      ▼
[Dashboard]
  Daily averages, per-domain comparison, trend graphs
```

### Alerting Rules

Set thresholds and alert when quality degrades:

```
Faithfulness      < 0.7  →  Alert: "RAG faithfulness degradation detected"
Context Precision < 0.5  →  Alert: "Retrieval precision degradation detected"
```

Why daily batch instead of per-request? RAGAS evaluation itself requires LLM calls — running it on every request would double your inference costs.

---

## 3. End-to-End Type Safety (OpenAPI Codegen)

Manual FE-BE type synchronization is a losing battle. A hand-maintained API client grows into an unmaintainable monolith (we've seen 1,400+ line single-file clients). Automate it.

### Pipeline

```
[FastAPI Backend]
      │
      ▼  auto-generated
[OpenAPI 3.1 Schema]  (/openapi.json)
      │
      ▼  openapi-typescript-codegen
[TypeScript Types + API Client]  (auto-generated, zero manual code)
      │
      ▼
[Frontend Code]  (type-safe API calls, no manual conversion)
```

### CI Integration

```
1. Backend PR merged      →  OpenAPI schema generated
2. Schema change detected →  Frontend types auto-regenerated
3. Type mismatch          →  CI fails (frontend build error)
```

### What You Eliminate

- Manual `snake_case` ↔ `camelCase` conversion code
- Hand-written API client functions (1,400+ lines → 0 lines maintained)
- Runtime FE-BE schema mismatches (caught at compile time instead)

---

## 4. CI/CD Pipeline (GitHub Actions)

### Pull Request Pipeline (`ci.yml`)

```yaml
jobs:
  lint:
    - ruff check (Python)
    - eslint (TypeScript)

  type-check:
    - mypy (Python)
    - tsc --noEmit (TypeScript)

  test-unit:
    - pytest tests/unit/ --cov=domain/ --cov-fail-under=80

  test-integration:
    - docker compose -f docker-compose.test.yml up -d
    - pytest tests/integration/

  test-e2e:
    - playwright test

  openapi-check:
    - Detect OpenAPI schema changes → regenerate FE types → verify build
```

### Deployment Pipeline (`deploy.yml` — on main merge)

```yaml
jobs:
  build:
    - docker build (backend, frontend)
    - docker push (registry)

  deploy:
    - docker compose pull
    - docker compose up -d --remove-orphans
    - health check
```

### Daily RAG Evaluation (`rag-eval.yml` — 06:00 daily)

```yaml
jobs:
  evaluate:
    - Run RAGAS evaluation pipeline
    - Push results to Grafana
    - Alert on threshold violations
```

This three-pipeline structure separates concerns: correctness (PR), delivery (deploy), and quality monitoring (daily eval).

---

## 5. Safety Pipeline

LLM applications need both input and output safety filtering. This is not optional — it's infrastructure.

### Input Safety (6 Stages)

```
[User Input]
  → 1. Unicode Normalization     (NFKC — prevent homoglyph attacks)
  → 2. Length Validation          (reject oversized inputs early)
  → 3. Prompt Injection Detection (regex patterns for known attack vectors)
  → 4. PII Masking               (phone numbers, emails, SSNs, etc.)
  → 5. Keyword Blocklist         (domain-specific forbidden terms)
  → 6. Sensitive Pattern Filter   (API keys, secrets, credentials)
  → [Clean Input to LLM]
```

### Output Safety (3 Stages)

```
[LLM Output]
  → 1. Data Leak Prevention      (detect internal data in responses)
  → 2. PII Masking               (re-mask any PII the model generates)
  → 3. Output Inspection         (content policy compliance)
  → [Safe Output to User]
```

The key insight: safety filtering must happen on **both sides** of the LLM. Input filtering prevents attacks; output filtering prevents leaks.

---

## 6. Middleware Pipeline

Every LLM request should pass through a preflight pipeline before reaching the model:

```
[Authenticated Request]
  ├─ 1. Rate Limiter      →  per-minute / per-hour / daily-token limits
  ├─ 2. Content Filter    →  attack pattern detection (injection, abuse, etc.)
  ├─ 3. Prompt Limits     →  max token clamping, prompt length validation
  └─ [Main Processing]
```

Implementation note: if your rate limiter depends on Redis and Redis is down, **fail open** with generous defaults (e.g., `remaining=999`). Don't let infrastructure failures block all users.

---

## 7. Streaming Infrastructure

If your platform has multiple streaming endpoints (chat, search, retrieval), **unify the protocol**. Protocol fragmentation is a common and painful anti-pattern.

### The Problem

Many platforms end up with inconsistent SSE implementations:

| Endpoint | Done Signal | Error Format | Event Format |
|----------|-------------|--------------|-------------|
| Chat | `data: [DONE]` | `{"type":"error"}` | `data: {type, ...}` |
| Search | `{"type":"done"}` | `{"type":"error"}` | `data: {type, ...}` |
| KG Query | Connection close | `event: error` | `event: X` + `data: Y` |

Three endpoints, three protocols. Every frontend developer has to learn three parsing strategies.

### The Solution — Standardize on One Protocol

Use an established streaming protocol (e.g., Vercel AI SDK Data Stream Protocol) across all endpoints:

**Backend:**
```python
async def stream_chat():
    async for chunk in agent.run_stream(...):
        yield format_standard_chunk(chunk)
```

**Frontend:**
```typescript
const { messages, ... } = useChat({ api: "/api/chat" })
// Zero manual SSE parsing code
```

### What You Eliminate

- Custom SSE parsing code per endpoint
- Inconsistent error handling across streams
- Frontend developers needing to know protocol differences
- Buffer management and reconnection logic duplication

---

## Summary

| Concern | Key Decision | Impact |
|---------|-------------|--------|
| Observability | OpenTelemetry + Logfire | See inside every LLM call and agent step |
| RAG Quality | RAGAS daily batch eval | Catch retrieval regressions before users do |
| Type Safety | OpenAPI codegen | Zero manual FE-BE sync, compile-time error detection |
| CI/CD | Three-pipeline structure | Separate correctness, delivery, and quality monitoring |
| Safety | 6-stage input + 3-stage output | Defense in depth for both attacks and leaks |
| Middleware | Rate limit → Content filter → Prompt limits | Protect your LLM from abuse |
| Streaming | Single unified protocol | One protocol, one parser, all endpoints |

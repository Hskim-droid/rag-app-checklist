"""
Minimal RAG app with FastAPI.
Includes: retrieval, streaming, safety filter, cost logging, similarity threshold.

Usage:
    python ingest.py          # index sample docs first
    uvicorn app:app --reload  # start the server
"""

import json
import logging
import os
import sys
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI

from safety import check_input, mask_pii_in_output

# --- config ---

VECTOR_STORE_PATH = "vector_store"
DEFAULT_MODEL = "gpt-4o-mini"
MIN_SIMILARITY_SCORE = 0.3
TOP_K = 5

PRICING = {
    "gpt-4o":       {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
}

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- ONLY use information from the context below to answer
- If the context does not contain the answer, say "I don't have enough information to answer this question."
- Do NOT make up information that is not in the context
- Cite specific details from the context when possible

Context:
{context}"""

# --- setup ---

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
cost_logger = logging.getLogger("cost")
rag_logger = logging.getLogger("rag")

app = FastAPI(title="Minimal RAG")
client = AsyncOpenAI()

if not os.path.exists(VECTOR_STORE_PATH):
    print("ERROR: Vector store not found. Run 'python ingest.py' first.")
    sys.exit(1)

vector_store = FAISS.load_local(
    VECTOR_STORE_PATH,
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)


# --- helpers ---

def pick_model(query: str, contexts: list[str]) -> str:
    total_context_len = sum(len(c) for c in contexts)
    if len(query) < 50 and total_context_len < 500:
        return "gpt-4o-mini"
    if len(query) > 200 or total_context_len > 3000:
        return "gpt-4o"
    return DEFAULT_MODEL


def log_cost(model: str, input_tokens: int, output_tokens: int, latency_ms: float):
    pricing = PRICING.get(model, {"input": 0, "output": 0})
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    cost_logger.info(
        f"model={model} tokens_in={input_tokens} tokens_out={output_tokens} "
        f"cost=${cost:.6f} latency={latency_ms:.0f}ms"
    )
    return cost


# --- routes ---

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("question", "").strip()

    if not query:
        return {"error": "question is required"}

    # safety check
    safety = check_input(query)
    if not safety["safe"]:
        return {"error": f"Input blocked: {safety['reason']}"}

    # retrieve with scores
    results = vector_store.similarity_search_with_score(query, k=TOP_K)

    # filter by minimum similarity
    filtered = [(doc, score) for doc, score in results if score >= MIN_SIMILARITY_SCORE]

    # log retrieval
    rag_logger.info(json.dumps({
        "question": query,
        "chunks": [
            {"content": doc.page_content[:100], "score": round(score, 3)}
            for doc, score in results
        ],
        "filtered_count": len(filtered),
    }, ensure_ascii=False))

    if not filtered:
        return {
            "answer": "I don't have enough information to answer this question.",
            "sources": [],
            "no_results": True,
        }

    # build context
    contexts = [doc.page_content for doc, _ in filtered]
    context = "\n---\n".join(contexts)
    model = pick_model(query, contexts)

    # generate
    start = time.time()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ],
    )
    latency = (time.time() - start) * 1000

    log_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens, latency)

    answer = mask_pii_in_output(response.choices[0].message.content)

    return {
        "answer": answer,
        "model": model,
        "sources": [doc.page_content[:200] for doc, _ in filtered],
    }


@app.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    query = body.get("question", "").strip()

    if not query:
        return {"error": "question is required"}

    safety = check_input(query)
    if not safety["safe"]:
        return {"error": f"Input blocked: {safety['reason']}"}

    results = vector_store.similarity_search_with_score(query, k=TOP_K)
    filtered = [(doc, score) for doc, score in results if score >= MIN_SIMILARITY_SCORE]

    if not filtered:
        async def no_results():
            yield f"data: {json.dumps({'content': 'I don\\'t have enough information to answer this question.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    contexts = [doc.page_content for doc, _ in filtered]
    context = "\n---\n".join(contexts)
    model = pick_model(query, contexts)

    async def generate():
        # send sources first
        yield f"data: {json.dumps({'sources': [doc.page_content[:200] for doc, _ in filtered]})}\n\n"

        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": query},
            ],
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {json.dumps({'content': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Minimal RAG</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #fafafa; }
  h1 { margin-bottom: 8px; font-size: 1.5rem; }
  p.sub { color: #666; margin-bottom: 24px; font-size: 0.9rem; }
  .input-row { display: flex; gap: 8px; margin-bottom: 16px; }
  input { flex: 1; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 1rem; }
  button { padding: 10px 20px; background: #111; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; }
  button:hover { background: #333; }
  #answer { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; min-height: 60px; white-space: pre-wrap; line-height: 1.6; }
  #sources { margin-top: 12px; font-size: 0.85rem; color: #888; }
  .chunk { background: #f0f0f0; padding: 8px 12px; border-radius: 6px; margin-top: 6px; font-family: monospace; font-size: 0.8rem; }
</style>
</head><body>
<h1>Minimal RAG</h1>
<p class="sub">Ask questions about the indexed documents. Answers stream in real-time.</p>
<div class="input-row">
  <input id="q" placeholder="Ask a question..." autofocus>
  <button onclick="ask()">Ask</button>
</div>
<div id="answer"></div>
<div id="sources"></div>
<script>
async function ask() {
  const q = document.getElementById("q").value.trim();
  if (!q) return;
  const answer = document.getElementById("answer");
  const sources = document.getElementById("sources");
  answer.textContent = "";
  sources.innerHTML = "";
  const res = await fetch("/chat/stream", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question: q}),
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value).split("\\n")) {
      if (line.startsWith("data: ") && line !== "data: [DONE]") {
        const data = JSON.parse(line.slice(6));
        if (data.content) answer.textContent += data.content;
        if (data.sources) {
          sources.innerHTML = "<b>Sources:</b>";
          data.sources.forEach(s => {
            sources.innerHTML += '<div class="chunk">' + s + '</div>';
          });
        }
      }
    }
  }
}
document.getElementById("q").addEventListener("keydown", e => { if (e.key === "Enter") ask(); });
</script>
</body></html>"""

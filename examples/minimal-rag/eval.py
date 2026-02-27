"""
Evaluate RAG quality using RAGAS.

Usage:
    python eval.py                    # run against golden_qa.json
    python eval.py --quick            # quick check (first 3 only)
"""

import json
import os
import sys

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

VECTOR_STORE_PATH = "vector_store"
GOLDEN_QA_PATH = "golden_qa.json"
MIN_SIMILARITY_SCORE = 0.3
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- ONLY use information from the context below to answer
- If the context does not contain the answer, say "I don't have enough information to answer this question."
- Do NOT make up information that is not in the context

Context:
{context}"""


def run_rag(query: str, store) -> dict:
    """Run the same RAG pipeline as app.py (but sync)."""
    client = OpenAI()

    results = store.similarity_search_with_score(query, k=TOP_K)
    filtered = [(doc, score) for doc, score in results if score >= MIN_SIMILARITY_SCORE]

    if not filtered:
        return {"answer": "I don't have enough information.", "contexts": []}

    contexts = [doc.page_content for doc, _ in filtered]
    context = "\n---\n".join(contexts)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "contexts": contexts,
    }


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY first.")
        sys.exit(1)

    if not os.path.exists(VECTOR_STORE_PATH):
        print("ERROR: Run 'python ingest.py' first.")
        sys.exit(1)

    quick = "--quick" in sys.argv

    print("Loading vector store...")
    store = FAISS.load_local(
        VECTOR_STORE_PATH,
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )

    with open(GOLDEN_QA_PATH) as f:
        golden = json.load(f)

    if quick:
        golden = golden[:3]
        print(f"Quick mode: testing {len(golden)} questions\n")
    else:
        print(f"Testing {len(golden)} questions\n")

    # --- Step 1: Run RAG and check basic quality ---
    results = []
    for i, item in enumerate(golden):
        q = item["question"]
        print(f"  [{i+1}/{len(golden)}] {q}")

        result = run_rag(q, store)
        answer = result["answer"]
        keyword = item.get("expected_keyword", "").lower()

        has_keyword = keyword in answer.lower() if keyword else True
        has_context = len(result["contexts"]) > 0

        results.append({
            "question": q,
            "answer": answer[:200],
            "contexts": result["contexts"],
            "ground_truth": item["ground_truth"],
            "keyword_found": has_keyword,
            "has_context": has_context,
        })

        status = "PASS" if has_keyword and has_context else "FAIL"
        print(f"         {status} (keyword={has_keyword}, context={has_context})")

    # --- Step 2: Basic metrics ---
    total = len(results)
    keyword_pass = sum(1 for r in results if r["keyword_found"])
    context_pass = sum(1 for r in results if r["has_context"])

    print(f"\n--- Basic Metrics ---")
    print(f"Keyword accuracy:  {keyword_pass}/{total} ({keyword_pass/total:.0%})")
    print(f"Context retrieval: {context_pass}/{total} ({context_pass/total:.0%})")

    # --- Step 3: RAGAS evaluation ---
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

        print(f"\n--- RAGAS Evaluation ---")

        samples = []
        for r in results:
            samples.append(SingleTurnSample(
                user_input=r["question"],
                retrieved_contexts=r["contexts"] if r["contexts"] else ["No context found."],
                response=r["answer"],
                reference=r["ground_truth"],
            ))

        dataset = EvaluationDataset(samples=samples)
        scores = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy])

        print(f"Faithfulness:     {scores['faithfulness']:.2f}")
        print(f"Answer Relevancy: {scores['answer_relevancy']:.2f}")

        if scores["faithfulness"] < 0.7:
            print("\nWARNING: Faithfulness is below 0.7 — your RAG may be hallucinating.")
        if scores["answer_relevancy"] < 0.6:
            print("\nWARNING: Answer relevancy is below 0.6 — answers may not match questions.")

        if scores["faithfulness"] >= 0.7 and scores["answer_relevancy"] >= 0.6:
            print("\nRAG quality check PASSED.")
        else:
            print("\nRAG quality check FAILED.")
            sys.exit(1)

    except ImportError:
        print("\nSkipping RAGAS evaluation (pip install ragas to enable).")
        print("Running basic checks only.")

    # basic check
    if keyword_pass / total < 0.7:
        print(f"\nFAILED: Keyword accuracy {keyword_pass/total:.0%} is below 70%")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

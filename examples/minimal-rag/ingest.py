"""
Ingest documents into a FAISS vector store.

Usage:
    python ingest.py                          # ingest sample_docs/
    python ingest.py /path/to/your/docs       # ingest your own docs
"""

import os
import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

VECTOR_STORE_PATH = "vector_store"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def ingest(docs_dir: str = "sample_docs"):
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"ERROR: Directory '{docs_dir}' not found.")
        sys.exit(1)

    print(f"Loading documents from {docs_path}...")
    loader = DirectoryLoader(str(docs_path), glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"  Loaded {len(documents)} document(s)")

    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunk(s)")

    # debug: show first few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  --- Chunk {i} ({len(chunk.page_content)} chars) ---")
        print(f"  {chunk.page_content[:150]}...")

    print(f"\nEmbedding and saving to {VECTOR_STORE_PATH}/...")
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(VECTOR_STORE_PATH)
    print(f"Done. {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "sample_docs"
    ingest(docs_dir)

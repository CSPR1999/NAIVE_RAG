#!/usr/bin/env python3
"""
Naive RAG – Example usage.

1. Index documents from the `data/` folder.
2. Ask questions and get answers grounded in those documents.

Before running:
  - pip install -r requirements.txt
  - Set OPENAI_API_KEY in your environment or create a .env file (see .env.example)
"""

import os
import sys
from pathlib import Path

# Load .env if present (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from naive_rag import NaiveRAG


def main():
    # Path to your documents (default: data/ with .txt files)
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        print(f"Creating example data folder: {data_dir}")
        data_dir.mkdir(parents=True)
        (data_dir / "faq.txt").write_text(
            "Returns: 30 days. Refunds in 5-7 days. Contact support@example.com."
        )
        print("Added a minimal faq.txt. Add more .txt files and run again.")
        return

    print("Building Naive RAG index...")
    rag = NaiveRAG(chunk_size=256, chunk_overlap=32, top_k=3)
    num_chunks = rag.index_directory(data_dir)
    print(f"Indexed {num_chunks} chunks from {data_dir}\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Set it in .env or environment to use the LLM.")
        print("You can still test retrieval (no answer):")
        chunks_with_scores = rag.retriever.retrieve("How do I return an item?", top_k=3)
        for chunk, score in chunks_with_scores:
            print(f"  score={score:.3f} | {chunk.text[:80]}...")
        return

    # Example questions
    questions = [
        "What is your return policy?",
        "How can I contact support?",
        "How long does shipping take?",
    ]

    for q in questions:
        print(f"Q: {q}")
        result = rag.query(q)
        print(f"A: {result['answer']}\n")

    # Optional: interactive mode (run with --interactive or -i)
    if "--interactive" in sys.argv or "-i" in sys.argv:
        print("Interactive mode. Type a question and press Enter (or 'quit' to exit).")
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit", "q"):
                break
            result = rag.query(q)
            print(f"RAG: {result['answer']}")


if __name__ == "__main__":
    main()

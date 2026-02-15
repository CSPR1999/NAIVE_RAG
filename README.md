# Naive RAG in Python

A minimal **Retrieval-Augmented Generation (RAG)** pipeline: convert documents into embeddings, retrieve similar chunks for each question, and send them to an LLM to get an answer. Best for FAQ bots, internal knowledge search, and simple support systems.

---

## What is Naive RAG?

**RAG** = **R**etrieval **A**ugmented **G**eneration. Instead of asking the LLM from memory alone, we:

1. **Index**: Turn your documents into small chunks and then into **embeddings** (vectors).
2. **Retrieve**: For each user question, find the **most similar** chunks (by embedding similarity).
3. **Generate**: Pass those chunks as **context** to the LLM and ask it to answer the question using only that context.

So the model answers **from your data**, not from its training. That reduces hallucination and keeps answers aligned with your docs.

**“Naive”** here means we keep everything simple:

- Fixed-size text chunks with overlap
- One embedding model for both chunks and queries
- In-memory vector store with cosine similarity (no external DB)
- One retrieval step, then one LLM call

No reranking, no query rewriting, no hybrid search—just the core flow.

---

## Pipeline Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│  Documents  │ ──► │ Chunk +      │ ──► │  Embed &    │ ──► │   Store     │
│  (.txt)     │     │ Overlap      │     │  Vectorize  │     │  Vectors    │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬──────┘
                                                                   │
  User question ──► Embed question ──► Similarity search ──────────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Answer    │ ◄── │  LLM         │ ◄── │  Context +  │
│             │     │  (OpenAI)    │     │  Question   │
└─────────────┘     └──────────────┘     └─────────────┘
```

---

## Project Structure

```
Naive_RAG/
├── naive_rag/
│   ├── __init__.py
│   ├── document_loader.py   # Load & chunk documents
│   ├── embeddings.py        # Turn text → vectors (sentence-transformers)
│   ├── retriever.py         # Store vectors, retrieve by similarity
│   └── pipeline.py          # Full pipeline: index + query → answer
├── data/
│   └── faq.txt              # Example knowledge base
├── main.py                  # Run example / interactive mode
├── requirements.txt
├── .env.example
└── README.md
```

---

## Step-by-Step (What Each Part Does)

### 1. Document loader (`document_loader.py`)

- **Why chunk?** LLMs have limited context; we can’t send whole docs. Small chunks also make retrieval more precise.
- **How:** Split text into segments of `chunk_size` characters with `chunk_overlap` so we don’t cut sentences in the middle.
- **Output:** List of `Chunk` objects (text + optional source/metadata).

### 2. Embeddings (`embeddings.py`)

- **What’s an embedding?** A vector that represents the *meaning* of the text. Similar meanings → similar vectors.
- **Why?** So we can search by *semantics* (e.g. “return policy” matches “how do I send something back?”) instead of only keywords.
- **How:** We use **sentence-transformers** (e.g. `all-MiniLM-L6-v2`). It runs locally; no API key for this step.

### 3. Retriever (`retriever.py`)

- **Role:** Store chunk vectors and, for a query vector, return the **top-k** most similar chunks.
- **Similarity:** Cosine similarity between query embedding and each chunk embedding (higher = more similar).
- **Naive choice:** In-memory list + brute-force search. For large scale you’d swap this for FAISS, Chroma, etc.

### 4. Pipeline (`pipeline.py`)

- **Index:** `index_directory()` or `index_file()` / `index_text()` → load → chunk → embed → add to retriever.
- **Query:** Embed the question → retrieve top-k chunks → build a prompt: “Context: … Question: …” → call OpenAI → return the model’s answer plus the chunks used (sources).

---

## Setup and Run

### 1. Install dependencies

```bash
cd Naive_RAG
pip install -r requirements.txt
```

### 2. OpenAI API key (for the LLM step)

Create a `.env` file (see `.env.example`):

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

Or set the env var in your shell:

```bash
export OPENAI_API_KEY=sk-your-key-here
```

Embeddings use **sentence-transformers** (local), so no key is needed for indexing/retrieval; only the final answer step needs OpenAI.

### 3. Add your documents

Put `.txt` files in `data/` (or point the code to another folder). `data/faq.txt` is an example.

### 4. Run the example

```bash
python main.py
```

This indexes `data/` and runs a few example questions.

### 5. Interactive Q&A (optional)

```bash
python main.py --interactive
# or
python main.py -i
```

Then type questions and get answers from your indexed docs.

---

## Usage in Code

```python
from naive_rag import NaiveRAG

# Build RAG and index a folder of .txt files
rag = NaiveRAG(chunk_size=512, chunk_overlap=64, top_k=5)
rag.index_directory("data/")

# Ask a question
result = rag.query("What is your return policy?")
print(result["answer"])
# Optional: result["sources"] and result["context_used"]
```

You can also:

- `rag.index_file("path/to/file.txt")`
- `rag.index_text("Some long text...", source="mydoc")`

---

## Pros and Cons of Naive RAG

| Pros | Cons |
|------|------|
| Easy to build and understand | Weak on multi-hop or complex reasoning |
| Low cost (small context, one LLM call) | Can miss deeper connections across chunks |
| Fast (simple retrieval + one call) | No reranking or query expansion |
| Good for FAQs and internal docs | Sensitive to chunk boundaries and top_k |

---

## When to Use It

- **FAQ bots** – one question → one clear answer from a small set of chunks.
- **Internal knowledge search** – “where do we say X?” with an answer in natural language.
- **Simple support** – policy, shipping, contact info, etc.

For harder tasks (multi-doc reasoning, nuanced analysis), you’d move toward **advanced RAG**: better chunking, reranking, hybrid search, or agents.

---

## License

Use and modify as you like; no formal license specified.

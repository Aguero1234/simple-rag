# simple-rag

> A minimal RAG (Retrieval-Augmented Generation) pipeline in Python.
> Works with any OpenAI-compatible API — including DeepSeek.

## What It Does

```
Your Documents → Chunk → Embed → Store in Vector DB
                                           ↓
User Question → Embed → Search Vector DB → Top-K Chunks → LLM → Answer
```

**One command to index, one command to ask.**

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set your API key (DeepSeek, OpenAI, or any compatible API)
export API_KEY="sk-your-key-here"
export API_BASE="https://api.deepseek.com/v1"  # default

# Index your documents
python rag.py index ./my-docs/

# Ask questions
python rag.py ask "What is the main idea of these documents?"
```

## Features

- **Zero config** — works out of the box with sensible defaults
- **Any LLM** — DeepSeek, OpenAI, Ollama, anything with OpenAI-compatible API
- **Local vector DB** — ChromaDB, runs locally, no cloud dependency
- **Smart chunking** — splits by paragraph, respects sentence boundaries
- **CLI first** — index and ask from terminal, no web UI needed

## How It Works

### Indexing
1. Reads all `.txt`, `.md`, `.pdf` files from a directory
2. Splits text into chunks (~500 tokens each)
3. Embeds chunks using the LLM API
4. Stores embeddings in ChromaDB

### Querying
1. Embeds your question
2. Finds top-K most similar chunks from ChromaDB
3. Sends chunks + question to LLM
4. Returns the answer with sources

## File Structure

```
simple-rag/
├── rag.py              # Main entry point
├── chunker.py          # Text splitting logic
├── embedder.py         # Embedding via OpenAI-compatible API
├── store.py            # ChromaDB vector store
├── llm.py              # LLM client for generation
├── requirements.txt
└── README.md
```

## Configuration

All via environment variables:

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | Required | Your LLM API key |
| `API_BASE` | `https://api.deepseek.com/v1` | API base URL |
| `MODEL` | `deepseek-chat` | Chat model name |
| `EMBED_MODEL` | `text-embedding-v3` | Embedding model name |
| `CHUNK_SIZE` | `500` | Max tokens per chunk |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `DB_DIR` | `./chroma_db` | ChromaDB storage path |

## Use as a Library

```python
from rag import RAG

rag = RAG(api_key="sk-xxx", api_base="https://api.deepseek.com/v1")

# Index
rag.index("./documents/")

# Query
answer = rag.ask("What are the key findings?")
print(answer)
```

## License

MIT

"""simple-rag: A minimal RAG pipeline that works with any OpenAI-compatible API.

Usage:
    python rag.py index <directory>    # Index documents
    python rag.py ask "<question>"     # Ask a question
    python rag.py reset                # Clear the vector store
"""
import os
import sys
from typing import List

from chunker import chunk_text
from embedder import Embedder
from store import VectorStore
from llm import LLM


# --- Config from env ---
API_KEY = os.environ.get("API_KEY", "")
API_BASE = os.environ.get("API_BASE", "https://api.deepseek.com/v1")
MODEL = os.environ.get("MODEL", "deepseek-chat")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-v3")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
TOP_K = int(os.environ.get("TOP_K", "5"))
DB_DIR = os.environ.get("DB_DIR", "./chroma_db")


class RAG:
    """One class to rule them all."""

    def __init__(self, api_key: str = "", api_base: str = "", model: str = "",
                 embed_model: str = "", db_dir: str = "", top_k: int = 0):
        key = api_key or API_KEY
        base = api_base or API_BASE
        if not key:
            raise ValueError("API_KEY is required (env var or constructor arg)")

        self.embedder = Embedder(key, base, embed_model or EMBED_MODEL)
        self.store = VectorStore(db_dir or DB_DIR)
        self.llm = LLM(key, base, model or MODEL)
        self.top_k = top_k or TOP_K

    def index(self, directory: str):
        """Index all documents in a directory."""
        documents = self._load_documents(directory)
        if not documents:
            print("No documents found.")
            return

        print(f"Found {len(documents)} documents")

        # Chunk all documents
        all_chunks = []
        all_metadata = []
        for doc in documents:
            chunks = chunk_text(doc["text"], max_tokens=CHUNK_SIZE)
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({"source": doc["path"], "chunk": j})

        print(f"Split into {len(all_chunks)} chunks")

        # Embed in batches
        print("Embedding...")
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch_texts = all_chunks[i : i + batch_size]
            batch_meta = all_metadata[i : i + batch_size]
            embeddings = self.embedder.embed(batch_texts)
            self.store.add(batch_texts, embeddings, batch_meta)
            print(f"  Indexed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

        print(f"Done! Total chunks in store: {self.store.count()}")

    def ask(self, question: str) -> str:
        """Ask a question and get an answer based on indexed documents."""
        if self.store.count() == 0:
            return "No documents indexed yet. Run `python rag.py index <dir>` first."

        # Embed question
        query_embedding = self.embedder.embed_query(question)

        # Search
        results = self.store.search(query_embedding, top_k=self.top_k)
        contexts = [r["text"] for r in results]

        # Print sources
        print("\n📄 Sources:")
        for i, r in enumerate(results):
            src = r["metadata"].get("source", "unknown")
            dist = r["distance"]
            print(f"  [{i+1}] {src} (distance: {dist:.4f})")

        # Generate answer
        print("\n🤖 Answer:")
        answer = self.llm.generate(question, contexts)
        return answer

    def reset(self):
        """Clear the vector store."""
        self.store.reset()
        print("Vector store cleared.")

    def _load_documents(self, directory: str) -> List[dict]:
        """Load .txt, .md, and .pdf files from a directory."""
        documents = []
        for root, dirs, files in os.walk(directory):
            for fname in files:
                fpath = os.path.join(root, fname)
                ext = fname.rsplit(".", 1)[-1].lower()

                if ext in ("txt", "md"):
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        documents.append({"path": fpath, "text": f.read()})
                elif ext == "pdf":
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(fpath)
                        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                        if text.strip():
                            documents.append({"path": fpath, "text": text})
                    except ImportError:
                        print(f"  Skipping {fname} (install pypdf for PDF support)")
                    except Exception as e:
                        print(f"  Skipping {fname}: {e}")

        return documents


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    rag = RAG()

    if command == "index":
        if len(sys.argv) < 3:
            print("Usage: python rag.py index <directory>")
            sys.exit(1)
        rag.index(sys.argv[2])

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: python rag.py ask \"<question>\"")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        answer = rag.ask(question)
        print(answer)

    elif command == "reset":
        rag.reset()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()

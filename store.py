"""Vector store — ChromaDB-backed storage for document chunks."""
from typing import List, Dict, Optional
import chromadb


class VectorStore:
    def __init__(self, db_dir: str = "./chroma_db", collection_name: str = "rag_docs"):
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict]] = None):
        """Add documents with pre-computed embeddings."""
        ids = [f"doc_{i}" for i in range(self.collection.count(), self.collection.count() + len(texts))]
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(texts),
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar documents using a query embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
            })
        return docs

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete all documents."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

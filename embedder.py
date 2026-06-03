"""Embedder — creates embeddings via OpenAI-compatible API."""
from typing import List
from openai import OpenAI


class Embedder:
    def __init__(self, api_key: str, api_base: str, model: str = "text-embedding-v3"):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts. Returns list of embedding vectors."""
        if not texts:
            return []

        # Batch in groups of 100 (API limit)
        all_embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        result = self.embed([query])
        return result[0] if result else []

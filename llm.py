"""LLM client — generates answers using OpenAI-compatible API."""
from typing import List
from openai import OpenAI


SYSTEM_PROMPT = """You are a helpful assistant answering questions based on the provided context.
Rules:
- Answer ONLY using the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer this."
- Cite which source(s) you used in your answer
- Be concise and accurate"""


class LLM:
    def __init__(self, api_key: str, api_base: str, model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def generate(self, question: str, contexts: List[str]) -> str:
        """Generate an answer given a question and retrieved context chunks."""
        context_text = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\n---\n\nQuestion: {question}",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )

        return response.choices[0].message.content or ""

"""Text chunker — splits documents into chunks respecting sentence boundaries."""
import re
from typing import List


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks of ~max_tokens with overlap.

    Strategy: split by paragraphs first, then merge/split to fit token budget.
    Approximate tokens as words * 1.3 for English, * 2 for Chinese.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        estimated_tokens = _estimate_tokens(current + "\n\n" + para if current else para)

        if estimated_tokens <= max_tokens:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current.strip())
            # If single paragraph exceeds limit, split by sentences
            if _estimate_tokens(para) > max_tokens:
                sentences = _split_sentences(para)
                current = ""
                for sent in sentences:
                    candidate = current + " " + sent if current else sent
                    if _estimate_tokens(candidate) <= max_tokens:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = sent
            else:
                current = para

    if current.strip():
        chunks.append(current.strip())

    # Add overlap: prepend tail of previous chunk to current chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1].split()[-overlap:]
            overlapped.append(" ".join(prev_tail) + " " + chunks[i])
        chunks = overlapped

    return chunks


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: Chinese chars count more."""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars / 4)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r'(?<=[.!?。！？])\s+', text)
    return [p.strip() for p in parts if p.strip()]

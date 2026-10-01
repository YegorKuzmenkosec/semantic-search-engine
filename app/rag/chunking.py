import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    index: int
    text: str


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    if len(normalized) <= chunk_size:
        return [TextChunk(index=0, text=normalized)]

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            split_at = normalized.rfind(" ", start, end)
            if split_at > start + chunk_size // 2:
                end = split_at

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(TextChunk(index=index, text=piece))
            index += 1

        if end >= len(normalized):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks

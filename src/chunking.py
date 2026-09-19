from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text.strip())
            if s.strip()
        ]
        if not raw_sentences:
            return []
        chunks: list[str] = []
        for i in range(0, len(raw_sentences), self.max_sentences_per_chunk):
            group = raw_sentences[i : i + self.max_sentences_per_chunk]
            chunk_str = " ".join(group).strip()
            if chunk_str:
                chunks.append(chunk_str)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        if sep == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        if sep not in current_text:
            return self._split(current_text, next_seps)

        parts = current_text.split(sep)
        chunks: list[str] = []
        current_chunk = ""

        for part in parts:
            if len(part) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.extend(self._split(part, next_seps))
            else:
                candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
                if len(candidate) <= self.chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return [c for c in chunks if c]


class HeadingChunker:
    """
    Split Markdown text into chunks at heading boundaries (one section per chunk).

    Design rationale (K4-L3A, university regulations): policy documents are
    already organised into sections ("## Điều 4 — ...") by their authors, and
    each section is a self-contained semantic unit. Splitting there keeps the
    rule, its conditions and its numbers together.

    Rules:
        - A section = heading line + everything up to the next heading.
        - Sections longer than chunk_size are split further with
          RecursiveChunker, and the heading is re-attached to every piece so
          that no piece loses the "what is this section about" context.
        - Text without any heading falls back to RecursiveChunker.
    """

    _HEADING = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)

    def __init__(self, chunk_size: int = 800, min_level: int = 1, max_level: int = 6) -> None:
        self.chunk_size = chunk_size
        self.min_level = min_level
        self.max_level = max_level
        self._fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sections = self._split_sections(text)
        if not sections:
            return self._fallback.chunk(text)

        chunks: list[str] = []
        for heading, body in sections:
            full = f"{heading}\n{body}".strip() if heading else body.strip()
            if not full:
                continue
            if len(full) <= self.chunk_size:
                chunks.append(full)
                continue
            # Too long: split the body, then prefix the heading onto each piece.
            for piece in self._fallback.chunk(body.strip()):
                piece = piece.strip()
                if piece:
                    chunks.append(f"{heading}\n{piece}" if heading else piece)
        return chunks

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Return [(heading_line, body)] — heading may be "" for a preamble."""
        matches = [
            m
            for m in self._HEADING.finditer(text)
            if self.min_level <= len(m.group(1)) <= self.max_level
        ]
        if not matches:
            return []

        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = m.group(0).strip()
            body = text[m.end() : end].strip()
            sections.append((heading, body))
        return sections


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_val = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    val = dot_val / (norm_a * norm_b)
    return max(-1.0, min(1.0, float(val)))


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, min(50, chunk_size // 5)))
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        def _stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_len = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            return {
                "count": count,
                "avg_length": avg_len,
                "chunks": chunks,
            }

        return {
            "fixed_size": _stats(fixed_chunks),
            "by_sentences": _stats(sentence_chunks),
            "recursive": _stats(recursive_chunks),
        }

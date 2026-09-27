"""Pure-Python BM25 index over locally mirrored official documentation. No network."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*|\d+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenize Latin identifiers, numbers, and individual CJK characters."""
    return TOKEN.findall((text or "").lower())


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    url: str
    title: str
    toolkit_version: str
    text: str


def split_chunks(doc_id: str, text: str, meta: dict, min_chars: int = 800) -> list[Chunk]:
    """Accumulate document lines into bounded, stable numbered chunks."""
    max_chars = max(2 * min_chars, 2400)
    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n|\n", text or ""):
        paragraph = raw.strip()
        while len(paragraph) > max_chars:
            split_at = paragraph.rfind(" ", min_chars, max_chars + 1)
            split_at = split_at if split_at > 0 else max_chars
            paragraphs.append(paragraph[:split_at].strip())
            paragraph = paragraph[split_at:].strip()
        if paragraph:
            paragraphs.append(paragraph)
    chunks: list[Chunk] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) + 2 > max_chars:
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", str(meta.get("url") or ""),
                                str(meta.get("title") or doc_id), str(meta.get("toolkit_version") or ""),
                                "\n\n".join(current)))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + (2 if len(current) > 1 else 0)
        if size >= min_chars:
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", str(meta.get("url") or ""),
                                str(meta.get("title") or doc_id), str(meta.get("toolkit_version") or ""),
                                "\n\n".join(current)))
            current, size = [], 0
    if current:
        chunks.append(Chunk(f"{doc_id}#{len(chunks)}", str(meta.get("url") or ""),
                            str(meta.get("title") or doc_id), str(meta.get("toolkit_version") or ""),
                            "\n\n".join(current)))
    return chunks


class DocIndex:
    """BM25 retrieval over already-mirrored text chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.terms = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.df = Counter(term for terms in self.terms for term in terms)

    @classmethod
    def from_dir(cls, docs_dir: Path) -> "DocIndex":
        """Read each text file with a sibling metadata JSON file."""
        chunks: list[Chunk] = []
        for path in sorted(Path(docs_dir).glob("*.txt")):
            meta_path = path.with_suffix(".meta.json")
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not meta.get("url"):
                    continue
                chunks.extend(split_chunks(path.stem, path.read_text(encoding="utf-8"), meta))
            except (OSError, ValueError, UnicodeError):
                continue
        return cls(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 6) -> list[tuple[Chunk, float]]:
        """Return top BM25 chunks by score, then chunk id."""
        query_terms = tokenize(query)
        if not self.chunks or not query_terms or k <= 0:
            return []
        n = len(self.chunks)
        ranked: list[tuple[Chunk, float]] = []
        for chunk, terms, length in zip(self.chunks, self.terms, self.lengths):
            score = 0.0
            for term in query_terms:
                tf = terms.get(term, 0)
                if not tf:
                    continue
                df = self.df[term]
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avgdl))
            if score > 0:
                ranked.append((chunk, score))
        ranked.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return ranked[:k]

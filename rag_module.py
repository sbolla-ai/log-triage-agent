"""
rag_module.py — A minimal, honest RAG implementation with caching + backoff.

Four components:
  1. chunk_runbooks()       — section-level markdown splitting
  2. Corpus class           — holds chunks + their embeddings in memory
  3. Corpus.retrieve()      — cosine similarity top-k
  4. rag_runbook_lookup()   — a tool function agents can call

Production hardening included:
  - On-disk cache of embeddings, keyed by content hash
  - Exponential backoff on Voyage rate-limit errors
"""
from __future__ import annotations
import hashlib
import pickle
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import voyageai
import voyageai.error
from dotenv import load_dotenv

load_dotenv()
_voyage = voyageai.Client()


# =========================================================================
# UTILITIES
# =========================================================================

def _embed_with_backoff(texts, model, input_type, max_retries=3):
    """Retry on RateLimitError with exponential backoff + jitter."""
    for attempt in range(max_retries + 1):
        try:
            return _voyage.embed(texts=texts, model=model, input_type=input_type)
        except voyageai.error.RateLimitError:
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.uniform(0, 1)
            print(f"[rag] rate-limited; sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)


# =========================================================================
# CHUNK REPRESENTATION
# =========================================================================

@dataclass
class Chunk:
    text: str
    source: str
    section: str


# =========================================================================
# CHUNKING — split markdown files by '##' sections
# =========================================================================

def chunk_runbooks(runbook_dir):
    """Split each .md file under runbook_dir on '##' headings."""
    runbook_dir = Path(runbook_dir).resolve()
    chunks = []
    for path in sorted(runbook_dir.glob("*.md")):
        content = path.read_text()
        parts = re.split(r"^## ", content, flags=re.MULTILINE)
        for part in parts[1:]:  # skip pre-first-heading content
            lines = part.strip().split("\n", 1)
            heading = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            chunk_text = f"## {heading}\n{body}".strip()
            chunks.append(Chunk(
                text=chunk_text,
                source=path.name,
                section=heading,
            ))
    return chunks


# =========================================================================
# CORPUS — chunks + their embeddings, all in memory
# =========================================================================

class Corpus:
    """In-memory corpus with on-disk embedding cache."""

    def __init__(self, chunks, model="voyage-3.5"):
        self.chunks = chunks
        self.model = model
        self.embeddings = None
        self._build_index()

    def _build_index(self):
        if not self.chunks:
            self.embeddings = np.zeros((0, 1024), dtype=np.float32)
            return

        texts = [c.text for c in self.chunks]

        # Cache key: hash of chunk contents + model name
        content_key = hashlib.sha256(
            "||".join(texts).encode() + self.model.encode()
        ).hexdigest()[:16]
        cache_path = Path(f".rag_cache_{content_key}.pkl")

        if cache_path.exists():
            print(f"[rag] loading cached embeddings from {cache_path}")
            with open(cache_path, "rb") as f:
                self.embeddings = pickle.load(f)
            print(f"[rag] loaded {len(self.chunks)} chunks from cache, "
                  f"embedding_dim={self.embeddings.shape[1]}")
            return

        print(f"[rag] cache miss — embedding {len(self.chunks)} chunks via Voyage...")
        result = _embed_with_backoff(texts, self.model, "document")
        self.embeddings = np.array(result.embeddings, dtype=np.float32)

        # Normalize once so retrieval is pure dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms

        with open(cache_path, "wb") as f:
            pickle.dump(self.embeddings, f)
        print(f"[rag] indexed {len(self.chunks)} chunks, "
              f"embedding_dim={self.embeddings.shape[1]} (cached to {cache_path})")

    def retrieve(self, query, top_k=3, min_score=0.45):
        if not self.chunks:
            return []

        result = _embed_with_backoff([query], self.model, "query")
        query_vec = np.array(result.embeddings[0], dtype=np.float32)

        qn = np.linalg.norm(query_vec)
        if qn == 0:
            return []
        query_vec = query_vec / qn

        scores = self.embeddings @ query_vec  # cosine since both normalized
        top_idx = np.argsort(-scores)[:top_k]
        return [
            (self.chunks[i], float(scores[i])) for i in top_idx
            if scores[i] >= min_score
        ]


# =========================================================================
# PUBLIC TOOL FUNCTION
# =========================================================================

_CORPUS = None


def _get_corpus():
    global _CORPUS
    if _CORPUS is None:
        chunks = chunk_runbooks("./runbooks")
        _CORPUS = Corpus(chunks)
    return _CORPUS


def rag_runbook_lookup(query, top_k=3):
    """Retrieve top_k runbook sections for a query with cited sources."""
    corpus = _get_corpus()
    hits = corpus.retrieve(query, top_k=top_k)
    if not hits:
        return (f"No runbook sections matched '{query}' with sufficient confidence. "
                f"Do NOT invent procedures if the synthesizer needs to answer a "
                f"question this refers to.")
    lines = [f"Retrieved {len(hits)} runbook section(s) for query: '{query}'\n"]
    for chunk, score in hits:
        lines.append(f"--- SOURCE: {chunk.source} § {chunk.section}  (relevance: {score:.2f}) ---")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)


# =========================================================================
# SANITY CHECK
# =========================================================================

if __name__ == "__main__":
    import time
    
    corpus = _get_corpus()

    test_queries = [
        "How do I fix an SSL certificate that expired?",
        "Database is timing out when I try to connect.",
        "My nginx keeps returning 502",
        "What's the capital of France?",
        "How to roll out a deployment safely",
    ]

    # Voyage free tier: 3 RPM = one call every 20s.
    # Index build used 1 call; each query is 1 call. Sleep 25s between
    # queries to stay comfortably under the cap with headroom.
    INTER_QUERY_SLEEP = 25

    for i, q in enumerate(test_queries):
        if i > 0:
            print(f"\n[demo] sleeping {INTER_QUERY_SLEEP}s to respect Voyage free-tier rate limit...")
            time.sleep(INTER_QUERY_SLEEP)
        print(f"\n{'=' * 70}")
        print(f"QUERY: {q}")
        print('=' * 70)
        print(rag_runbook_lookup(q, top_k=2))
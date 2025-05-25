"""
Result formatting utilities for different experiment types.
"""

from typing import Callable
from rag_testing.core.base import SearchResult

def format_hybrid_result(result: SearchResult) -> str:
    """Format hybrid search results for display."""
    return f"[ID: {result.document.id}] (Emb: {result.embedding_score:.4f}, TFIDF: {result.tfidf_score:.4f}, Combined: {result.score:.4f})\n  {result.document.content}" 

def default_formatter(result: SearchResult) -> str:
    return f"[ID: {result.document.id}] (Similarity: {result.score:.4f})\n  {result.document.content}"


def get_formatter(type: str) -> Callable[[SearchResult], str]:
    formatters = {
        "hybrid": format_hybrid_result,
    }
    return formatters.get(type, default_formatter)
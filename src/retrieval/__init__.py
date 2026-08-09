"""
Pacote de Recuperação (Retrieval)

Contém:
- VectorStore: Banco vetorial ChromaDB
- Reranker: Cross-Encoder para reranking
- HybridSearch: Busca híbrida (semântica + palavras-chave)
"""

from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_search import HybridSearch

__all__ = [
    "VectorStore",
    "Reranker",
    "HybridSearch"
]

__version__ = "1.0.0"
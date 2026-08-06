"""
Pacote do Agente RAG com LangGraph

Exporta as classes principais para uso externo.
"""

from src.agent.graph import RAGGraph, RAGState

__all__ = [
    "RAGGraph",
    "RAGState"
]

__version__ = "1.0.0"
"""
Reranking para melhorar a precisão da busca
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import numpy as np


class Reranker:
    """
    Reordena resultados usando um Cross-Encoder para melhor precisão
    
    Cross-Encoder: Analisa a relação entre pergunta e documento
    (diferente do embedding, que analisa apenas similaridade semântica)
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: Nome do modelo Cross-Encoder
            - cross-encoder/ms-marco-MiniLM-L-6-v2 (rápido, 80MB, bom para produção)
            - cross-encoder/ms-marco-MiniLM-L-12-v2 (mais preciso, 420MB)
            - cross-encoder/ms-marco-MiniLM-L-4-v2 (mais rápido, 40MB)
        """
        print(f"🔧 Inicializando Reranker: {model_name}")
        self.model = CrossEncoder(model_name, device='cpu')
        self.model_name = model_name
        print(f"✅ Reranker pronto")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Reordena documentos baseado na relevância com a pergunta
        
        Args:
            query: Pergunta do usuário
            documents: Lista de documentos (com 'content' e 'metadata')
            top_k: Quantos documentos manter após reranking
            
        Returns:
            Lista de documentos reordenados com novos scores
        """
        if not documents:
            return []
        
        # Prepara pares (query, documento) para o Cross-Encoder
        pairs = []
        for doc in documents:
            # Usa o conteúdo do documento como contexto
            content = doc.get('content', '')
            # Limita a 512 tokens para performance
            if len(content) > 2000:
                content = content[:2000]
            pairs.append([query, content])
        
        # Executa o Cross-Encoder
        # Retorna scores entre 0 e 1 (quanto maior, mais relevante)
        rerank_scores = self.model.predict(pairs)
        
        # Adiciona os novos scores aos documentos
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(rerank_scores[i])
            # Mantém o score original também
            doc['original_score'] = doc.get('score', 0)
        
        # Reordena por score do reranker
        reranked = sorted(
            documents,
            key=lambda x: x.get('rerank_score', 0),
            reverse=True
        )
        
        # Retorna apenas os top_k
        return reranked[:top_k]
    
    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rerank mantendo metadados
        """
        reranked = self.rerank(query, documents, top_k)
        
        # Formata com metadados
        formatted = []
        for i, doc in enumerate(reranked):
            formatted.append({
                'id': doc.get('id', f"result_{i}"),
                'content': doc.get('content', ''),
                'metadata': doc.get('metadata', {}),
                'score': doc.get('rerank_score', 0),
                'original_score': doc.get('original_score', 0),
                'position': i + 1
            })
        
        return formatted
"""
Busca Híbrida: Semântica + Palavras-chave (BM25)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import json
from pathlib import Path


class HybridSearch:
    """
    Combina busca semântica (embeddings) com busca por palavras-chave (BM25)
    
    A busca híbrida é uma prática recomendada em RAGs profissionais porque:
    1. Captura tanto o significado (semântica) quanto termos exatos (palavras-chave)
    2. Melhora a recuperação de documentos com termos específicos
    3. Reduz a dependência apenas de embeddings
    """
    
    def __init__(
        self,
        vector_store,
        embedder,
        chunks_path: str = "./data/processed/chunks.json"
    ):
        """
        Args:
            vector_store: Instância do VectorStore (ChromaDB)
            embedder: Instância do Embedder
            chunks_path: Caminho para o arquivo de chunks
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks_path = Path(chunks_path)
        self.chunks = self._load_chunks()
        self.bm25 = self._build_bm25()
        
        print(f"✅ Busca Híbrida inicializada com {len(self.chunks)} chunks")
    
    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Carrega os chunks do arquivo JSON"""
        if not self.chunks_path.exists():
            print(f"⚠️  Arquivo de chunks não encontrado: {self.chunks_path}")
            return []
        
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_bm25(self) -> BM25Okapi:
        """Constrói o índice BM25 para busca por palavras-chave"""
        # Tokeniza os textos (converte para minúsculas e divide por espaços)
        tokenized_docs = [
            doc['content'].lower().split() 
            for doc in self.chunks
        ]
        return BM25Okapi(tokenized_docs)
    
    def _keyword_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Busca por palavras-chave usando BM25
        
        Args:
            query: Texto da consulta
            top_k: Número de resultados
            
        Returns:
            Lista de resultados com scores de palavras-chave
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pega os top_k índices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Só inclui se tiver score positivo
                results.append({
                    'chunk': self.chunks[idx],
                    'keyword_score': float(scores[idx])
                })
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca híbrida: combina semântica e palavras-chave
        
        Args:
            query: Texto da consulta
            top_k: Número de resultados finais
            alpha: Peso da busca semântica (0=apenas keyword, 1=apenas semântica)
            filters: Filtros para a busca semântica
            
        Returns:
            Lista de resultados combinados e ordenados
        """
        if not self.chunks:
            print("⚠️  Nenhum chunk disponível para busca")
            return []
        
        # 1. Busca semântica (embeddings)
        query_emb = self.embedder.embed_text(query)
        semantic_results = self.vector_store.search(
            query_emb.tolist(),
            top_k=top_k * 2,  # Busca o dobro para ter mais candidatos
            filters=filters
        )
        
        # 2. Busca por palavras-chave (BM25)
        keyword_results = self._keyword_search(query, top_k=top_k * 2)
        
        # 3. Combina os resultados
        combined = {}
        
        # Adiciona resultados semânticos
        for r in semantic_results:
            chunk_id = r['id']
            combined[chunk_id] = {
                'id': chunk_id,
                'content': r['content'],
                'metadata': r.get('metadata', {}),
                'semantic_score': r.get('score', 0),
                'keyword_score': 0,
                'combined_score': alpha * r.get('score', 0)
            }
        
        # Adiciona resultados de palavras-chave
        for r in keyword_results:
            chunk = r['chunk']
            chunk_id = chunk['id']
            keyword_score = r['keyword_score']
            
            # Normaliza o keyword_score para ficar entre 0 e 1
            # (BM25 pode retornar valores grandes)
            keyword_score_norm = min(keyword_score / 10, 1.0)  # Normalização simples
            
            if chunk_id in combined:
                # Atualiza resultado existente
                combined[chunk_id]['keyword_score'] = keyword_score_norm
                combined[chunk_id]['combined_score'] = (
                    alpha * combined[chunk_id]['semantic_score'] +
                    (1 - alpha) * keyword_score_norm
                )
            else:
                # Adiciona novo resultado
                combined[chunk_id] = {
                    'id': chunk_id,
                    'content': chunk['content'],
                    'metadata': chunk.get('metadata', {}),
                    'semantic_score': 0,
                    'keyword_score': keyword_score_norm,
                    'combined_score': (1 - alpha) * keyword_score_norm
                }
        
        # 4. Ordena por score combinado (do maior para o menor)
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        # 5. Retorna os top_k
        return sorted_results[:top_k]
    
    def search_with_rerank(
        self,
        query: str,
        reranker,
        top_k: int = 3,
        initial_k: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca híbrida + reranking (combinação completa)
        
        Args:
            query: Texto da consulta
            reranker: Instância do Reranker
            top_k: Número final de resultados
            initial_k: Número inicial de resultados antes do rerank
            alpha: Peso da busca semântica
            filters: Filtros
            
        Returns:
            Lista de resultados com reranking aplicado
        """
        # 1. Busca híbrida (mais resultados)
        initial_results = self.search(
            query,
            top_k=initial_k,
            alpha=alpha,
            filters=filters
        )
        
        if not initial_results:
            return []
        
        # 2. Reranking
        reranked = reranker.rerank(query, initial_results, top_k=top_k)
        
        return reranked
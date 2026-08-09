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
    """
    
    def __init__(
        self,
        vector_store,
        embedder,
        chunks_path: str = "./data/processed/chunks.json"
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks_path = Path(chunks_path)
        self.chunks = self._load_chunks()
        self.bm25 = self._build_bm25()
        
        if self.bm25:
            print(f"✅ Busca Híbrida inicializada com {len(self.chunks)} chunks | BM25 ativo")
        else:
            print(f"⚠️ Busca Híbrida inicializada com {len(self.chunks)} chunks | BM25 desativado")
    
    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Carrega os chunks do arquivo JSON"""
        if not self.chunks_path.exists():
            print(f"⚠️ Arquivo de chunks não encontrado: {self.chunks_path}")
            return []
        
        try:
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Erro ao carregar chunks: {e}")
            return []
    
    def _build_bm25(self) -> Optional[BM25Okapi]:
        """Constrói o índice BM25 para busca por palavras-chave"""
        # 🔥 CORREÇÃO: Filtra chunks vazios e tokeniza
        tokenized_docs = []
        valid_chunks = []
        
        for doc in self.chunks:
            content = doc.get('content', '')
            if content and isinstance(content, str):
                tokens = content.lower().split()
                if tokens:
                    tokenized_docs.append(tokens)
                    valid_chunks.append(doc)
        
        # Atualiza chunks para manter alinhamento com BM25
        self.chunks = valid_chunks
        
        # 🔥 CORREÇÃO: Se não houver documentos, retorna None
        if not tokenized_docs:
            print("⚠️ Nenhum documento válido para BM25")
            return None
        
        try:
            return BM25Okapi(tokenized_docs)
        except Exception as e:
            print(f"❌ Erro ao construir BM25: {e}")
            return None
    
    def _keyword_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Busca por palavras-chave usando BM25"""
        # 🔥 CORREÇÃO: Verifica se BM25 está disponível
        if self.bm25 is None:
            return []
        
        if not self.chunks:
            return []
        
        tokenized_query = query.lower().split()
        if not tokenized_query:
            return []
        
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
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
        """Busca híbrida: semântica + palavras-chave"""
        if not query or not query.strip():
            return []
        
        # 1. Busca semântica
        query_emb = self.embedder.embed_text(query)
        semantic_results = self.vector_store.search(
            query_emb.tolist(),
            top_k=top_k * 2,
            filters=filters
        )
        
        # 2. Busca por palavras-chave (BM25)
        keyword_results = self._keyword_search(query, top_k=top_k * 2)
        
        # 3. Combina resultados
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
            keyword_score_norm = min(keyword_score / 10, 1.0)
            
            if chunk_id in combined:
                combined[chunk_id]['keyword_score'] = keyword_score_norm
                combined[chunk_id]['combined_score'] = (
                    alpha * combined[chunk_id]['semantic_score'] +
                    (1 - alpha) * keyword_score_norm
                )
            else:
                combined[chunk_id] = {
                    'id': chunk_id,
                    'content': chunk['content'],
                    'metadata': chunk.get('metadata', {}),
                    'semantic_score': 0,
                    'keyword_score': keyword_score_norm,
                    'combined_score': (1 - alpha) * keyword_score_norm
                }
        
        # 4. Ordena
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
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
        """Busca híbrida + reranking"""
        initial_results = self.search(
            query=query,
            top_k=initial_k,
            alpha=alpha,
            filters=filters
        )
        
        if not initial_results:
            return []
        
        return reranker.rerank(query, initial_results, top_k=top_k)
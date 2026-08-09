"""
Busca Híbrida: Semântica + Palavras-chave (BM25)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi


class HybridSearch:
    """
    Combina busca semântica (embeddings) com busca por palavras-chave (BM25)
    
    Se o arquivo de chunks não existir ou estiver vazio, o BM25 é desativado
    e apenas a busca semântica é utilizada.
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
        self.bm25 = None
        self.chunks = []
        
        # Tenta carregar os chunks
        self._load_chunks()
        
        # Tenta construir o BM25
        self._build_bm25()
        
        if self.bm25 is not None:
            print(f"✅ Busca Híbrida: {len(self.chunks)} chunks | BM25 ATIVO")
        else:
            print(f"⚠️ Busca Híbrida: {len(self.chunks)} chunks | BM25 DESATIVADO")
    
    def _load_chunks(self) -> None:
        """Carrega os chunks do arquivo JSON"""
        if not self.chunks_path.exists():
            print(f"⚠️ Arquivo não encontrado: {self.chunks_path}")
            self.chunks = []
            return
        
        try:
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print("⚠️ chunks.json não é uma lista")
                self.chunks = []
                return
            
            self.chunks = data
            print(f"📚 {len(self.chunks)} chunks carregados")
            
        except json.JSONDecodeError as e:
            print(f"❌ Erro ao decodificar JSON: {e}")
            self.chunks = []
        except Exception as e:
            print(f"❌ Erro ao carregar chunks: {e}")
            self.chunks = []
    
    def _build_bm25(self) -> None:
        """Constrói o índice BM25 (ou desativa se não houver chunks)"""
        if not self.chunks:
            print("⚠️ Nenhum chunk disponível - BM25 desativado")
            self.bm25 = None
            return
        
        # Tokeniza os chunks válidos
        tokenized_docs = []
        valid_chunks = []
        
        for doc in self.chunks:
            content = doc.get('content', '')
            if content and isinstance(content, str):
                tokens = content.lower().split()
                if tokens:
                    tokenized_docs.append(tokens)
                    valid_chunks.append(doc)
        
        # Atualiza chunks para manter alinhamento
        self.chunks = valid_chunks
        
        if not tokenized_docs:
            print("⚠️ Nenhum documento válido para BM25")
            self.bm25 = None
            return
        
        try:
            self.bm25 = BM25Okapi(tokenized_docs)
            print(f"✅ Índice BM25 criado com {len(tokenized_docs)} documentos")
        except Exception as e:
            print(f"❌ Erro ao criar BM25: {e}")
            self.bm25 = None
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Busca por palavras-chave usando BM25"""
        if self.bm25 is None or not self.chunks:
            return []
        
        tokenized_query = query.lower().split()
        if not tokenized_query:
            return []
        
        try:
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
        except Exception as e:
            print(f"⚠️ Erro na busca BM25: {e}")
            return []
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca híbrida: semântica + palavras-chave
        
        Args:
            query: Consulta do usuário
            top_k: Número de resultados
            alpha: Peso da busca semântica (0-1)
            filters: Filtros para a busca semântica
        """
        if not query or not query.strip():
            return []
        
        # 1. Busca semântica (sempre disponível)
        query_emb = self.embedder.embed_text(query)
        semantic_results = self.vector_store.search(
            query_emb.tolist(),
            top_k=top_k * 2,
            filters=filters
        )
        
        # 2. Busca BM25 (se disponível)
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
                'keyword_score': 0.0,
                'combined_score': alpha * r.get('score', 0)
            }
        
        # Adiciona resultados BM25
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
                    'semantic_score': 0.0,
                    'keyword_score': keyword_score_norm,
                    'combined_score': (1 - alpha) * keyword_score_norm
                }
        
        # 4. Ordena e retorna
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
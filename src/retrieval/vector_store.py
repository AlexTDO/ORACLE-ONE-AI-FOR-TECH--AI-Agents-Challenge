"""
Banco vetorial para armazenar e buscar embeddings
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from tqdm import tqdm


class VectorStore:
    """Interface para banco vetorial ChromaDB"""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "techflow_docs",
        embedding_function = None
    ):
        """
        Args:
            persist_directory: Diretório para persistência
            collection_name: Nome da coleção
            embedding_function: Função de embedding (opcional)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Inicializa cliente
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = collection_name
        self.collection = None
        self.embedding_function = embedding_function
        
        # Tenta carregar coleção existente
        self._load_or_create_collection()
    
    def _load_or_create_collection(self):
        """Carrega ou cria a coleção"""
        try:
            # Verifica se coleção existe
            collections = self.client.list_collections()
            if self.collection_name in [c.name for c in collections]:
                self.collection = self.client.get_collection(self.collection_name)
                print(f"📂 Coleção existente carregada: {self.collection_name}")
                print(f"   Documentos: {self.collection.count()}")
            else:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"🆕 Nova coleção criada: {self.collection_name}")
        except Exception as e:
            print(f"❌ Erro ao acessar coleção: {e}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Adiciona chunks ao banco vetorial
        
        Args:
            chunks: Lista de chunks com 'id', 'embedding', 'content', 'metadata'
            batch_size: Tamanho do batch para inserção
        """
        if not chunks:
            print("⚠️  Nenhum chunk para adicionar")
            return
        
        # Prepara dados
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            if "embedding" not in chunk:
                print(f"⚠️  Chunk sem embedding: {chunk.get('id')}")
                continue
            
            chunk_id = chunk.get("id", f"doc_{len(ids)}")
            ids.append(chunk_id)
            
            # Converte embedding para lista se for numpy
            embedding = chunk["embedding"]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            embeddings.append(embedding)
            
            documents.append(chunk["content"])
            
            # Prepara metadados (remove embeddings e content)
            metadata = {
                k: v for k, v in chunk["metadata"].items()
                if isinstance(v, (str, int, float, bool))
            }
            # Adiciona campos úteis
            metadata["chunk_id"] = chunk_id
            metadata["token_count"] = chunk["metadata"].get("token_count", 0)
            metadatas.append(metadata)
        
        # Insere em batches
        total = len(ids)
        print(f"📥 Adicionando {total} chunks ao banco vetorial...")
        
        for i in tqdm(range(0, total, batch_size), desc="Inserindo"):
            batch_end = min(i + batch_size, total)
            
            batch_ids = ids[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_documents = documents[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                print(f"❌ Erro ao inserir batch {i}-{batch_end}: {e}")
        
        print(f"✅ {total} chunks adicionados com sucesso!")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Busca chunks similares
        
        Args:
            query_embedding: Vetor de embedding da consulta
            top_k: Número de resultados
            filters: Filtros de metadados (ex: {"category": "RH"})
            include_metadata: Inclui metadados nos resultados
            
        Returns:
            Lista de resultados com 'id', 'content', 'metadata', 'distance'
        """
        # Prepara filtros
        where_filter = None
        if filters:
            where_filter = filters
        
        # Busca
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Formata resultados
            formatted_results = []
            if results and results['ids']:
                for i in range(len(results['ids'][0])):
                    result = {
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "distance": results['distances'][0][i],
                        "score": 1 - results['distances'][0][i]  # Similaridade
                    }
                    if include_metadata and results['metadatas']:
                        result["metadata"] = results['metadatas'][0][i]
                    
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []
    
    def search_by_text(
        self,
        query: str,
        embedder,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca por texto (usa embedder para gerar embedding)
        
        Args:
            query: Texto da consulta
            embedder: Instância do Embedder
            top_k: Número de resultados
            filters: Filtros de metadados
            
        Returns:
            Lista de resultados
        """
        embedding = embedder.embed_text(query)
        return self.search(embedding.tolist(), top_k, filters)
    
    def search_with_rerank(
        self,
        query: str,
        query_embedding: List[float],
        reranker,
        top_k: int = 3,
        initial_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca com reranking (NOVO MÉTODO)
        
        1. Busca inicial (semântica) - pega mais documentos
        2. Reranking (Cross-Encoder) - reordena por relevância
        3. Retorna apenas os top_k finais
        
        Args:
            query: Texto da consulta (para o reranker)
            query_embedding: Vetor de embedding da consulta
            reranker: Instância do Reranker
            top_k: Número final de resultados
            initial_k: Número inicial de resultados (antes do rerank)
            filters: Filtros de metadados
            
        Returns:
            Lista de resultados rerankeados
        """
        # 1. Busca inicial (pega mais documentos)
        initial_results = self.search(
            query_embedding,
            top_k=initial_k,
            filters=filters,
            include_metadata=True
        )
        
        if not initial_results:
            return []
        
        print(f"🔍 Busca inicial: {len(initial_results)} documentos")
        
        # 2. Reranking
        reranked = reranker.rerank_with_metadata(
            query,
            initial_results,
            top_k=top_k
        )
        
        print(f"🔄 Reranking: top {len(reranked)} documentos")
        
        return reranked
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da coleção"""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": str(self.persist_directory)
        }
    
    def delete_collection(self):
        """Deleta a coleção"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️  Coleção {self.collection_name} deletada")
        except Exception as e:
            print(f"❌ Erro ao deletar coleção: {e}")
    
    def reset(self):
        """Reseta o banco (deleta e recria)"""
        self.delete_collection()
        self._load_or_create_collection()
        print("🔄 Banco resetado com sucesso")


class HybridSearch:
    """Busca híbrida (semântica + palavras-chave)"""
    
    def __init__(self, vector_store: VectorStore, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Busca combinando semântica e palavras-chave
        
        Args:
            query: Texto da consulta
            top_k: Número de resultados
            filters: Filtros de metadados
            alpha: Peso da busca semântica (0-1)
            
        Returns:
            Lista de resultados combinados
        """
        # Busca semântica
        embedding = self.embedder.embed_text(query)
        semantic_results = self.vector_store.search(
            embedding.tolist(),
            top_k=top_k * 2,
            filters=filters
        )
        
        # Busca por palavras-chave (simples)
        keywords = query.lower().split()
        keyword_results = []
        
        # Se não houver resultados semânticos, retorna vazio
        if not semantic_results:
            return []
        
        # Combina resultados
        combined = {}
        for idx, result in enumerate(semantic_results):
            # Score semântico (normalizado)
            semantic_score = result.get('score', 0)
            
            # Score por palavra-chave
            keyword_score = 0
            content_lower = result['content'].lower()
            for keyword in keywords:
                if len(keyword) > 3:  # Ignora palavras muito curtas
                    if keyword in content_lower:
                        keyword_score += 1
            
            keyword_score = min(keyword_score / len(keywords), 1.0) if keywords else 0
            
            # Score final combinado
            combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
            
            result['combined_score'] = combined_score
            result['semantic_score'] = semantic_score
            result['keyword_score'] = keyword_score
            
            combined[result['id']] = result
        
        # Ordena por score combinado
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return sorted_results[:top_k]
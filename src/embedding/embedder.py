"""
Gerador de embeddings para chunks de documentos
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch
import hashlib
import json
from pathlib import Path


class Embedder:
    """Gerador de embeddings usando Sentence Transformers"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model: Optional[SentenceTransformer] = None  # <--- ADICIONADO
    ):
        """
        Args:
            model_name: Nome do modelo Sentence Transformers
            device: 'cpu' ou 'cuda' (auto-detect se None)
            cache_dir: Diretório para cache dos embeddings
            model: Objeto SentenceTransformer pré-carregado (prioritário)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/processed/embeddings_cache")
        
        # Cria diretório de cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Inicializando embedder: {model_name}")
        print(f"📱 Dispositivo: {self.device}")
        
        # ===== CORREÇÃO: Aceita modelo pré-carregado =====
        if model is not None:
            print(f"✅ Usando modelo pré-carregado fornecido externamente.")
            self.model = model
        else:
            # Carrega modelo do zero
            self.model = SentenceTransformer(model_name, device=self.device)
        # =================================================
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Embedder pronto (dimensão: {self.embedding_dim})")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Gera embedding para um texto
        
        Args:
            text: Texto para embedding
            
        Returns:
            Vetor de embedding (numpy array)
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings para múltiplos textos
        
        Args:
            texts: Lista de textos
            
        Returns:
            Matriz de embeddings
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Gera embeddings para chunks e adiciona ao objeto
        
        Args:
            chunks: Lista de chunks com 'content'
            
        Returns:
            Chunks com 'embedding' adicionado
        """
        # Verifica cache
        cached_chunks = self._load_from_cache(chunks)
        if cached_chunks:
            print(f"📦 Carregados {len(cached_chunks)} embeddings do cache")
            return cached_chunks
        
        # Gera embeddings
        texts = [chunk["content"] for chunk in chunks]
        print(f"🔄 Gerando embeddings para {len(texts)} chunks...")
        embeddings = self.embed_batch(texts)
        
        # Adiciona embeddings aos chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
        
        # Salva cache
        self._save_to_cache(chunks)
        
        print(f"✅ Embeddings gerados e salvos no cache")
        return chunks
    
    def _get_cache_key(self, chunks: List[Dict[str, Any]]) -> str:
        """Gera chave de cache baseada no conteúdo dos chunks"""
        # Cria hash do conteúdo
        content_hash = hashlib.md5(
            "".join([c["content"] for c in chunks]).encode()
        ).hexdigest()
        
        return f"embedding_cache_{self.model_name}_{content_hash}.json"
    
    def _load_from_cache(self, chunks: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Carrega embeddings do cache se disponível"""
        cache_key = self._get_cache_key(chunks)
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                
                # Verifica se o número de chunks bate
                if len(cached) == len(chunks):
                    for i, chunk in enumerate(chunks):
                        if cached[i].get("embedding"):
                            chunk["embedding"] = cached[i]["embedding"]
                    return chunks
            except Exception as e:
                print(f"⚠️  Erro ao carregar cache: {e}")
        
        return None
    
    def _save_to_cache(self, chunks: List[Dict[str, Any]]) -> None:
        """Salva embeddings no cache"""
        cache_key = self._get_cache_key(chunks)
        cache_path = self.cache_dir / cache_key
        
        # Prepara para serialização
        serializable = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            if "embedding" in chunk_copy:
                # Converte numpy array para lista
                if isinstance(chunk_copy["embedding"], np.ndarray):
                    chunk_copy["embedding"] = chunk_copy["embedding"].tolist()
            serializable.append(chunk_copy)
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Cache salvo em: {cache_path}")


# Modelos de embedding recomendados (pequenos e eficientes)
EMBEDDING_MODELS = {
    "mini": "all-MiniLM-L6-v2",           # 384 dim, ~80MB
    "small": "all-mpnet-base-v2",         # 768 dim, ~420MB
    "large": "intfloat/e5-large-v2",      # 1024 dim, ~1.3GB
}

# Modelos gratuitos via Ollama (alternativa)
OLLAMA_EMBEDDING_MODELS = {
    "nomic-embed-text": "nomic-embed-text",  # 768 dim
    "all-minilm": "all-minilm",              # 384 dim
}
#!/usr/bin/env python
"""
Script para indexar chunks no banco vetorial
"""

import sys
import json
from pathlib import Path

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore


def main():
    """Indexa chunks no banco vetorial"""
    
    print("=" * 60)
    print("🚀 INDEXANDO DOCUMENTOS NO BANCO VETORIAL")
    print("=" * 60)
    
    # Configuração
    chunks_path = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"
    chroma_path = Path(__file__).parent.parent / "data" / "chroma_db"
    
    # 1. Carrega chunks
    if not chunks_path.exists():
        print(f"❌ Arquivo de chunks não encontrado: {chunks_path}")
        print("Execute primeiro: python scripts/ingest_documents.py")
        return
    
    print(f"📂 Carregando chunks de: {chunks_path}")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"📄 Total de chunks: {len(chunks)}")
    
    # 2. Inicializa embedder
    print("\n🔧 Inicializando embedder...")
    embedder = Embedder(
        model_name="all-MiniLM-L6-v2",
        cache_dir="./data/processed/embeddings_cache"
    )
    
    # 3. Gera embeddings
    print("\n🔄 Gerando embeddings...")
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    
    # 4. Inicializa banco vetorial
    print("\n📊 Inicializando banco vetorial...")
    vector_store = VectorStore(
        persist_directory=str(chroma_path),
        collection_name="techflow_docs"
    )
    
    # 5. Adiciona chunks ao banco
    print("\n📥 Adicionando chunks ao banco...")
    vector_store.add_chunks(chunks_with_embeddings)
    
    # 6. Estatísticas
    stats = vector_store.get_stats()
    print("\n📊 ESTATÍSTICAS DO BANCO")
    print("=" * 40)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 7. Teste rápido
    print("\n🧪 Testando busca...")
    test_query = "Como funciona o processo de onboarding?"
    results = vector_store.search_by_text(
        test_query,
        embedder,
        top_k=3
    )
    
    print(f"\nConsulta: '{test_query}'")
    print("\nResultados:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. ID: {result['id']}")
        print(f"   Similaridade: {result['score']:.3f}")
        print(f"   Categoria: {result['metadata'].get('category', 'N/A')}")
        print(f"   Conteúdo: {result['content'][:150]}...")
    
    print("\n✅ Indexação concluída com sucesso!")


if __name__ == "__main__":
    main()
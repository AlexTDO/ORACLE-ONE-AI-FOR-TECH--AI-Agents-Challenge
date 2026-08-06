"""
Testes para verificar o banco vetorial
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore


def test_vector_store():
    """Testa funcionalidade do banco vetorial"""
    
    print("=" * 60)
    print("🧪 TESTANDO BANCO VETORIAL")
    print("=" * 60)
    
    # 1. Inicializa
    embedder = Embedder()
    vector_store = VectorStore()
    
    # 2. Estatísticas
    stats = vector_store.get_stats()
    print(f"\n📊 Estatísticas:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 3. Testa busca
    test_queries = [
        "Como solicito reembolso?",
        "Quais são os benefícios?",
        "Como funciona a API?",
        "Qual a política de privacidade?",
        "Quantos dias de férias?",
    ]
    
    print("\n🔍 Testando buscas:")
    for query in test_queries:
        print(f"\nPergunta: '{query}'")
        results = vector_store.search_by_text(query, embedder, top_k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Categoria: {result['metadata'].get('category', 'N/A')}")
                print(f"     Preview: {result['content'][:100]}...")
        else:
            print("  ❌ Nenhum resultado encontrado")


if __name__ == "__main__":
    test_vector_store()
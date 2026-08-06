"""
Teste completo do sistema de busca com e sem reranking
"""

import sys
import json
from pathlib import Path
import time

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import Reranker


def test_busca_sem_rerank(vector_store, embedder, query, top_k=5):
    """Testa busca sem reranking"""
    print("\n🔍 BUSCA SEM RERANKING")
    print("-" * 40)
    
    start_time = time.time()
    query_emb = embedder.embed_text(query)
    results = vector_store.search(query_emb.tolist(), top_k=top_k)
    elapsed = time.time() - start_time
    
    print(f"⏱️ Tempo: {elapsed:.2f}s")
    print(f"📄 Resultados: {len(results)}")
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Score: {r['score']:.4f}")
        print(f"   Categoria: {r['metadata'].get('category', 'N/A')}")
        print(f"   Arquivo: {r['metadata'].get('filename', 'N/A')}")
        print(f"   Preview: {r['content'][:120]}...")
    
    return results


def test_busca_com_rerank(vector_store, embedder, reranker, query, top_k=3):
    """Testa busca com reranking"""
    print("\n🔄 BUSCA COM RERANKING")
    print("-" * 40)
    
    start_time = time.time()
    query_emb = embedder.embed_text(query)
    results = vector_store.search_with_rerank(
        query=query,
        query_embedding=query_emb.tolist(),
        reranker=reranker,
        top_k=top_k,
        initial_k=10
    )
    elapsed = time.time() - start_time
    
    print(f"⏱️ Tempo: {elapsed:.2f}s")
    print(f"📄 Resultados: {len(results)}")
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Score Rerank: {r['score']:.4f}")
        print(f"   Score Original: {r.get('original_score', 0):.4f}")
        print(f"   Categoria: {r['metadata'].get('category', 'N/A')}")
        print(f"   Arquivo: {r['metadata'].get('filename', 'N/A')}")
        print(f"   Preview: {r['content'][:120]}...")
    
    return results


def comparar_resultados(sem_rerank, com_rerank):
    """Compara os resultados com e sem reranking"""
    print("\n📊 COMPARAÇÃO")
    print("=" * 40)
    
    if not sem_rerank or not com_rerank:
        print("⚠️ Sem dados para comparar")
        return
    
    # Melhor score sem rerank
    best_sem = sem_rerank[0]['score'] if sem_rerank else 0
    
    # Melhor score com rerank
    best_com = com_rerank[0]['score'] if com_rerank else 0
    
    # Melhoria
    improvement = ((best_com - best_sem) / best_sem * 100) if best_sem > 0 else 0
    
    print(f"📈 Melhor score sem rerank: {best_sem:.4f}")
    print(f"📈 Melhor score com rerank: {best_com:.4f}")
    print(f"📈 Melhoria: +{improvement:.1f}%")
    
    # Verifica se a categoria mudou
    cat_sem = sem_rerank[0]['metadata'].get('category', 'N/A') if sem_rerank else 'N/A'
    cat_com = com_rerank[0]['metadata'].get('category', 'N/A') if com_rerank else 'N/A'
    
    if cat_sem != cat_com:
        print(f"\n🔄 Categoria mudou:")
        print(f"   Sem rerank: {cat_sem}")
        print(f"   Com rerank: {cat_com}")
    
    # Mostra os top resultados lado a lado
    print("\n📋 TOP 3 RESULTADOS:")
    print("-" * 60)
    
    print("\nSem Rerank:")
    for i, r in enumerate(sem_rerank[:3], 1):
        print(f"  {i}. {r['score']:.4f} - {r['metadata'].get('category', 'N/A')}")
    
    print("\nCom Rerank:")
    for i, r in enumerate(com_rerank[:3], 1):
        print(f"  {i}. {r['score']:.4f} - {r['metadata'].get('category', 'N/A')}")


def main():
    """Executa todos os testes"""
    
    print("=" * 70)
    print("🧪 TESTE COMPLETO - RAG COM RERANKING")
    print("=" * 70)
    
    # 1. Inicializa componentes
    print("\n🔧 Inicializando componentes...")
    embedder = Embedder()
    vector_store = VectorStore()
    reranker = Reranker()
    
    # Verifica se há documentos
    stats = vector_store.get_stats()
    if stats['document_count'] == 0:
        print("❌ Banco vazio! Execute primeiro: python scripts/index_documents.py")
        return
    
    print(f"\n📊 Banco: {stats['document_count']} documentos")
    
    # 2. Perguntas de teste
    test_queries = [
        "Quais são os benefícios oferecidos pela empresa?",
        "Como solicito reembolso de despesas?",
        "Como funciona o processo de onboarding?",
        "Qual a política de privacidade de dados?",
        "Quantos dias de férias eu tenho direito?",
        "Como funciona a API do FlowManager?",
        "Quais são os objetivos da empresa para 2025?",
    ]
    
    # 3. Executa testes
    resultados = []
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"📝 TESTE {i}/{len(test_queries)}: '{query}'")
        print("=" * 70)
        
        # Busca sem rerank
        sem_rerank = test_busca_sem_rerank(vector_store, embedder, query)
        
        # Busca com rerank
        com_rerank = test_busca_com_rerank(vector_store, embedder, reranker, query)
        
        # Compara
        comparar_resultados(sem_rerank, com_rerank)
        
        resultados.append({
            "query": query,
            "sem_rerank": sem_rerank,
            "com_rerank": com_rerank
        })
        
        # Pausa entre testes
        if i < len(test_queries):
            print("\n⏳ Pressione Enter para próximo teste...")
            input()
    
    # 4. Resumo final
    print("\n" + "=" * 70)
    print("📊 RESUMO FINAL")
    print("=" * 70)
    
    melhorias = []
    for r in resultados:
        if r["sem_rerank"] and r["com_rerank"]:
            best_sem = r["sem_rerank"][0]['score']
            best_com = r["com_rerank"][0]['score']
            improvement = ((best_com - best_sem) / best_sem * 100) if best_sem > 0 else 0
            melhorias.append(improvement)
            
            print(f"\n'{r['query'][:40]}...'")
            print(f"  Sem: {best_sem:.4f} → Com: {best_com:.4f} (+{improvement:.1f}%)")
    
    if melhorias:
        avg_improvement = sum(melhorias) / len(melhorias)
        print(f"\n📈 Média de melhoria: +{avg_improvement:.1f}%")
    
    print("\n✅ Testes concluídos!")


if __name__ == "__main__":
    main()
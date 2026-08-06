"""
Teste do agente RAG com LangGraph
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.agent.graph import RAGGraph


def main():
    """Testa o agente com LangGraph"""
    
    print("=" * 70)
    print("🧪 TESTE DO AGENTE RAG COM LANGGRAPH")
    print("=" * 70)
    
    # Inicializa o agente
    print("\n🚀 Inicializando agente com LangGraph...")
    agent = RAGGraph(
        llm_provider="ollama",
        llm_model="gemma3:1b",
        top_k=3
    )
    
    # Perguntas de teste
    queries = [
        "Quem fundou a TechFlow Solutions?",
        "Quais são os benefícios oferecidos pela empresa?",
        "Como solicito reembolso de despesas?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"📝 TESTE {i}/{len(queries)}")
        print(f"{'='*70}")
        
        result = agent.ask(query)
        
        if not result['success']:
            print(f"❌ Erro: {result.get('erros', [])}")
    
    print("\n" + "=" * 70)
    print("✅ TESTES CONCLUÍDOS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
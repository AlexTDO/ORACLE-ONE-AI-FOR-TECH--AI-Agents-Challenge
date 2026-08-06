"""
Teste do Agente RAG via linha de comando
Útil para validar o sistema antes de rodar a interface Streamlit
"""

import sys
import os
from pathlib import Path

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import Reranker
from src.generation.llm import LLMFactory, RAGAgent


def test_agent():
    """Testa o agente com perguntas pré-definidas"""
    
    print("=" * 70)
    print("🧪 TESTE DO AGENTE RAG - LINHA DE COMANDO")
    print("=" * 70)
    
    # ==================== CONFIGURAÇÃO ====================
    
    # Escolha o provedor e modelo
    # Opções: "ollama", "gemini", "openai", "claude"
    PROVIDER = "ollama"
    MODEL_NAME = "gemma3:1b"  # ou "phi3:mini", "llama3.2:3b"
    TEMPERATURE = 0.3
    TOP_K = 3
    
    print(f"\n📋 Configuração:")
    print(f"   Provedor: {PROVIDER}")
    print(f"   Modelo: {MODEL_NAME}")
    print(f"   Temperatura: {TEMPERATURE}")
    print(f"   Documentos por resposta: {TOP_K}")
    
    # ==================== INICIALIZAÇÃO ====================
    
    print("\n🔧 Inicializando componentes...")
    
    try:
        embedder = Embedder()
        print("   ✅ Embedder OK")
        
        vector_store = VectorStore()
        stats = vector_store.get_stats()
        print(f"   ✅ VectorStore OK ({stats['document_count']} chunks)")
        
        reranker = Reranker()
        print("   ✅ Reranker OK")
        
        # Cria LLM
        llm = LLMFactory.create(
            provider=PROVIDER,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=500
        )
        print(f"   ✅ LLM OK ({MODEL_NAME})")
        
        # Cria agente
        agent = RAGAgent(
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            llm=llm,
            top_k=TOP_K
        )
        print("   ✅ Agente criado com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro ao inicializar: {e}")
        return
    
    # ==================== PERGUNTAS DE TESTE ====================
    
    test_queries = [
        "Quais são os benefícios oferecidos pela empresa?",
        "Como solicito reembolso de despesas?",
        "Como funciona o processo de onboarding?",
        "Qual a política de privacidade de dados?",
        "Quantos dias de férias eu tenho direito?",
        "Como funciona a API do FlowManager?",
        "Quais são os objetivos da empresa para 2025?",
    ]
    
    # ==================== EXECUTA TESTES ====================
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"📝 TESTE {i}/{len(test_queries)}: '{query}'")
        print("=" * 70)
        
        try:
            print("\n🤖 Gerando resposta...")
            result = agent.ask(query)
            
            if result['success']:
                print("\n📝 RESPOSTA:")
                print("-" * 50)
                print(result['response'])
                print("-" * 50)
                
                print(f"\n📊 Informações:")
                print(f"   Modelo: {result.get('model', 'N/A')}")
                print(f"   Provedor: {result.get('provider', 'N/A')}")
                print(f"   Tokens usados: {result.get('tokens_used', 0)}")
                
                print(f"\n📄 Fontes consultadas ({len(result.get('sources', []))}):")
                for source in result.get('sources', []):
                    score = source.get('score', 0)
                    print(f"   - {source['filename']} ({source.get('category', 'N/A')}) [score: {score:.3f}]")
            else:
                print(f"\n❌ {result['response']}")
                
        except Exception as e:
            print(f"\n❌ Erro ao processar pergunta: {e}")
        
        # Pausa entre perguntas
        if i < len(test_queries):
            print("\n⏳ Pressione Enter para próxima pergunta...")
            input()
    
    # ==================== RESUMO FINAL ====================
    
    print("\n" + "=" * 70)
    print("✅ TESTES CONCLUÍDOS!")
    print("=" * 70)
    
    print("\n💡 Dicas:")
    print("   - Para testar outro modelo, mude MODEL_NAME")
    print("   - Para testar API, mude PROVIDER e configure .env")
    print("   - Para interface web: streamlit run src/interface/app.py")


if __name__ == "__main__":
    test_agent()
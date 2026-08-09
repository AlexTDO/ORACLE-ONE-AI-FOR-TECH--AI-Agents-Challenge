"""
Teste do Agente RAG via linha de comando
Útil para validar o sistema antes de rodar a interface Streamlit
Suporta: Ollama, Gemini, OpenAI, Claude, OpenRouter, Groq, DeepSeek
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import Reranker
from src.generation.llm import LLMFactory, RAGAgent


# ==================== CONFIGURAÇÃO ====================

# Carrega variáveis de ambiente
load_dotenv()

# Provedores disponíveis
PROVIDERS = {
    "ollama": {
        "name": "🟢 Ollama (Local)",
        "default_model": "gemma3:1b",
        "models": ["gemma3:1b", "phi3:mini", "llama3.2:3b", "mistral:7b"],
        "needs_api_key": False
    },
    "gemini": {
        "name": "🔵 Google Gemini",
        "default_model": "gemini-2.5-pro",
        "models": ["gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"],
        "needs_api_key": True,
        "env_var": "GEMINI_API_KEY"
    },
    "openai": {
        "name": "🟢 OpenAI",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "needs_api_key": True,
        "env_var": "OPENAI_API_KEY"
    },
    "claude": {
        "name": "🟠 Anthropic Claude",
        "default_model": "claude-3-haiku-20240307",
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "needs_api_key": True,
        "env_var": "ANTHROPIC_API_KEY"
    },
    "openrouter": {
        "name": "🟣 OpenRouter (Gratuito)",
        "default_model": "nvidia/nemotron-3-super-120b-a12b:free",
        "models": [
            "nvidia/nemotron-3-super-120b-a12b:free",
            "google/gemma-3-27b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free"
        ],
        "needs_api_key": True,
        "env_var": "OPENROUTER_API_KEY"
    },
    "groq": {
        "name": "🟡 Groq (Rápido)",
        "default_model": "mixtral-8x7b-32768",
        "models": ["mixtral-8x7b-32768", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"],
        "needs_api_key": True,
        "env_var": "GROQ_API_KEY"
    },
    "deepseek": {
        "name": "🔷 DeepSeek",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "needs_api_key": True,
        "env_var": "DEEPSEEK_API_KEY"
    }
}

# Perguntas de teste padrão
DEFAULT_QUERIES = [
    "Quem fundou a TechFlow Solutions?",
    "Qual foi o orçamento inicial da TechFlow Solutions?",
    "Em que ano a TechFlow foi fundada?",
    "Onde fica a sede da TechFlow Solutions?",
    "Quais são os benefícios oferecidos pela empresa?",
    "Como solicito reembolso de despesas?",
    "Como funciona o processo de onboarding?",
    "Qual a política de privacidade de dados?",
    "Quantos dias de férias eu tenho direito?",
    "Como funciona a API do FlowManager?",
    "Quais são os objetivos da empresa para 2025?",
]


def select_provider():
    """Interface interativa para selecionar o provedor"""
    print("\n📋 PROVEDORES DISPONÍVEIS:")
    print("-" * 50)
    provider_list = list(PROVIDERS.keys())
    for i, (key, value) in enumerate(PROVIDERS.items(), 1):
        api_status = "🔑" if value["needs_api_key"] else "🆓"
        print(f"  {i}. {value['name']} {api_status}")
    
    while True:
        try:
            choice = input("\n👉 Escolha o provedor (1-{}): ".format(len(provider_list)))
            idx = int(choice) - 1
            if 0 <= idx < len(provider_list):
                return provider_list[idx]
        except ValueError:
            pass
        print("❌ Opção inválida. Tente novamente.")


def select_model(provider_key):
    """Interface interativa para selecionar o modelo"""
    provider = PROVIDERS[provider_key]
    
    print(f"\n📋 MODELOS DISPONÍVEIS PARA {provider['name']}:")
    print("-" * 50)
    for i, model in enumerate(provider["models"], 1):
        default_marker = " (padrão)" if model == provider["default_model"] else ""
        print(f"  {i}. {model}{default_marker}")
    
    if len(provider["models"]) > 1:
        custom_option = len(provider["models"]) + 1
        print(f"  {custom_option}. 🔧 Digitar modelo personalizado")
    
    while True:
        try:
            choice = input(f"\n👉 Escolha o modelo (1-{len(provider['models'])}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(provider["models"]):
                return provider["models"][idx]
            elif idx == len(provider["models"]) and len(provider["models"]) > 1:
                custom = input("Digite o nome do modelo: ").strip()
                if custom:
                    return custom
        except ValueError:
            pass
        print("❌ Opção inválida. Tente novamente.")


def select_queries():
    """Interface interativa para selecionar perguntas"""
    print("\n📋 PERGUNTAS DE TESTE:")
    print("-" * 50)
    for i, q in enumerate(DEFAULT_QUERIES, 1):
        print(f"  {i}. {q}")
    
    print(f"\n  {len(DEFAULT_QUERIES) + 1}. 🔧 Todas as perguntas")
    print(f"  {len(DEFAULT_QUERIES) + 2}. 📝 Digitar pergunta personalizada")
    
    while True:
        try:
            choice = input(f"\n👉 Escolha (1-{len(DEFAULT_QUERIES) + 2}): ")
            idx = int(choice) - 1
            
            if 0 <= idx < len(DEFAULT_QUERIES):
                return [DEFAULT_QUERIES[idx]]
            elif idx == len(DEFAULT_QUERIES):
                return DEFAULT_QUERIES
            elif idx == len(DEFAULT_QUERIES) + 1:
                custom = input("Digite sua pergunta: ").strip()
                if custom:
                    return [custom]
        except ValueError:
            pass
        print("❌ Opção inválida. Tente novamente.")


def test_agent():
    """Testa o agente com perguntas pré-definidas"""
    
    print("=" * 70)
    print("🚀 TESTE DO AGENTE RAG - LINHA DE COMANDO")
    print("=" * 70)
    print("🧠 TechFlow Solutions - Agente Corporativo IA")
    print("=" * 70)
    
    # ==================== SELECIONAR CONFIGURAÇÃO ====================
    
    # Escolher provedor
    provider_key = select_provider()
    provider = PROVIDERS[provider_key]
    
    # Verificar API Key (se necessário)
    if provider["needs_api_key"]:
        api_key = os.getenv(provider["env_var"])
        if not api_key:
            print(f"\n⚠️  ATENÇÃO: {provider['env_var']} não encontrada no .env!")
            print(f"   Configure a chave no arquivo .env ou use Ollama (local).")
            if input("\n   Continuar mesmo assim? (s/N): ").lower() != 's':
                return
    
    # Escolher modelo
    model_name = select_model(provider_key)
    
    # Configurações
    print("\n⚙️ CONFIGURAÇÕES:")
    print("-" * 50)
    
    try:
        temperature = float(input("   🌡️ Temperatura (0.0-1.0, padrão 0.3): ") or "0.3")
        temperature = max(0.0, min(1.0, temperature))
    except ValueError:
        temperature = 0.3
    
    try:
        top_k = int(input("   📄 Número de documentos (1-10, padrão 5): ") or "5")
        top_k = max(1, min(10, top_k))
    except ValueError:
        top_k = 5
    
    # Perguntar se quer usar reranker
    use_reranker = input("   🎯 Usar reranker? (s/N): ").lower() == 's'
    
    # Escolher perguntas
    queries = select_queries()
    
    # ==================== RESUMO DA CONFIGURAÇÃO ====================
    
    print("\n" + "=" * 70)
    print("📋 CONFIGURAÇÃO FINAL:")
    print("=" * 70)
    print(f"   Provedor: {provider['name']}")
    print(f"   Modelo: {model_name}")
    print(f"   Temperatura: {temperature}")
    print(f"   Documentos: {top_k}")
    print(f"   Reranker: {'✅ Ativado' if use_reranker else '❌ Desativado'}")
    print(f"   Perguntas: {len(queries)}")
    if provider["needs_api_key"]:
        api_key = os.getenv(provider["env_var"])
        print(f"   API Key: {'✅ Configurada' if api_key else '❌ Não configurada'}")
    print("=" * 70)
    
    confirm = input("\n👉 Confirmar e executar? (Enter para continuar, N para cancelar): ")
    if confirm.lower() == 'n':
        print("❌ Teste cancelado.")
        return
    
    # ==================== INICIALIZAÇÃO ====================
    
    print("\n🔧 Inicializando componentes...")
    
    try:
        embedder = Embedder()
        print("   ✅ Embedder OK")
        
        vector_store = VectorStore()
        stats = vector_store.get_stats()
        print(f"   ✅ VectorStore OK ({stats['document_count']} chunks)")
        
        # Cria reranker apenas se for usar
        reranker = Reranker() if use_reranker else None
        if reranker:
            print("   ✅ Reranker OK")
        else:
            print("   ⏭️ Reranker desativado")
        
        # Cria LLM
        llm = LLMFactory.create(
            provider=provider_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=800
        )
        print(f"   ✅ LLM OK ({model_name})")
        
        # Cria agente
        agent = RAGAgent(
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,  # Pode ser None
            llm=llm,
            top_k=top_k
        )
        print("   ✅ Agente criado com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro ao inicializar: {e}")
        return
    
    # ==================== EXECUTA TESTES ====================
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print("\n" + "=" * 70)
        print(f"📝 TESTE {i}/{len(queries)}")
        print(f"   Pergunta: {query}")
        print("=" * 70)
        
        try:
            print("🤖 Gerando resposta...")
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
                
                results.append({"query": query, "status": "✅ OK", "tokens": result.get('tokens_used', 0)})
            else:
                print(f"\n❌ {result['response']}")
                results.append({"query": query, "status": "❌ FALHOU", "tokens": 0})
                
        except Exception as e:
            print(f"\n❌ Erro ao processar: {e}")
            results.append({"query": query, "status": f"❌ ERRO: {e}", "tokens": 0})
        
        # Pausa entre perguntas
        if i < len(queries):
            print("\n⏳ Pressione Enter para próxima pergunta...")
            input()
    
    # ==================== RESUMO FINAL ====================
    
    print("\n" + "=" * 70)
    print("📊 RESUMO DOS TESTES")
    print("=" * 70)
    
    ok = sum(1 for r in results if "✅" in r['status'])
    total = len(results)
    
    print(f"✅ Sucesso: {ok}/{total}")
    print(f"❌ Falhas: {total - ok}/{total}")
    
    for i, r in enumerate(results, 1):
        status_icon = "✅" if "OK" in r['status'] else "❌"
        print(f"  {i}. {status_icon} {r['status']} (Tokens: {r.get('tokens', 0)})")
    
    print("\n" + "=" * 70)
    print("✅ TESTES CONCLUÍDOS!")
    print("=" * 70)
    
    print("\n💡 Dicas:")
    print("   - Para testar outro modelo, execute novamente")
    print("   - Para API, configure as chaves no arquivo .env")
    print("   - Para interface web: streamlit run src/interface/app.py")
    print("   - Modelos gratuitos recomendados:")
    print("     * OpenRouter: nvidia/nemotron-3-super-120b-a12b:free")
    print("     * Groq: mixtral-8x7b-32768")
    print("     * DeepSeek: deepseek-chat")


if __name__ == "__main__":
    test_agent()
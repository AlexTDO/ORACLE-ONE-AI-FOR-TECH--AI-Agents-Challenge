"""
Interface Streamlit para o Agente RAG Corporativo - TechFlow Solutions
Suporte universal para múltiplos LLMs (Ollama, Gemini, OpenAI, Claude, OpenRouter, Groq, DeepSeek)
"""

import sys
import os
import time
from pathlib import Path

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import Reranker
from src.generation.llm import LLMFactory, RAGAgent


# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="TechFlow Solutions - Agente Corporativo IA",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS PERSONALIZADO ====================
st.markdown("""
<style>
    /* ===== HEADER ===== */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0A2463;
        text-align: center;
        padding: 0.5rem 0 0 0;
        letter-spacing: -0.5px;
    }
    .main-header .highlight {
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6B7C;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .divider {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #0A2463, #1E88E5, #0A2463);
        margin: 0.5rem 0 2rem 0;
        border-radius: 10px;
    }
    
    /* ===== BADGES ===== */
    .provider-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 700;
        color: white;
        margin-left: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .provider-ollama { background: linear-gradient(135deg, #10B981, #059669); }
    .provider-gemini { background: linear-gradient(135deg, #8B5CF6, #7C3AED); }
    .provider-openai { background: linear-gradient(135deg, #22C55E, #16A34A); }
    .provider-claude { background: linear-gradient(135deg, #D97706, #B45309); }
    .provider-openrouter { background: linear-gradient(135deg, #EC4899, #DB2777); }
    .provider-groq { background: linear-gradient(135deg, #F59E0B, #D97706); }
    .provider-deepseek { background: linear-gradient(135deg, #06B6D4, #0891B2); }
    
    /* ===== SOURCE CARDS ===== */
    .source-card {
        background-color: #F8FAFC;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s;
    }
    .source-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        transform: translateX(4px);
    }
    .source-card-ollama { border-left-color: #10B981; }
    .source-card-gemini { border-left-color: #8B5CF6; }
    .source-card-openai { border-left-color: #22C55E; }
    .source-card-claude { border-left-color: #D97706; }
    .source-card-openrouter { border-left-color: #EC4899; }
    .source-card-groq { border-left-color: #F59E0B; }
    .source-card-deepseek { border-left-color: #06B6D4; }
    
    .score-high { color: #10B981; font-weight: 600; }
    .score-medium { color: #F59E0B; font-weight: 600; }
    .score-low { color: #EF4444; font-weight: 600; }
    
    /* ===== CHAT ===== */
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
    }
    
    /* ===== SIDEBAR ===== */
    .sidebar-logo {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .sidebar-logo h2 {
        color: #0A2463;
        font-weight: 700;
        margin: 0;
    }
    .sidebar-logo p {
        color: #5A6B7C;
        font-size: 0.8rem;
        margin: 0;
    }
    .sidebar-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #D1D5DB, transparent);
        margin: 1rem 0;
    }
    
    /* ===== STATUS ===== */
    .status-online {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #10B981;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: #9CA3AF;
        font-size: 0.75rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 2rem;
    }
    .footer strong {
        color: #0A2463;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    # Logo e Título
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🚀 TechFlow</h2>
        <p>Solutions</p>
        <hr class="sidebar-divider">
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Sobre o Agente")
    st.markdown("""
    Agente de IA corporativo com **RAG** (Retrieval-Augmented Generation) para responder perguntas com base em documentos internos.
    
    **Funcionalidades:**
    - 🔍 Busca semântica híbrida
    - 🎯 Reranking de precisão
    - 🧠 Geração com IA
    - 📄 Citação de fontes
    - 🌐 Múltiplos provedores
    """)
    
    st.markdown("---")
    
    st.markdown("### 📁 Documentos Indexados")
    if 'doc_count' in st.session_state:
        st.metric("📄 Total de Chunks", st.session_state.doc_count, delta="Indexados")
    else:
        st.info("Carregando...")
    
    st.markdown("---")
    
    st.markdown("### 🏷️ Filtros")
    categories = ["Todas", "Rh", "Financeiro", "Legal", "Operacional", "Estrategico"]
    selected_category = st.selectbox(
        "Categoria:",
        categories,
        help="Filtra a busca apenas nos documentos da categoria selecionada"
    )
    
    st.markdown("---")
    
    st.markdown("### 🧠 Modelo de IA")
    
    # Provedores disponíveis
    providers = [
        "ollama", 
        "gemini", 
        "openai", 
        "claude",
        "openrouter",
        "groq",
        "deepseek"
    ]
    
    provider = st.selectbox(
        "Provedor:",
        providers,
        index=4,
        help="Escolha qual LLM usar. Configure as chaves no arquivo .env"
    )
    
    # Modelos padrão por provedor
    default_models = {
        "ollama": "gemma3:1b",
        "gemini": "gemini-2.5-pro",
        "openai": "gpt-3.5-turbo",
        "claude": "claude-3-haiku-20240307",
        "openrouter": "nvidia/nemotron-3-super-120b-a12b:free",
        "groq": "mixtral-8x7b-32768",
        "deepseek": "deepseek-chat"
    }
    
    # Sugestões de modelos gratuitos por provedor
    free_models = {
        "openrouter": [
            "nvidia/nemotron-3-super-120b-a12b:free",
            "google/gemma-3-27b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free"
        ],
        "groq": [
            "mixtral-8x7b-32768",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "gemma2-9b-it"
        ],
        "deepseek": [
            "deepseek-chat",
            "deepseek-reasoner"
        ]
    }
    
    model_name = st.text_input(
        "Modelo:",
        value=default_models.get(provider, "gemma3:1b"),
        help="Nome do modelo. Para OpenRouter, use o slug completo com :free"
    )
    
    # Botão para sugerir modelos gratuitos
    if provider in free_models and st.button("💡 Ver modelos gratuitos", use_container_width=True):
        st.write("**Modelos gratuitos disponíveis:**")
        for m in free_models[provider]:
            st.write(f"- `{m}`")
    
    # Botão para ver modelos Ollama
    if provider == "ollama" and st.button("🔍 Ver modelos disponíveis", use_container_width=True):
        try:
            import ollama
            models = ollama.list()
            st.write("**Modelos disponíveis:**")
            for m in models.get('models', []):
                st.write(f"- `{m['model']}`")
        except Exception as e:
            st.warning(f"⚠️ Erro ao listar modelos: {e}")
    
    # Aviso sobre API Keys
    if provider != "ollama":
        env_var_map = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "groq": "GROQ_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY"
        }
        env_var = env_var_map.get(provider, f"{provider.upper()}_API_KEY")
        
        st.info(f"📌 Configure `{env_var}` no arquivo `.env`")
        st.code(f"# .env\n{env_var}=sua_chave_aqui", language="bash")
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Configurações")
    
    temperature = st.slider(
        "🌡️ Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Valores baixos = mais precisos, valores altos = mais criativos"
    )
    
    top_k = st.slider(
        "📄 Documentos",
        min_value=1,
        max_value=5,
        value=3,
        help="Quantos documentos usar para gerar a resposta"
    )
    
    st.markdown("---")
    
    # Status do modelo ativo
    if 'current_model' in st.session_state:
        provider_label = st.session_state.current_provider
        provider_class = f"provider-{provider_label}"
        st.markdown(f"""
        <div style="background: #F0F4F8; padding: 0.8rem 1rem; border-radius: 12px; border: 1px solid #E5E7EB;">
            <span style="font-size: 0.8rem; color: #6B7280;">🔮 Modelo ativo</span>
            <br>
            <span style="font-weight: 700; font-size: 0.95rem; color: #0A2463;">{st.session_state.current_model}</span>
            <span class="provider-badge {provider_class}">{provider_label}</span>
            <br>
            <span style="font-size: 0.7rem; color: #9CA3AF;">
                <span class="status-online"></span> Online
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🔄 Resetar Conversa", use_container_width=True):
        st.session_state.messages = []
        time.sleep(0.1)
        st.rerun()
    
    st.markdown("---")
    st.caption("🚀 TechFlow Solutions © 2025")

# ==================== INICIALIZAÇÃO ====================

@st.cache_resource(ttl=3600)  # <--- ADICIONADO TTL PARA LIMPAR CACHE NA NUVEM
def init_components(provider, model_name, temperature, top_k):
    """Inicializa os componentes do RAG com o provedor e modelo selecionados"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        embedder = Embedder()
        vector_store = VectorStore()
        reranker = Reranker()
        
        try:
            llm = LLMFactory.create(
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=800
            )
        except ValueError as e:
            st.error(f"❌ Erro ao criar LLM: {e}")
            st.stop()
        
        agent = RAGAgent(
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            llm=llm,
            top_k=top_k
        )
        
        stats = vector_store.get_stats()
        st.session_state.doc_count = stats['document_count']
        st.session_state.current_model = model_name
        st.session_state.current_provider = provider
        
        return agent
        
    except Exception as e:
        st.error(f"❌ Erro ao inicializar: {e}")
        st.stop()


# ==================== ESTADO DA SESSÃO ====================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'agent' not in st.session_state:
    with st.spinner("🔄 Carregando agente..."):
        st.session_state.agent = init_components(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            top_k=top_k
        )

# ==================== GERENCIAMENTO DE TROCA DE MODELO ====================
# Não usamos st.rerun() aqui. A troca de modelo será aplicada na próxima pergunta
if st.session_state.get('last_provider') != provider or \
   st.session_state.get('last_model') != model_name:
    
    st.session_state.last_provider = provider
    st.session_state.last_model = model_name
    st.session_state.messages = [] # Limpa o chat visualmente
    
    # Apenas invalidamos o cache antigo. O Streamlit vai recriar no próximo uso.
    if 'agent' in st.session_state:
        del st.session_state.agent 

# Aplica configurações ao agente existente (se disponível)
if 'agent' in st.session_state:
    st.session_state.agent.top_k = top_k
    if hasattr(st.session_state.agent, 'llm'):
        st.session_state.agent.llm.temperature = temperature

# ==================== HEADER ====================

st.markdown("""
<div style="text-align: center; padding: 0.5rem 0 0 0;">
    <h1 class="main-header">
        🚀 TechFlow <span class="highlight">Solutions</span>
    </h1>
    <p class="sub-header">
        🤖 Agente Corporativo com Inteligência Artificial
    </p>
    <hr class="divider">
</div>
""", unsafe_allow_html=True)

# ==================== CHAT ====================

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Olá! Sou o assistente virtual da TechFlow Solutions!**  
        Estou aqui para ajudar você a encontrar informações sobre a empresa.

        **Posso ajudar com:**
        - 📋 **RH**: Benefícios, férias, onboarding
        - 💰 **Financeiro**: Reembolso, orçamentos
        - ⚖️ **Legal**: Políticas, privacidade, LGPD
        - 🔧 **Operacional**: API, processos técnicos
        - 🎯 **Estratégico**: OKRs, roadmap, visão

        **Exemplos de perguntas:**
        - *"Quais são os benefícios oferecidos?"*
        - *"Como solicito reembolso?"*
        - *"Quem fundou a empresa?"*

        💡 **Modelo ativo:** `{}` ({})
        
        Digite sua pergunta abaixo! 👇
        """.format(
            st.session_state.get('current_model', 'gemma3:1b'),
            st.session_state.get('current_provider', 'ollama')
        ))

# Histórico
for message_data in st.session_state.messages:
    with st.chat_message(message_data["role"]):
        st.markdown(message_data["content"])
        
        if "sources" in message_data and message_data["sources"]:
            with st.expander("📄 Ver fontes"):
                for source in message_data["sources"]:
                    score = source.get('score', 0)
                    score_class = "score-high" if score > 0.7 else "score-medium" if score > 0.4 else "score-low"
                    category = source.get('category', 'Geral')
                    
                    st.markdown(f"""
                    <div class="source-card source-card-{category.lower()}">
                        <strong>📎 {source['filename']}</strong>
                        <br>
                        <span style="color: #6B7280; font-size: 0.85rem;">Categoria: {category}</span>
                        <br>
                        <span class="{score_class}">Relevância: {score:.3f}</span>
                        <br>
                        <span style="color: #9CA3AF; font-size: 0.8rem;">{source.get('preview', '')}</span>
                    </div>
                    """, unsafe_allow_html=True)

# ==================== INPUT ====================

if prompt := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Consultando documentos..."):
            try:
                # Verifica se o agente foi deletado (troca de modelo) e recria sob demanda
                if 'agent' not in st.session_state:
                    with st.spinner("🔄 Aplicando novo modelo..."):
                        st.session_state.agent = init_components(
                            provider=provider,
                            model_name=model_name,
                            temperature=temperature,
                            top_k=top_k
                        )
                
                filters = None
                if selected_category != "Todas":
                    filters = {"category": selected_category}
                
                result = st.session_state.agent.ask(prompt, filters=filters)
                
                if result['success']:
                    model_info = result.get('model', 'unknown')
                    provider_info = result.get('provider', 'unknown')
                    provider_class = f"provider-{provider_info}" if provider_info in ['ollama', 'gemini', 'openai', 'claude', 'openrouter', 'groq', 'deepseek'] else ""
                    
                    st.markdown(f"""
                    <div style="font-size: 0.75rem; color: #9CA3AF; margin-bottom: 0.5rem;">
                        🤖 Gerado por <strong style="color: #0A2463;">{model_info}</strong>
                        <span class="provider-badge {provider_class}">{provider_info}</span>
                        {f'| 🔢 {result.get("tokens_used", 0)} tokens' if result.get('tokens_used', 0) > 0 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(result['response'])
                    
                    if result.get('sources'):
                        with st.expander(f"📄 Fontes consultadas ({len(result['sources'])})"):
                            for source in result['sources']:
                                score = source.get('score', 0)
                                score_class = "score-high" if score > 0.7 else "score-medium" if score > 0.4 else "score-low"
                                category = source.get('category', 'Geral')
                                
                                st.markdown(f"""
                                <div class="source-card source-card-{category.lower()}">
                                    <strong>📎 {source['filename']}</strong>
                                    <br>
                                    <span style="color: #6B7280; font-size: 0.85rem;">Categoria: {category}</span>
                                    <br>
                                    <span class="{score_class}">Relevância: {score:.3f}</span>
                                    <br>
                                    <span style="color: #9CA3AF; font-size: 0.8rem;">{source.get('preview', '')}</span>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['response'],
                        "sources": result.get('sources', [])
                    })
                else:
                    st.error(result['response'])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['response']
                    })
                    
            except Exception as e:
                error_msg = f"❌ Erro ao processar: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# ==================== FOOTER ====================

st.markdown("""
<div class="footer">
    🚀 <strong>TechFlow Solutions</strong> • Agente Corporativo IA • 
    <span style="color: #D1D5DB;">|</span> 
    Baseado em documentos internos da empresa •
    <span style="color: #D1D5DB;">|</span> 
    🔒 Dados processados localmente
</div>
""", unsafe_allow_html=True)
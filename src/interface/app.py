"""
Interface Streamlit para o Agente RAG Corporativo
Suporte universal para múltiplos LLMs (Ollama, Gemini, OpenAI, Claude)
"""

import sys
import os
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
    page_title="🤖 Agente Corporativo - TechFlow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS PERSONALIZADO ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #F3F4F6;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
    }
    .source-card-ollama {
        border-left-color: #10B981;
    }
    .source-card-gemini {
        border-left-color: #8B5CF6;
    }
    .source-card-openai {
        border-left-color: #22C55E;
    }
    .source-card-claude {
        border-left-color: #D97706;
    }
    .score-high {
        color: #10B981;
        font-weight: 600;
    }
    .score-medium {
        color: #F59E0B;
        font-weight: 600;
    }
    .score-low {
        color: #EF4444;
        font-weight: 600;
    }
    .stChatMessage {
        padding: 1rem;
    }
    .provider-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        color: white;
        margin-left: 0.5rem;
    }
    .provider-ollama {
        background-color: #10B981;
    }
    .provider-gemini {
        background-color: #8B5CF6;
    }
    .provider-openai {
        background-color: #22C55E;
    }
    .provider-claude {
        background-color: #D97706;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🤖 Agente Corporativo")
    st.markdown("---")
    
    st.markdown("### 📊 Sobre o Agente")
    st.markdown("""
    Este agente responde perguntas baseado em documentos internos da empresa.
    
    **Funcionalidades:**
    - 🔍 Busca semântica
    - 🎯 Reranking de resultados
    - 🧠 Geração de respostas com IA
    - 📄 Citação de fontes
    - 🌐 Múltiplos provedores de IA
    """)
    
    st.markdown("---")
    
    st.markdown("### 📁 Documentos Indexados")
    
    # Mostra estatísticas
    if 'doc_count' in st.session_state:
        st.metric("Total de Chunks", st.session_state.doc_count)
    
    st.markdown("---")
    
    st.markdown("### 🏷️ Filtros")
    
    # Filtro por categoria
    categories = ["Todas", "Rh", "Financeiro", "Legal", "Operacional", "Estrategico"]
    selected_category = st.selectbox(
        "Filtrar por categoria:",
        categories,
        help="Filtra a busca apenas nos documentos da categoria selecionada"
    )
    
    st.markdown("---")
    
    st.markdown("### 🧠 Modelo de IA")
    
    # Provedor
    provider = st.selectbox(
        "Provedor:",
        ["ollama", "gemini", "openai", "claude"],
        index=0,
        help="Escolha qual LLM usar. Para APIs, configure as chaves no arquivo .env"
    )
    
    # Modelo específico
    default_models = {
        "ollama": "gemma3:1b",
        "gemini": "gemini-1.5-flash",
        "openai": "gpt-3.5-turbo",
        "claude": "claude-3-haiku-20240307"
    }
    
    model_name = st.text_input(
        "Modelo:",
        value=default_models.get(provider, "gemma3:1b"),
        help="Nome do modelo. Para Ollama, veja os disponíveis com 'ollama list'"
    )
    
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
        st.info(f"📌 Configure a API Key para **{provider.upper()}** no arquivo `.env`")
        
        # Mostra qual variável de ambiente
        env_var = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY"
        }.get(provider, f"{provider.upper()}_API_KEY")
        
        st.code(f"# .env\n{env_var}=sua_chave_aqui", language="bash")
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Configurações")
    
    # Configurações do LLM
    temperature = st.slider(
        "🌡️ Temperatura (criatividade)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Valores baixos = mais precisos, valores altos = mais criativos"
    )
    
    top_k = st.slider(
        "📄 Número de documentos",
        min_value=1,
        max_value=5,
        value=3,
        help="Quantos documentos usar para gerar a resposta"
    )
    
    st.markdown("---")
    
    # Mostra modelo atual
    if 'current_model' in st.session_state:
        provider_label = st.session_state.current_provider
        provider_class = f"provider-{provider_label}"
        st.markdown(f"""
        <div style="background-color: #F3F4F6; padding: 0.8rem; border-radius: 8px;">
            <span style="font-size: 0.9rem;">🔮 Modelo ativo:</span>
            <br>
            <span style="font-weight: 600;">{st.session_state.current_model}</span>
            <span class="provider-badge {provider_class}">{provider_label}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Botão de reset
    if st.button("🔄 Resetar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("TechFlow Solutions © 2025")

# ==================== INICIALIZAÇÃO ====================

@st.cache_resource
def init_components(provider, model_name, temperature, top_k):
    """
    Inicializa os componentes do RAG com o provedor e modelo selecionados
    """
    try:
        # Carrega variáveis de ambiente do arquivo .env
        from dotenv import load_dotenv
        load_dotenv()
        
        # Componentes base
        embedder = Embedder()
        vector_store = VectorStore()
        reranker = Reranker()
        
        # Cria LLM usando Factory
        try:
            llm = LLMFactory.create(
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=500
            )
        except ValueError as e:
            st.error(f"❌ Erro ao criar LLM: {e}")
            st.stop()
        
        # Cria agente
        agent = RAGAgent(
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            llm=llm,
            top_k=top_k
        )
        
        # Atualiza contagem
        stats = vector_store.get_stats()
        st.session_state.doc_count = stats['document_count']
        
        # Salva informações do modelo
        st.session_state.current_model = model_name
        st.session_state.current_provider = provider
        
        return agent
        
    except Exception as e:
        st.error(f"❌ Erro ao inicializar componentes: {e}")
        st.stop()


# ==================== ESTADO DA SESSÃO ====================

# Inicializa histórico
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Inicializa agente (com os parâmetros atuais)
if 'agent' not in st.session_state:
    with st.spinner("🔄 Carregando agente..."):
        st.session_state.agent = init_components(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            top_k=top_k
        )

# Verifica se o provedor ou modelo mudou para recarregar
if st.session_state.get('last_provider') != provider or \
   st.session_state.get('last_model') != model_name:
    
    # Salva o estado atual
    st.session_state.last_provider = provider
    st.session_state.last_model = model_name
    
    # Recarrega o agente
    with st.spinner("🔄 Trocando modelo..."):
        st.session_state.agent = init_components(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            top_k=top_k
        )
        # Limpa conversa ao trocar de modelo
        st.session_state.messages = []
        st.rerun()

# Atualiza configurações do agente
if 'agent' in st.session_state:
    st.session_state.agent.top_k = top_k
    if hasattr(st.session_state.agent, 'llm'):
        st.session_state.agent.llm.temperature = temperature

# ==================== HEADER ====================
st.markdown('<p class="main-header">🤖 Agente Corporativo</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Pergunte sobre políticas, benefícios, documentos e mais!</p>',
    unsafe_allow_html=True
)

# ==================== CHAT ====================

# Mensagem de boas-vindas
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Olá! Sou o agente corporativo da TechFlow Solutions!**
        
        Posso ajudar você com:
        - 📋 **RH**: Benefícios, férias, onboarding
        - 💰 **Financeiro**: Reembolso, orçamentos
        - ⚖️ **Legal**: Políticas, privacidade, LGPD
        - 🔧 **Operacional**: API, processos
        - 🎯 **Estratégico**: OKRs, roadmap
        
        **Exemplos de perguntas:**
        - "Quais são os benefícios oferecidos pela empresa?"
        - "Como solicito reembolso de despesas?"
        - "Como funciona o processo de onboarding?"
        
        💡 **Modelo ativo:** `{}` ({})
        
        Digite sua pergunta abaixo! 👇
        """.format(
            st.session_state.get('current_model', 'gemma3:1b'),
            st.session_state.get('current_provider', 'ollama')
        ))

# Histórico de mensagens
for message_data in st.session_state.messages:
    with st.chat_message(message_data["role"]):
        st.markdown(message_data["content"])
        
        # Mostra fontes se for resposta do assistente
        if "sources" in message_data and message_data["sources"]:
            with st.expander("📄 Ver fontes"):
                for source in message_data["sources"]:
                    score = source.get('score', 0)
                    score_class = "score-high" if score > 0.7 else "score-medium" if score > 0.4 else "score-low"
                    
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>{source['filename']}</strong>
                        <br>
                        <span>Categoria: {source.get('category', 'N/A')}</span>
                        <br>
                        <span class="{score_class}">Score: {score:.3f}</span>
                        <br>
                        <span style="color: #6B7280; font-size: 0.9rem;">{source.get('preview', '')[:150]}...</span>
                    </div>
                    """, unsafe_allow_html=True)

# ==================== INPUT DO USUÁRIO ====================

if prompt := st.chat_input("Digite sua pergunta..."):
    # Adiciona pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gera resposta
    with st.chat_message("assistant"):
        with st.spinner("🔍 Consultando documentos..."):
            try:
                # Filtro por categoria
                filters = None
                if selected_category != "Todas":
                    filters = {"category": selected_category}
                
                # Chama o agente
                result = st.session_state.agent.ask(prompt, filters=filters)
                
                if result['success']:
                    # Mostra qual modelo gerou a resposta
                    model_info = result.get('model', 'unknown')
                    provider_info = result.get('provider', 'unknown')
                    provider_class = f"provider-{provider_info}" if provider_info in ['ollama', 'gemini', 'openai', 'claude'] else ""
                    
                    st.markdown(f"""
                    <div style="font-size: 0.8rem; color: #6B7280; margin-bottom: 0.5rem;">
                        🤖 Resposta gerada por <strong>{model_info}</strong>
                        <span class="provider-badge {provider_class}">{provider_info}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostra resposta
                    st.markdown(result['response'])
                    
                    # Mostra fontes
                    if result.get('sources'):
                        with st.expander(f"📄 Fontes consultadas ({len(result['sources'])})"):
                            for source in result['sources']:
                                score = source.get('score', 0)
                                score_class = "score-high" if score > 0.7 else "score-medium" if score > 0.4 else "score-low"
                                
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>{source['filename']}</strong>
                                    <br>
                                    <span>Categoria: {source.get('category', 'N/A')}</span>
                                    <br>
                                    <span class="{score_class}">Score: {score:.3f}</span>
                                    <br>
                                    <span style="color: #6B7280; font-size: 0.9rem;">{source.get('preview', '')}</span>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Mostra tokens usados (se disponível)
                    if result.get('tokens_used', 0) > 0:
                        st.caption(f"📊 Tokens usados: {result.get('tokens_used', 0)}")
                    
                    # Salva no histórico
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
                error_msg = f"❌ Erro ao processar sua pergunta: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# ==================== RODAPÉ ====================
st.markdown("---")
st.markdown("""
<center>
<small style="color: #9CA3AF;">
🤖 Agente Corporativo TechFlow Solutions<br>
Baseado em documentos internos da empresa
</small>
</center>
""", unsafe_allow_html=True)
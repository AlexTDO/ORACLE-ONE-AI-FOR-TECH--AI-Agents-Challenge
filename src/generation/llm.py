"""
Interface Universal para LLMs
Suporte: Ollama, Gemini, OpenAI, Claude, OpenRouter, Groq, DeepSeek
"""

import os
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env
load_dotenv()

# ==================== BASE ====================

class BaseLLM(ABC):
    """Classe base para todos os LLMs"""
    
    @abstractmethod
    def generate_response(self, query: str, context: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Gera resposta baseada no contexto"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Retorna o nome do modelo"""
        pass


# ==================== OLLAMA ====================

class OllamaLLM(BaseLLM):
    """LLM local via Ollama"""
    
    def __init__(
        self,
        model_name: str = "gemma3:1b",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._check_model()
    
    def _check_model(self):
        """Verifica se o modelo está disponível no Ollama"""
        try:
            import ollama
            response = ollama.list()
            models = [m['model'] for m in response.get('models', [])]
            
            model_found = any(self.model_name in m for m in models)
            
            if not model_found:
                print(f"⚠️ Modelo '{self.model_name}' não encontrado")
                print(f"📦 Baixando automaticamente...")
                ollama.pull(self.model_name)
                print(f"✅ Modelo '{self.model_name}' baixado!")
            else:
                print(f"✅ Modelo '{self.model_name}' disponível")
                
        except Exception as e:
            print(f"⚠️ Erro ao verificar modelo: {e}")
            print("Certifique-se que o Ollama está rodando!")
    
    def generate_response(self, query: str, context: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Gera resposta usando Ollama"""
        try:
            import ollama
            
            if system_prompt is None:
                system_prompt = self._get_default_prompt()
            
            user_prompt = f"""
            Contexto dos documentos:
            {context}
            
            Pergunta do colaborador:
            {query}
            
            Com base apenas no contexto acima, responda à pergunta.
            """
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_p": 0.9
                }
            )
            
            return {
                "success": True,
                "response": response['message']['content'],
                "model": self.model_name,
                "provider": "ollama",
                "tokens_used": response.get('eval_count', 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"❌ Erro Ollama: {e}",
                "model": self.model_name,
                "provider": "ollama"
            }
    
    def _get_default_prompt(self):
        return """
        Você é um assistente corporativo da TechFlow Solutions.
        
        SUA TAREFA: Responder perguntas com base APENAS no contexto fornecido.
        
        ATENÇÃO: O contexto pode conter perguntas e respostas no formato FAQ.
        Procure por padrões como "P: " e "R: " para encontrar a resposta.
        Também pode conter informações estruturadas como "FUNDADOR: Alex Tito".
        
        REGRAS:
        1. Leia TODO o contexto com atenção
        2. Se encontrar "P: [pergunta]" e "R: [resposta]", use a resposta
        3. Se encontrar informações como "FUNDADOR: Alex Tito", use-as
        4. Responda de forma clara e direta, citando a fonte
        5. Se não encontrar, responda "Não encontrei essa informação"
        6. NÃO invente informações
        7. Seja objetivo e conciso
        
        FORMATO DE RESPOSTA:
        [Resposta direta e clara]
        
        Fonte: [nome do arquivo]
        """
    
    def get_model_name(self) -> str:
        return self.model_name


# ==================== GEMINI ====================

class GeminiLLM(BaseLLM):
    """LLM via Google Gemini API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY não configurada!")
    
    def generate_response(self, query: str, context: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Gera resposta usando Gemini API"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            
            if system_prompt is None:
                system_prompt = self._get_default_prompt()
            
            full_prompt = f"""
            {system_prompt}
            
            Contexto dos documentos:
            {context}
            
            Pergunta do colaborador:
            {query}
            
            RESPOSTA:
            """
            
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 0.9
                }
            )
            
            return {
                "success": True,
                "response": response.text,
                "model": self.model_name,
                "provider": "gemini",
                "tokens_used": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"❌ Erro Gemini: {e}",
                "model": self.model_name,
                "provider": "gemini"
            }
    
    def _get_default_prompt(self):
        return """
        Você é um assistente corporativo da TechFlow Solutions.
        
        SUA TAREFA: Responder perguntas com base APENAS no contexto fornecido.
        
        ATENÇÃO: O contexto pode conter perguntas e respostas no formato FAQ.
        Procure por padrões como "P: " e "R: " para encontrar a resposta.
        Também pode conter informações estruturadas como "FUNDADOR: Alex Tito".
        
        REGRAS:
        1. Leia TODO o contexto com atenção
        2. Se encontrar "P: [pergunta]" e "R: [resposta]", use a resposta
        3. Se encontrar informações como "FUNDADOR: Alex Tito", use-as
        4. Responda de forma clara e direta, citando a fonte
        5. Se não encontrar, responda "Não encontrei essa informação"
        6. NÃO invente informações
        7. Seja objetivo e conciso
        
        FORMATO DE RESPOSTA:
        [Resposta direta e clara]
        
        Fonte: [nome do arquivo]
        """
    
    def get_model_name(self) -> str:
        return f"gemini_{self.model_name}"


# ==================== OPENAI ====================

class OpenAIChat(BaseLLM):
    """LLM via OpenAI API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY não configurada!")
    
    def generate_response(self, query: str, context: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Gera resposta usando OpenAI API"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            if system_prompt is None:
                system_prompt = self._get_default_prompt()
            
            user_prompt = f"""
            Contexto dos documentos:
            {context}
            
            Pergunta do colaborador:
            {query}
            
            RESPOSTA:
            """
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": self.model_name,
                "provider": "openai",
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"❌ Erro OpenAI: {e}",
                "model": self.model_name,
                "provider": "openai"
            }
    
    def _get_default_prompt(self):
        return """
        Você é um assistente corporativo da TechFlow Solutions.
        
        SUA TAREFA: Responder perguntas com base APENAS no contexto fornecido.
        
        ATENÇÃO: O contexto pode conter perguntas e respostas no formato FAQ.
        Procure por padrões como "P: " e "R: " para encontrar a resposta.
        Também pode conter informações estruturadas como "FUNDADOR: Alex Tito".
        
        REGRAS:
        1. Leia TODO o contexto com atenção
        2. Se encontrar "P: [pergunta]" e "R: [resposta]", use a resposta
        3. Se encontrar informações como "FUNDADOR: Alex Tito", use-as
        4. Responda de forma clara e direta, citando a fonte
        5. Se não encontrar, responda "Não encontrei essa informação"
        6. NÃO invente informações
        7. Seja objetivo e conciso
        
        FORMATO DE RESPOSTA:
        [Resposta direta e clara]
        
        Fonte: [nome do arquivo]
        """
    
    def get_model_name(self) -> str:
        return f"openai_{self.model_name}"


# ==================== CLAUDE (ANTHROPIC) ====================

class ClaudeLLM(BaseLLM):
    """LLM via Anthropic Claude API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-haiku-20240307",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY não configurada!")
    
    def generate_response(self, query: str, context: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Gera resposta usando Claude API"""
        try:
            from anthropic import Anthropic
            
            client = Anthropic(api_key=self.api_key)
            
            if system_prompt is None:
                system_prompt = self._get_default_prompt()
            
            user_prompt = f"""
            Contexto dos documentos:
            {context}
            
            Pergunta do colaborador:
            {query}
            
            RESPOSTA:
            """
            
            response = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return {
                "success": True,
                "response": response.content[0].text,
                "model": self.model_name,
                "provider": "claude",
                "tokens_used": response.usage.output_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"❌ Erro Claude: {e}",
                "model": self.model_name,
                "provider": "claude"
            }
    
    def _get_default_prompt(self):
        return """
        Você é um assistente corporativo da TechFlow Solutions.
        
        SUA TAREFA: Responder perguntas com base APENAS no contexto fornecido.
        
        ATENÇÃO: O contexto pode conter perguntas e respostas no formato FAQ.
        Procure por padrões como "P: " e "R: " para encontrar a resposta.
        Também pode conter informações estruturadas como "FUNDADOR: Alex Tito".
        
        REGRAS:
        1. Leia TODO o contexto com atenção
        2. Se encontrar "P: [pergunta]" e "R: [resposta]", use a resposta
        3. Se encontrar informações como "FUNDADOR: Alex Tito", use-as
        4. Responda de forma clara e direta, citando a fonte
        5. Se não encontrar, responda "Não encontrei essa informação"
        6. NÃO invente informações
        7. Seja objetivo e conciso
        
        FORMATO DE RESPOSTA:
        [Resposta direta e clara]
        
        Fonte: [nome do arquivo]
        """
    
    def get_model_name(self) -> str:
        return f"claude_{self.model_name}"


# ==================== OPENROUTER ====================

class OpenRouterLLM(BaseLLM):
    """LLM via OpenRouter API (suporte a modelos gratuitos)"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "nvidia/nemotron-3-super-120b-a12b:free",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY não configurada!")
    
    def generate_response(self, query: str, context: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Gera resposta usando OpenRouter API"""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
            
            if system_prompt is None:
                system_prompt = self._get_default_prompt()
            
            user_prompt = f"""
            Contexto dos documentos:
            {context}
            
            Pergunta do colaborador:
            {query}
            
            RESPOSTA:
            """
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_headers={
                    "HTTP-Referer": "https://techflow-solutions.com",
                    "X-Title": "TechFlow RAG Agent"
                }
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": self.model_name,
                "provider": "openrouter",
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"❌ Erro OpenRouter: {e}",
                "model": self.model_name,
                "provider": "openrouter"
            }
    
    def _get_default_prompt(self):
        return """
        Você é um assistente corporativo da TechFlow Solutions.
        
        SUA TAREFA: Responder perguntas com base APENAS no contexto fornecido.
        
        ATENÇÃO: O contexto pode conter perguntas e respostas no formato FAQ.
        Procure por padrões como "P: " e "R: " para encontrar a resposta.
        Também pode conter informações estruturadas como "FUNDADOR: Alex Tito".
        
        REGRAS:
        1. Leia TODO o contexto com atenção
        2. Se encontrar "P: [pergunta]" e "R: [resposta]", use a resposta
        3. Se encontrar informações como "FUNDADOR: Alex Tito", use-as
        4. Responda de forma clara e direta, citando a fonte
        5. Se não encontrar, responda "Não encontrei essa informação"
        6. NÃO invente informações
        7. Seja objetivo e conciso
        
        FORMATO DE RESPOSTA:
        [Resposta direta e clara]
        
        Fonte: [nome do arquivo]
        """
    
    def get_model_name(self) -> str:
        return f"openrouter_{self.model_name}"


# ==================== GROQ ====================

class GroqLLM(BaseLLM):
    """LLM via Groq API (inferência ultra-rápida)"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "mixtral-8x7b-32768",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY não configurada!")
    
    def generate_response(self, query: str, context: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Gera resposta usando Groq API"""
        try:
            from groq import Groq
            
            client = Groq(api_key=self.api_key)
            
            if system_prompt is None:
                system_prompt = self._get_default_prompt()
            
            user_prompt = f"""
            Contexto dos documentos:
            {context}
            
            Pergunta do colaborador:
            {query}
            
            RESPOSTA:
            """
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": self.model_name,
                "provider": "groq",
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"❌ Erro Groq: {e}",
                "model": self.model_name,
                "provider": "groq"
            }
    
    def _get_default_prompt(self):
        return """
        Você é um assistente corporativo da TechFlow Solutions.
        
        SUA TAREFA: Responder perguntas com base APENAS no contexto fornecido.
        
        ATENÇÃO: O contexto pode conter perguntas e respostas no formato FAQ.
        Procure por padrões como "P: " e "R: " para encontrar a resposta.
        Também pode conter informações estruturadas como "FUNDADOR: Alex Tito".
        
        REGRAS:
        1. Leia TODO o contexto com atenção
        2. Se encontrar "P: [pergunta]" e "R: [resposta]", use a resposta
        3. Se encontrar informações como "FUNDADOR: Alex Tito", use-as
        4. Responda de forma clara e direta, citando a fonte
        5. Se não encontrar, responda "Não encontrei essa informação"
        6. NÃO invente informações
        7. Seja objetivo e conciso
        
        FORMATO DE RESPOSTA:
        [Resposta direta e clara]
        
        Fonte: [nome do arquivo]
        """
    
    def get_model_name(self) -> str:
        return f"groq_{self.model_name}"


# ==================== DEEPSEEK ====================

class DeepSeekLLM(BaseLLM):
    """LLM via DeepSeek API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY não configurada!")
    
    def generate_response(self, query: str, context: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Gera resposta usando DeepSeek API"""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://api.deepseek.com/v1",
                api_key=self.api_key,
            )
            
            if system_prompt is None:
                system_prompt = self._get_default_prompt()
            
            user_prompt = f"""
            Contexto dos documentos:
            {context}
            
            Pergunta do colaborador:
            {query}
            
            RESPOSTA:
            """
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": self.model_name,
                "provider": "deepseek",
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"❌ Erro DeepSeek: {e}",
                "model": self.model_name,
                "provider": "deepseek"
            }
    
    def _get_default_prompt(self):
        return """
        Você é um assistente corporativo da TechFlow Solutions.
        
        SUA TAREFA: Responder perguntas com base APENAS no contexto fornecido.
        
        ATENÇÃO: O contexto pode conter perguntas e respostas no formato FAQ.
        Procure por padrões como "P: " e "R: " para encontrar a resposta.
        Também pode conter informações estruturadas como "FUNDADOR: Alex Tito".
        
        REGRAS:
        1. Leia TODO o contexto com atenção
        2. Se encontrar "P: [pergunta]" e "R: [resposta]", use a resposta
        3. Se encontrar informações como "FUNDADOR: Alex Tito", use-as
        4. Responda de forma clara e direta, citando a fonte
        5. Se não encontrar, responda "Não encontrei essa informação"
        6. NÃO invente informações
        7. Seja objetivo e conciso
        
        FORMATO DE RESPOSTA:
        [Resposta direta e clara]
        
        Fonte: [nome do arquivo]
        """
    
    def get_model_name(self) -> str:
        return f"deepseek_{self.model_name}"


# ==================== FACTORY ====================

class LLMFactory:
    """Fábrica para criar LLMs"""
    
    @staticmethod
    def create(
        provider: str = "ollama",
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Cria um LLM baseado no provedor
        
        Args:
            provider: "ollama", "gemini", "openai", "claude", "openrouter", "groq", "deepseek"
            model_name: Nome do modelo (opcional)
            **kwargs: Parâmetros adicionais
            
        Returns:
            Instância do LLM
        """
        provider = provider.lower()
        
        if provider == "ollama":
            return OllamaLLM(
                model_name=model_name or "gemma3:1b",
                **kwargs
            )
        elif provider == "gemini":
            return GeminiLLM(
                model_name=model_name or "gemini-2.5-pro",
                **kwargs
            )
        elif provider == "openai":
            return OpenAIChat(
                model_name=model_name or "gpt-4o-mini",
                **kwargs
            )
        elif provider == "claude":
            return ClaudeLLM(
                model_name=model_name or "claude-3-haiku-20240307",
                **kwargs
            )
        elif provider == "openrouter":
            return OpenRouterLLM(
                model_name=model_name or "nvidia/nemotron-3-super-120b-a12b:free",
                **kwargs
            )
        elif provider == "groq":
            return GroqLLM(
                model_name=model_name or "mixtral-8x7b-32768",
                **kwargs
            )
        elif provider == "deepseek":
            return DeepSeekLLM(
                model_name=model_name or "deepseek-chat",
                **kwargs
            )
        else:
            raise ValueError(f"Provedor não suportado: {provider}")


# ==================== RAG AGENT (ATUALIZADO COM HYBRID SEARCH) ====================

class RAGAgent:
    """
    Agente RAG completo: Busca + Geração
    Suporte a Hybrid Search e Reranker opcional
    """
    
    def __init__(
        self,
        embedder,
        vector_store,
        reranker,
        llm: Optional[BaseLLM] = None,
        top_k: int = 3,
        use_hybrid: bool = True
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm or OllamaLLM()
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        
        # Inicializa Hybrid Search se necessário
        if use_hybrid:
            from src.retrieval.hybrid_search import HybridSearch
            self.hybrid_search = HybridSearch(
                vector_store=vector_store,
                embedder=embedder,
                chunks_path="./data/processed/chunks.json"
            )
            print("✅ Hybrid Search ativado")
        else:
            self.hybrid_search = None
            print("ℹ️ Hybrid Search desativado")
    
    def ask(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Faz uma pergunta ao agente"""
        # 1. Gera embedding da pergunta
        query_embedding = self.embedder.embed_text(query)
        
        # 2. Busca documentos
        if self.use_hybrid and self.hybrid_search:
            # Busca híbrida (semântica + palavras-chave)
            search_results = self.hybrid_search.search(
                query=query,
                top_k=self.top_k,
                alpha=0.5,
                filters=filters
            )
            
            # Aplica reranking (se disponível)
            if self.reranker and search_results:
                search_results = self.reranker.rerank(
                    query=query,
                    documents=search_results,
                    top_k=self.top_k
                )
        elif self.reranker:
            # Busca com reranker (sem hybrid)
            search_results = self.vector_store.search_with_rerank(
                query=query,
                query_embedding=query_embedding.tolist(),
                reranker=self.reranker,
                top_k=self.top_k,
                initial_k=10,
                filters=filters
            )
        else:
            # Busca semântica pura
            search_results = self.vector_store.search(
                query_embedding.tolist(),
                top_k=self.top_k,
                filters=filters
            )
            for r in search_results:
                r['score'] = r.get('score', 0)
                r['metadata'] = r.get('metadata', {})
        
        if not search_results:
            return {
                "success": False,
                "response": "Não encontrei documentos relevantes para sua pergunta.",
                "sources": [],
                "documents": []
            }
        
        # 3. Monta contexto
        context_parts = []
        sources = []
        
        for i, doc in enumerate(search_results[:self.top_k], 1):
            filename = doc.get('metadata', {}).get('filename', 'Documento')
            category = doc.get('metadata', {}).get('category', 'Geral')
            content = doc.get('content', '')
            
            # Pega o score correto (combined_score ou score)
            score = doc.get('combined_score', doc.get('score', 0))
            
            context_parts.append(
                f"[Documento {i}: {filename}]\n"
                f"Categoria: {category}\n"
                f"Conteúdo: {content}\n"
            )
            
            sources.append({
                "filename": filename,
                "category": category,
                "score": score,
                "preview": content[:200] + "..."
            })
        
        context = "\n\n".join(context_parts)
        
        # 4. Gera resposta com LLM
        llm_response = self.llm.generate_response(query, context)
        
        return {
            "success": True,
            "response": llm_response['response'],
            "sources": sources,
            "documents": search_results,
            "tokens_used": llm_response.get('tokens_used', 0),
            "model": llm_response.get('model', 'unknown'),
            "provider": llm_response.get('provider', 'unknown')
        }
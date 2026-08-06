"""
Agente RAG com LangGraph - Com Streaming e Acompanhamento
"""

import sys
import time
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional, Iterator

sys.path.append(str(Path(__file__).parent.parent.parent))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import Reranker
from src.generation.llm import LLMFactory


# ============================================
# 1. DEFINIÇÃO DO ESTADO
# ============================================

class RAGState(TypedDict):
    """Estado que viaja por todos os nós do grafo"""
    
    pergunta: str
    filtros: Optional[Dict[str, Any]]
    embedding: Optional[List[float]]
    documentos_busca: Optional[List[Dict[str, Any]]]
    documentos_rerank: Optional[List[Dict[str, Any]]]
    resposta: Optional[str]
    tokens_usados: int
    passos: List[str]
    erros: List[str]
    stream: List[Dict[str, Any]]  # Para streaming


# ============================================
# 2. NÓS DO GRAFO
# ============================================

class RAGGraph:
    """Grafo RAG completo com LangGraph e Streaming"""
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        reranker: Optional[Reranker] = None,
        llm_provider: str = "ollama",
        llm_model: str = "gemma3:1b",
        top_k: int = 3,
        temperature: float = 0.3
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.reranker = reranker or Reranker()
        self.llm = LLMFactory.create(
            provider=llm_provider,
            model_name=llm_model,
            temperature=temperature
        )
        self.top_k = top_k
        
        # Constrói o grafo com MemorySaver
        self.app = self._build_graph().compile(checkpointer=MemorySaver())
        
        print("✅ Agente RAG com LangGraph e Streaming inicializado!")
    
    def _build_graph(self) -> StateGraph:
        """Constrói o grafo RAG"""
        builder = StateGraph(RAGState)
        
        builder.add_node("embed", self._node_embed)
        builder.add_node("search", self._node_search)
        builder.add_node("rerank", self._node_rerank)
        builder.add_node("generate", self._node_generate)
        
        builder.add_edge(START, "embed")
        builder.add_edge("embed", "search")
        builder.add_edge("search", "rerank")
        builder.add_edge("rerank", "generate")
        builder.add_edge("generate", END)
        
        return builder
    
    # ==========================================
    # NÓS DO GRAFO
    # ==========================================
    
    def _node_embed(self, state: RAGState) -> dict:
        """Nó 1: Gera embedding da pergunta"""
        print("🔍 [Passo 1] Gerando embedding...")
        
        try:
            embedding = self.embedder.embed_text(state["pergunta"])
            state["embedding"] = embedding.tolist()
            state["passos"] = state.get("passos", []) + ["embed"]
            
            # Adiciona ao stream
            state["stream"] = state.get("stream", []) + [{
                "step": "embed",
                "status": "completed",
                "message": f"✅ Embedding gerado (dimensão: {len(embedding)})"
            }]
            
            print(f"✅ Embedding gerado (dimensão: {len(embedding)})")
        except Exception as e:
            print(f"❌ Erro no embedding: {e}")
            state["erros"] = state.get("erros", []) + [f"embed: {e}"]
            state["stream"] = state.get("stream", []) + [{
                "step": "embed",
                "status": "error",
                "message": f"❌ Erro: {e}"
            }]
        
        return {
            "embedding": state["embedding"],
            "passos": state["passos"],
            "erros": state["erros"],
            "stream": state["stream"]
        }
    
    def _node_search(self, state: RAGState) -> dict:
        """Nó 2: Busca documentos similares"""
        print("🔍 [Passo 2] Buscando documentos...")
        
        try:
            results = self.vector_store.search(
                state["embedding"],
                top_k=self.top_k * 2,
                filters=state.get("filtros")
            )
            state["documentos_busca"] = results
            state["passos"] = state.get("passos", []) + ["search"]
            
            state["stream"] = state.get("stream", []) + [{
                "step": "search",
                "status": "completed",
                "message": f"✅ Encontrados {len(results)} documentos"
            }]
            
            print(f"✅ Encontrados {len(results)} documentos")
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            state["erros"] = state.get("erros", []) + [f"search: {e}"]
            state["stream"] = state.get("stream", []) + [{
                "step": "search",
                "status": "error",
                "message": f"❌ Erro: {e}"
            }]
        
        return {
            "documentos_busca": state["documentos_busca"],
            "passos": state["passos"],
            "erros": state["erros"],
            "stream": state["stream"]
        }
    
    def _node_rerank(self, state: RAGState) -> dict:
        """Nó 3: Reranking dos documentos"""
        print("🔍 [Passo 3] Aplicando reranking...")
        
        try:
            if state.get("documentos_busca"):
                results = self.reranker.rerank(
                    state["pergunta"],
                    state["documentos_busca"],
                    top_k=self.top_k
                )
                state["documentos_rerank"] = results
            else:
                state["documentos_rerank"] = []
            
            state["passos"] = state.get("passos", []) + ["rerank"]
            
            state["stream"] = state.get("stream", []) + [{
                "step": "rerank",
                "status": "completed",
                "message": f"✅ Reranking concluído ({len(state['documentos_rerank'])} documentos)"
            }]
            
            print(f"✅ Reranking concluído ({len(state['documentos_rerank'])} documentos)")
        except Exception as e:
            print(f"❌ Erro no reranking: {e}")
            state["erros"] = state.get("erros", []) + [f"rerank: {e}"]
            state["documentos_rerank"] = state.get("documentos_busca", [])[:self.top_k]
            state["stream"] = state.get("stream", []) + [{
                "step": "rerank",
                "status": "error",
                "message": f"❌ Erro: {e}"
            }]
        
        return {
            "documentos_rerank": state["documentos_rerank"],
            "passos": state["passos"],
            "erros": state["erros"],
            "stream": state["stream"]
        }
    
    def _node_generate(self, state: RAGState) -> dict:
        """Nó 4: Gera a resposta final"""
        print("🔍 [Passo 4] Gerando resposta...")
        
        try:
            # Monta contexto
            context_parts = []
            for doc in state.get("documentos_rerank", []):
                filename = doc['metadata'].get('filename', 'Documento')
                content = doc['content']
                context_parts.append(f"[{filename}]\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Gera resposta
            response = self.llm.generate_response(state["pergunta"], context)
            
            state["resposta"] = response.get("response", "Erro ao gerar resposta")
            state["tokens_usados"] = response.get("tokens_used", 0)
            state["passos"] = state.get("passos", []) + ["generate"]
            
            state["stream"] = state.get("stream", []) + [{
                "step": "generate",
                "status": "completed",
                "message": f"✅ Resposta gerada ({state['tokens_usados']} tokens)"
            }]
            
            print(f"✅ Resposta gerada ({state['tokens_usados']} tokens)")
        except Exception as e:
            print(f"❌ Erro na geração: {e}")
            state["erros"] = state.get("erros", []) + [f"generate: {e}"]
            state["resposta"] = f"Erro ao gerar resposta: {e}"
            state["stream"] = state.get("stream", []) + [{
                "step": "generate",
                "status": "error",
                "message": f"❌ Erro: {e}"
            }]
        
        return {
            "resposta": state["resposta"],
            "tokens_usados": state["tokens_usados"],
            "passos": state["passos"],
            "erros": state["erros"],
            "stream": state["stream"]
        }
    
    # ==========================================
    # MÉTODOS DE EXECUÇÃO
    # ==========================================
    
    def ask(self, pergunta: str, filtros: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Executa o agente RAG completo (sem streaming)
        Retorna o resultado completo de uma vez
        """
        print(f"\n🤖 PERGUNTA: {pergunta}")
        
        # Estado inicial
        initial_state = {
            "pergunta": pergunta,
            "filtros": filtros,
            "embedding": None,
            "documentos_busca": None,
            "documentos_rerank": None,
            "resposta": None,
            "tokens_usados": 0,
            "passos": [],
            "erros": [],
            "stream": []
        }
        
        try:
            start = time.time()
            config = {"configurable": {"thread_id": "rag_session"}}
            result = self.app.invoke(initial_state, config=config)
            elapsed = time.time() - start
            result["tempo_total"] = elapsed
            
            # Formata fontes
            sources = []
            for doc in result.get("documentos_rerank", []):
                sources.append({
                    "filename": doc['metadata'].get('filename', 'Documento'),
                    "category": doc['metadata'].get('category', 'Geral'),
                    "score": doc.get('score', 0),
                    "preview": doc['content'][:200] + "..."
                })
            
            print(f"\n⏱️ Tempo total: {elapsed:.2f}s")
            print(f"🔢 Tokens: {result.get('tokens_usados', 0)}")
            print(f"📊 Passos: {result.get('passos', [])}")
            
            return {
                "success": len(result.get("erros", [])) == 0,
                "response": result.get("resposta", ""),
                "sources": sources,
                "passos": result.get("passos", []),
                "tokens_usados": result.get("tokens_usados", 0),
                "tempo_total": elapsed,
                "erros": result.get("erros", [])
            }
            
        except Exception as e:
            print(f"❌ Erro no agente: {e}")
            return {
                "success": False,
                "response": f"❌ Erro no agente: {e}",
                "sources": [],
                "passos": [],
                "tokens_usados": 0,
                "tempo_total": 0,
                "erros": [str(e)]
            }
    
    def ask_stream(self, pergunta: str, filtros: Optional[Dict] = None) -> Iterator[Dict[str, Any]]:
        """
        Executa o agente com streaming
        Yield a cada passo para acompanhamento em tempo real
        """
        print(f"\n🤖 PERGUNTA: {pergunta}")
        
        # Estado inicial
        initial_state = {
            "pergunta": pergunta,
            "filtros": filtros,
            "embedding": None,
            "documentos_busca": None,
            "documentos_rerank": None,
            "resposta": None,
            "tokens_usados": 0,
            "passos": [],
            "erros": [],
            "stream": []
        }
        
        try:
            start = time.time()
            config = {"configurable": {"thread_id": "rag_session"}}
            
            # Stream dos eventos
            for event in self.app.stream(initial_state, config=config):
                for node_name, node_data in event.items():
                    if node_name in ["embed", "search", "rerank", "generate"]:
                        yield {
                            "step": node_name,
                            "status": "processing",
                            "data": node_data,
                            "stream": node_data.get("stream", [])
                        }
            
            # Estado final
            final_state = self.app.get_state(config)
            if final_state and final_state.values:
                elapsed = time.time() - start
                final_state.values["tempo_total"] = elapsed
                
                # Formata fontes para o resultado final
                sources = []
                for doc in final_state.values.get("documentos_rerank", []):
                    sources.append({
                        "filename": doc['metadata'].get('filename', 'Documento'),
                        "category": doc['metadata'].get('category', 'Geral'),
                        "score": doc.get('score', 0),
                        "preview": doc['content'][:200] + "..."
                    })
                
                yield {
                    "step": "complete",
                    "status": "completed",
                    "data": final_state.values,
                    "sources": sources,
                    "stream": final_state.values.get("stream", [])
                }
            
        except Exception as e:
            print(f"❌ Erro no agente: {e}")
            yield {
                "step": "error",
                "status": "error",
                "data": {"erros": [str(e)]},
                "stream": [{"step": "error", "status": "error", "message": str(e)}]
            }
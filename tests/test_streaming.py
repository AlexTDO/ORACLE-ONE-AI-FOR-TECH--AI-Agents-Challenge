"""
Teste do Streaming no LangGraph
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.agent.graph import RAGGraph


def main():
    """Testa o streaming do agente"""
    
    print("=" * 70)
    print("🧪 TESTE DE STREAMING DO LANGGRAPH")
    print("=" * 70)
    
    # Inicializa
    agent = RAGGraph(
        llm_provider="ollama",
        llm_model="gemma3:1b",
        top_k=3
    )
    
    query = "Quem fundou a TechFlow Solutions?"
    
    print(f"\n📝 PERGUNTA: {query}")
    print("\n" + "=" * 50)
    print("🔄 ACOMPANHANDO PASSOS (STREAMING):")
    print("=" * 50)
    
    # Executa com streaming
    for event in agent.ask_stream(query):
        step = event.get("step", "unknown")
        status = event.get("status", "unknown")
        
        if step == "complete":
            print("\n" + "=" * 50)
            print("✅ RESPOSTA FINAL:")
            print("-" * 50)
            data = event.get("data", {})
            print(data.get("resposta", "Sem resposta"))
            print("-" * 50)
            
            stream_data = event.get("stream", [])
            if stream_data:
                print("\n📊 RESUMO DOS PASSOS:")
                for s in stream_data:
                    print(f"  {s.get('message', s.get('step', ''))}")
            
            print(f"\n⏱️ Tempo total: {data.get('tempo_total', 0):.2f}s")
            print(f"🔢 Tokens: {data.get('tokens_usados', 0)}")
            
        elif step in ["embed", "search", "rerank", "generate"]:
            status_icon = "🔄" if status == "processing" else "✅" if status == "completed" else "❌"
            print(f"{status_icon} {step.capitalize()}...")
            
            stream_msgs = event.get("stream", [])
            for msg in stream_msgs:
                if msg.get("message"):
                    print(f"   {msg.get('message')}")
    
    print("\n" + "=" * 70)
    print("✅ TESTE DE STREAMING CONCLUÍDO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Script para processar todos os documentos e gerar chunks
"""

import sys
from pathlib import Path

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.loader import DocumentLoader


def main():
    """Processa documentos e gera chunks"""
    
    # Configuração
    data_root = Path(__file__).parent.parent / "data" / "raw"
    output_path = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"
    
    print("=" * 60)
    print("🚀 INICIANDO PROCESSAMENTO DE DOCUMENTOS")
    print("=" * 60)
    
    # Verifica se a pasta existe
    if not data_root.exists():
        print(f"❌ Pasta não encontrada: {data_root}")
        print("Crie a estrutura:")
        print("  data/")
        print("  ├── raw/")
        print("  │   ├── rh/")
        print("  │   ├── financeiro/")
        print("  │   ├── legal/")
        print("  │   ├── operacional/")
        print("  │   └── estrategico/")
        return
    
    # Carrega e processa
    loader = DocumentLoader(chunk_size=500, chunk_overlap=100)
    chunks = loader.load_all_documents(str(data_root))
    
    if chunks:
        # Salva chunks
        loader.save_chunks(chunks, str(output_path))
        
        # Estatísticas
        total_size = sum(len(c["content"]) for c in chunks)
        avg_size = total_size / len(chunks) if chunks else 0
        
        print("\n" + "=" * 60)
        print("📊 ESTATÍSTICAS")
        print("=" * 60)
        print(f"Total de chunks: {len(chunks)}")
        print(f"Total de caracteres: {total_size:,}")
        print(f"Média por chunk: {avg_size:.0f} caracteres")
        
        # Categorias
        categories = {}
        for chunk in chunks:
            cat = chunk["metadata"].get("category", "Desconhecido")
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n📂 Chunks por categoria:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        
        print("\n✅ Processamento concluído!")
        print(f"📁 Chunks salvos em: {output_path}")
    else:
        print("❌ Nenhum chunk foi gerado!")


if __name__ == "__main__":
    main()
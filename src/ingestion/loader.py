"""
Carregador de documentos - Orquestra a extração e chunking
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import json
from tqdm import tqdm

from src.ingestion.extractor import DocumentExtractor
from src.ingestion.chunker import DocumentChunker


class DocumentLoader:
    """Carrega e processa documentos da pasta"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):   # Mudar tamanha da janela de caracteres e sopreposição
        self.extractor = DocumentExtractor()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        
    def load_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Carrega todos os documentos de uma pasta
        
        Args:
            folder_path: Caminho da pasta com documentos
            
        Returns:
            Lista de chunks processados
        """
        all_chunks = []
        folder_path = Path(folder_path)
        
        # Mapeia categorias pela pasta
        category = folder_path.name.capitalize()
        
        # Lista arquivos suportados
        files = []
        for ext in self.extractor.supported_formats:
            files.extend(folder_path.glob(f"*{ext}"))
        
        print(f"📂 Processando {len(files)} arquivos em {folder_path}")
        
        for file_path in tqdm(files, desc=f"Processando {category}"):
            try:
                # Extrai conteúdo
                extracted = self.extractor.extract(file_path)
                content = extracted["content"]
                
                # Verifica se o conteúdo é válido
                if not content or len(content) < 10:
                    print(f"⚠️  Arquivo ignorado (pouco conteúdo): {file_path.name}")
                    continue
                
                # Metadados base
                base_metadata = {
                    "filename": file_path.name,
                    "category": category,
                    "format": file_path.suffix[1:],
                    "file_path": str(file_path),
                    **extracted["metadata"]
                }
                
                # Chunking
                chunks = self.chunker.chunk_document(
                    content,
                    base_metadata,
                    strategy="by_section"  # ← MUDAR DE "recursive" PARA "by_section"
                )
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"❌ Erro ao processar {file_path.name}: {str(e)}")
        
        print(f"✅ Gerados {len(all_chunks)} chunks para {category}")
        return all_chunks
    
    def load_all_documents(self, data_root: str) -> List[Dict[str, Any]]:
        """
        Carrega todos os documentos do diretório raiz
        
        Args:
            data_root: Diretório raiz com subpastas por categoria
            
        Returns:
            Lista de chunks processados
        """
        all_chunks = []
        data_root = Path(data_root)
        
        # Encontra todas as subpastas
        categories = [d for d in data_root.iterdir() if d.is_dir()]
        
        if not categories:
            # Se não houver subpastas, processa direto
            return self.load_folder(str(data_root))
        
        for category_path in categories:
            chunks = self.load_folder(str(category_path))
            all_chunks.extend(chunks)
        
        print(f"\n🎯 Total: {len(all_chunks)} chunks processados")
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str):
        """Salva chunks em arquivo JSON para cache"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepara para serialização
        serializable_chunks = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            # Converte Path para string se necessário
            if "file_path" in chunk_copy.get("metadata", {}):
                chunk_copy["metadata"]["file_path"] = str(chunk_copy["metadata"]["file_path"])
            serializable_chunks.append(chunk_copy)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Chunks salvos em: {output_path}")
    
    def load_chunks(self, input_path: str) -> List[Dict[str, Any]]:
        """Carrega chunks de arquivo JSON"""
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"📂 Carregados {len(chunks)} chunks de {input_path}")
        return chunks
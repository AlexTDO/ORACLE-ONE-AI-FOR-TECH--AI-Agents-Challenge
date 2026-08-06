"""
Divisão de documentos em chunks para indexação
"""

import re
from typing import List, Dict, Any
import tiktoken


class DocumentChunker:
    """Divide documentos em chunks otimizados para RAG"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        encoding_name: str = "cl100k_base"
    ):
        """
        Args:
            chunk_size: Tamanho do chunk em caracteres
            chunk_overlap: Sobreposição entre chunks
            encoding_name: Nome do encoding para contagem de tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        strategy: str = "recursive"
    ) -> List[Dict[str, Any]]:
        """
        Divide o documento em chunks
        
        Args:
            content: Texto do documento
            metadata: Metadados do documento
            strategy: Estratégia de chunking
            
        Returns:
            Lista de chunks com metadados
        """
        if strategy == "recursive":
            return self._recursive_chunk(content, metadata)
        elif strategy == "by_paragraph":
            return self._by_paragraph(content, metadata)
        elif strategy == "by_section":
            return self._by_section(content, metadata)
        else:
            return self._fixed_size_chunk(content, metadata)
    
    def _recursive_chunk(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunking recursivo (prioriza parágrafos e sentenças)"""
        chunks = []
        
        # Separa por parágrafos
        paragraphs = content.split('\n\n')
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size <= self.chunk_size:
                current_chunk.append(para)
                current_size += para_size
            else:
                # Se o parágrafo é maior que o chunk_size, quebra em sentenças
                if len(para) > self.chunk_size:
                    sentences = self._split_sentences(para)
                    for sent in sentences:
                        if len(sent) > self.chunk_size:
                            # Se a sentença é muito longa, quebra por tamanho
                            sub_chunks = self._split_by_size(sent, self.chunk_size)
                            for sub in sub_chunks:
                                chunks.append(self._create_chunk(
                                    sub, metadata, len(chunks)
                                ))
                        else:
                            chunks.append(self._create_chunk(
                                sent, metadata, len(chunks)
                            ))
                else:
                    # Finaliza chunk atual
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            '\n\n'.join(current_chunk), metadata, len(chunks)
                        ))
                    
                    # Inicia novo chunk com overlap
                    overlap_content = '\n\n'.join(current_chunk[-2:]) if len(current_chunk) > 2 else ''
                    current_chunk = [overlap_content, para] if overlap_content else [para]
                    current_size = len(overlap_content) + len(para)
        
        # Último chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                '\n\n'.join(current_chunk), metadata, len(chunks)
            ))
        
        return chunks
    
    def _by_paragraph(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunking por parágrafo (cada parágrafo é um chunk)"""
        chunks = []
        paragraphs = content.split('\n\n')
        
        for para in paragraphs:
            if para.strip():
                chunks.append(self._create_chunk(para, metadata, len(chunks)))
        
        return chunks
    
    def _by_section(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunking por seção (baseado em títulos)"""
        chunks = []
        
        # Detecta seções por padrões comuns
        section_patterns = [
            r'\n(\d+\.\s+[A-Z][^\n]+)\n',  # 1. Título
            r'\n([A-Z][A-Z\s]+)\n',          # TÍTULO EM MAIÚSCULAS
            r'\n(===.+===)\n',               # === TÍTULO ===
        ]
        
        sections = []
        current_section = {"title": "Início", "content": ""}
        
        lines = content.split('\n')
        for line in lines:
            is_section = False
            for pattern in section_patterns:
                if re.match(pattern, line):
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = {"title": line.strip(), "content": ""}
                    is_section = True
                    break
            
            if not is_section:
                current_section["content"] += line + "\n"
        
        if current_section["content"]:
            sections.append(current_section)
        
        for section in sections:
            # Se o conteúdo da seção é grande, aplica chunking recursivo
            if len(section["content"]) > self.chunk_size:
                section_chunks = self._fixed_size_chunk(
                    section["content"], 
                    {**metadata, "section": section["title"]}
                )
                chunks.extend(section_chunks)
            else:
                chunks.append(self._create_chunk(
                    section["content"],
                    {**metadata, "section": section["title"]},
                    len(chunks)
                ))
        
        return chunks
    
    def _fixed_size_chunk(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunking por tamanho fixo com overlap"""
        chunks = []
        start = 0
        content_len = len(content)
        
        while start < content_len:
            end = min(start + self.chunk_size, content_len)
            
            # Tenta não cortar no meio de palavra
            if end < content_len:
                while end > start and content[end] not in [' ', '\n', '.']:
                    end -= 1
            
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append(self._create_chunk(
                    chunk_text, metadata, len(chunks)
                ))
            
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Divide texto em sentenças"""
        # Padrão simples de quebra de sentenças
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_size(self, text: str, size: int) -> List[str]:
        """Divide texto por tamanho"""
        chunks = []
        for i in range(0, len(text), size):
            chunks.append(text[i:i+size].strip())
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_index: int
    ) -> Dict[str, Any]:
        """Cria um chunk com metadados"""
        return {
            "id": f"{metadata.get('filename', 'doc')}_{chunk_index}",
            "content": content,
            "metadata": {
                **metadata,
                "chunk_index": chunk_index,
                "token_count": len(self.encoding.encode(content))
            }
        }
    
    def count_tokens(self, text: str) -> int:
        """Conta tokens no texto"""
        return len(self.encoding.encode(text))
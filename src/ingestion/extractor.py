"""
Extrator de documentos para múltiplos formatos
Suporte: PDF, DOCX, XLSX, CSV, JSON, HTML, MD, PPTX
"""

import os
import json
import csv
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile

# PDF
from pypdf import PdfReader

# Word
from docx import Document

# Excel
import pandas as pd
from openpyxl import load_workbook

# HTML
from bs4 import BeautifulSoup

# Markdown
import markdown

# PowerPoint
from pptx import Presentation

# JSON e CSV já nativos


class DocumentExtractor:
    """Extrator de conteúdo de documentos em múltiplos formatos"""
    
    def __init__(self):
        self.supported_formats = [
            '.pdf', '.docx', '.xlsx', '.xls', 
            '.csv', '.json', '.html', '.htm', 
            '.md', '.txt', '.pptx'
        ]
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extrai conteúdo e metadados de um documento
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            Dict com 'content' (texto extraído) e 'metadata'
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_pdf(file_path)
        elif extension == '.docx':
            return self._extract_docx(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self._extract_excel(file_path)
        elif extension == '.csv':
            return self._extract_csv(file_path)
        elif extension == '.json':
            return self._extract_json(file_path)
        elif extension in ['.html', '.htm']:
            return self._extract_html(file_path)
        elif extension == '.md':
            return self._extract_markdown(file_path)
        elif extension == '.pptx':
            return self._extract_pptx(file_path)
        elif extension == '.txt':
            return self._extract_txt(file_path)
        else:
            raise ValueError(f"Formato não suportado: {extension}")
    
    def _extract_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extrai texto de PDF"""
        try:
            reader = PdfReader(file_path)
            text = ""
            metadata = {
                "pages": len(reader.pages),
                "author": reader.metadata.author if reader.metadata else None,
                "title": reader.metadata.title if reader.metadata else None,
                "creator": reader.metadata.creator if reader.metadata else None
            }
            
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            
            return {"content": text.strip(), "metadata": metadata}
        except Exception as e:
            return {"content": f"Erro ao extrair PDF: {str(e)}", "metadata": {}}
    
    def _extract_docx(self, file_path: Path) -> Dict[str, Any]:
        """Extrai texto de DOCX"""
        try:
            doc = Document(file_path)
            text = []
            
            # Extrai parágrafos
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)
            
            # Extrai tabelas
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text.append(" | ".join(row_text))
            
            return {
                "content": "\n\n".join(text),
                "metadata": {"paragraphs": len(doc.paragraphs), "tables": len(doc.tables)}
            }
        except Exception as e:
            return {"content": f"Erro ao extrair DOCX: {str(e)}", "metadata": {}}
    
    def _extract_excel(self, file_path: Path) -> Dict[str, Any]:
        """Extrai texto de Excel (XLSX/XLS)"""
        try:
            df_dict = pd.read_excel(file_path, sheet_name=None)
            text_parts = []
            sheets_info = []
            
            for sheet_name, df in df_dict.items():
                sheets_info.append({"name": sheet_name, "rows": len(df), "cols": len(df.columns)})
                
                # Converte para texto legível
                text_parts.append(f"=== Planilha: {sheet_name} ===\n")
                text_parts.append(df.to_string(index=False))
                text_parts.append("\n" + "="*50 + "\n")
            
            return {
                "content": "\n".join(text_parts),
                "metadata": {"sheets": sheets_info}
            }
        except Exception as e:
            return {"content": f"Erro ao extrair Excel: {str(e)}", "metadata": {}}
    
    def _extract_csv(self, file_path: Path) -> Dict[str, Any]:
        """Extrai texto de CSV"""
        try:
            df = pd.read_csv(file_path)
            text = f"=== CSV: {file_path.name} ===\n\n"
            text += df.to_string(index=False)
            
            return {
                "content": text,
                "metadata": {"rows": len(df), "columns": list(df.columns)}
            }
        except Exception as e:
            return {"content": f"Erro ao extrair CSV: {str(e)}", "metadata": {}}
    
    def _extract_json(self, file_path: Path) -> Dict[str, Any]:
        """Extrai texto de JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text = f"=== JSON: {file_path.name} ===\n\n"
            text += json.dumps(data, indent=2, ensure_ascii=False)
            
            return {
                "content": text,
                "metadata": {"keys": list(data.keys()) if isinstance(data, dict) else []}
            }
        except Exception as e:
            return {"content": f"Erro ao extrair JSON: {str(e)}", "metadata": {}}
    
    def _extract_html(self, file_path: Path) -> Dict[str, Any]:
        """Extrai texto de HTML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html = f.read()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove scripts e styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extrai título
            title = soup.title.string if soup.title else None
            
            # Extrai texto limpo
            text = soup.get_text(separator='\n')
            text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
            
            return {
                "content": text,
                "metadata": {"title": title}
            }
        except Exception as e:
            return {"content": f"Erro ao extrair HTML: {str(e)}", "metadata": {}}
    
    def _extract_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Extrai texto de Markdown"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Converte para HTML e depois extrai texto
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator='\n')
            
            return {
                "content": text,
                "metadata": {"format": "markdown"}
            }
        except Exception as e:
            return {"content": f"Erro ao extrair Markdown: {str(e)}", "metadata": {}}
    
    def _extract_pptx(self, file_path: Path) -> Dict[str, Any]:
        """Extrai texto de PowerPoint"""
        try:
            prs = Presentation(file_path)
            text_parts = []
            slides_info = []
            
            for i, slide in enumerate(prs.slides):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text.strip())
                
                if slide_text:
                    text_parts.append(f"=== Slide {i+1} ===\n")
                    text_parts.append("\n".join(slide_text))
                    text_parts.append("\n")
                    slides_info.append({"number": i+1, "texts": len(slide_text)})
            
            return {
                "content": "\n".join(text_parts),
                "metadata": {"slides": len(prs.slides)}
            }
        except Exception as e:
            return {"content": f"Erro ao extrair PPTX: {str(e)}", "metadata": {}}
    
    def _extract_txt(self, file_path: Path) -> Dict[str, Any]:
        """Extrai texto de TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "content": content,
                "metadata": {"format": "txt"}
            }
        except Exception as e:
            return {"content": f"Erro ao extrair TXT: {str(e)}", "metadata": {}}
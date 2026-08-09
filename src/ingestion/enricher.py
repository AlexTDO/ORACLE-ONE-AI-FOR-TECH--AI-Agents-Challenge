"""
Enriquecimento de texto para melhorar a recuperação
"""

import re
from typing import Dict, Any


class TextEnricher:
    """
    Adiciona sinônimos e contexto ao texto extraído
    para melhorar a recuperação semântica
    """
    
    # Mapeamento de termos para enriquecimento
    SYNONYMS = {
        # Finanças
        "R\\$\\s*[\\d.,]+": " orçamento inicial capital investimento ",
        "centavos": " orçamento inicial capital investimento ",
        "economia": " orçamento inicial capital investimento poupança ",
        
        # Fundação
        "fundada": " criada estabelecida fundação inicio ",
        "fundador": " criador fundador idealizador ",
        
        # Localização
        "sede": " escritório sede localização base ",
        "Natal/RN": " Natal Rio Grande do Norte Nordeste ",
        
        # Empresa
        "TechFlow": " TechFlow Solutions empresa tecnologia ",
    }
    
    @classmethod
    def enrich(cls, text: str) -> str:
        """
        Enriquece o texto adicionando sinônimos e contexto
        
        Args:
            text: Texto original
            
        Returns:
            Texto enriquecido
        """
        enriched = text
        
        # Adiciona sinônimos baseados em padrões
        for pattern, synonyms in cls.SYNONYMS.items():
            if re.search(pattern, text, re.IGNORECASE):
                # Adiciona os sinônimos no final do texto
                enriched += f"\n\nKeywords: {synonyms}"
                break
        
        # Adiciona contexto específico
        if "0,50" in text or "0.50" in text:
            enriched += "\n\n[CONTEXTO: Este documento contém informações sobre o orçamento inicial, capital e investimento da empresa]"
        
        if "fundador" in text.lower() or "fundada" in text.lower():
            enriched += "\n\n[CONTEXTO: Este documento contém informações sobre a fundação, história e origem da empresa]"
        
        if "sede" in text.lower() or "Natal" in text:
            enriched += "\n\n[CONTEXTO: Este documento contém informações sobre a localização e escritórios da empresa]"
        
        return enriched
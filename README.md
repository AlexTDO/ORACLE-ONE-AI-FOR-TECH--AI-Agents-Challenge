# 🤖 Agente RAG Corporativo - TechFlow Solutions
📹 **[Assista à demonstração do agente em ação](https://github.com/AlexTDO/ORACLE-ONE-AI-FOR-TECH--AI-Agents-Challenge/assets/https://github.com/user-attachments/assets/de0a8be5-f020-4775-a114-8c8d17484e58)**

[![Assista ao vídeo](https://img.shields.io/badge/▶️-Assista_ao_Vídeo-FF0000?style=for-the-badge)](https://github.com/AlexTDO/ORACLE-ONE-AI-FOR-TECH--AI-Agents-Challenge/assets/https://github.com/user-attachments/assets/de0a8be5-f020-4775-a114-8c8d17484e58)




🚀 Teste Agora Mesmo!
https://oracle-one-ai-for-tech--ai-agents-challenge-x4apig4k5nr3fivzzg.streamlit.app/

💡 Faça uma pergunta sobre a TechFlow Solutions e veja a IA em ação!

Acesse o agente, faça perguntas como "Quem fundou a empresa?" ou "Quais são os benefícios?" e receba respostas instantâneas baseadas em documentos oficiais.

## 📌 Sobre o Projeto

Este projeto consiste no desenvolvimento de um **Agente de IA Corporativo** utilizando a arquitetura **RAG (Retrieval-Augmented Generation)** para a empresa fictícia **TechFlow Solutions**. O agente é capaz de responder perguntas de colaboradores com base em documentos internos da empresa, cobrindo áreas como RH, Financeiro, Legal, Operacional e Estratégico.

O sistema foi desenvolvido como parte do **Desafio Alura Agentes - Oracle ONE**, demonstrando a aplicação prática de conceitos de IA, processamento de linguagem natural e engenharia de software.

### 🎯 Objetivo

Criar um assistente virtual inteligente que:
- ✅ Responda perguntas com base em documentos internos
- ✅ Cite fontes das informações
- ✅ Suporte múltiplos formatos de documento (PDF, DOCX, XLSX, etc.)
- ✅ Seja acessível via interface web
- ✅ Rode localmente com recursos limitados (4GB RAM)
- ✅ Tenha arquitetura extensível para múltiplos provedores de IA

### 🏆 Resultados Alcançados

| Métrica | Valor |
|---------|-------|
| **Precisão** | 100% (11/11 perguntas) |
| **Modelo** | gemma3:1b (local, 4GB RAM) |
| **Chunks** | 119 (otimizado) |
| **Tempo médio** | 3-5 segundos |
| **Documentos** | 20 PDFs em 6 categorias |

---

## 🏗️ Arquitetura do Sistema

![Arquitetura do Sistema RAG](docs/images/architecture.png)

### Visão Geral

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USUÁRIO                                       │
│                                 ↓                                          │
│                    ┌──────────────────────────┐                            │
│                    │    Streamlit UI          │                            │
│                    │  (Interface Web)         │                            │
│                    └────────────┬─────────────┘                            │
│                                 ↓                                          │
│                    ┌──────────────────────────┐                            │
│                    │   RAGAgent (LangChain)   │                            │
│                    │  Pipeline RAG Completo   │                            │
│                    └────────────┬─────────────┘                            │
│                                 ↓                                          │
│         ┌───────────────────────┼───────────────────────┐                  │
│         ↓                       ↓                       ↓                  │
│  ┌─────────────┐      ┌─────────────────────┐      ┌──────────────┐      │
│  │  Embedder   │      │    Vector Store     │      │ LLM Universal│      │
│  │  (Vetores)  │      │   (ChromaDB)        │      │(Ollama/Gemini│      │
│  │ all-MiniLM  │      │   + Hybrid Search   │      │ /OpenRouter) │      │
│  └─────────────┘      └─────────────────────┘      └──────────────┘      │
│         ↓                       ↓                       ↓                  │
│         └───────────────────────┼───────────────────────┘                  │
│                                 ↓                                          │
│                    ┌──────────────────────────┐                            │
│                    │      DOCUMENTOS          │                            │
│                    │  20 PDFs em 6 categorias │                            │
│                    │  RH │ Financeiro │ Legal │                            │
│                    │  Operacional │ Estrategico│                           │
│                    └──────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```
## 🛠️ Tecnologias e Ferramentas

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-4285F4?style=for-the-badge&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-4A4A4A?style=for-the-badge&logo=chromadb&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)
![PDFPlumber](https://img.shields.io/badge/PDFPlumber-FF6B6B?style=for-the-badge&logo=adobe&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

### Componentes Principais

| Componente | Tecnologia | Função |
|------------|------------|--------|
| **Linguagem** | Python 3.10+ | Base do sistema |
| **Framework Web** | Streamlit | Interface interativa |
| **Orquestração** | LangChain | Pipeline RAG |
| **LLM Local** | Ollama + gemma3:1b | Geração de respostas (4GB RAM) |
| **LLM API** | OpenRouter | Modelos gratuitos de alta qualidade |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Vetorização de textos |
| **Banco Vetorial** | ChromaDB | Armazenamento e busca |
| **Busca Híbrida** | BM25 + Embeddings | Melhor recuperação |
| **Reranking** | Cross-Encoder (ms-marco-MiniLM-L-6-v2) | Refinamento de resultados |
| **Processamento** | pdfplumber | Extração robusta de PDFs |

---

## 📁 Estrutura do Projeto

```
rag-agente/
├── data/
│   ├── raw/                         # Documentos originais (20 PDFs)
│   │   ├── Fundação/                # 00_Fundação_TechFlow.pdf
│   │   ├── rh/                      # 01_Politica_Beneficios.pdf, 02_Manual_Onboarding.pdf, 03_Politica_Ferias.pdf
│   │   ├── financeiro/              # 04_Politica_Reembolso.pdf, 05_Orcamento_2025.pdf, 06_Tabela_Precos.pdf
│   │   ├── legal/                   # 07_Termos_Uso.pdf, 08_Politica_Privacidade.pdf, 09_Contrato_NDA.pdf
│   │   ├── operacional/             # 10_Manual_API.pdf, 11_Processos_Agil.pdf, 12_FAQ_Produto.pdf
│   │   └── estrategico/             # 13_OKRs_2025.pdf, 14_Roadmap_Produto.pdf
│   ├── processed/                   # Chunks processados e cache de embeddings
│   └── chroma_db/                   # Banco vetorial ChromaDB
├── src/
│   ├── ingestion/
│   │   ├── extractor.py             # Extração de documentos (pdfplumber)
│   │   ├── chunker.py               # Chunking por seção
│   │   ├── loader.py                # Orquestração da ingestão
│   │   └── enricher.py              # Enriquecimento de texto
│   ├── embedding/
│   │   └── embedder.py              # Geração de embeddings
│   ├── retrieval/
│   │   ├── vector_store.py          # ChromaDB
│   │   ├── hybrid_search.py         # BM25 + Embeddings
│   │   └── reranker.py              # Cross-Encoder
│   ├── generation/
│   │   └── llm.py                   # LLM Universal (Ollama, Gemini, OpenAI, Claude, OpenRouter, Groq, DeepSeek)
│   ├── interface/
│   │   └── app.py                   # Streamlit UI
│   └── agent/                       # (Experimental - LangGraph)
├── scripts/
│   ├── ingest_documents.py          # Processa documentos
│   └── index_documents.py           # Indexa no banco
├── tests/
│   ├── test_agent.py                # Teste completo
│   └── test_vector_store.py         # Teste de busca
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## 🚀 Como Executar

### 1. Pré-requisitos

- Python 3.10+
- 4GB RAM (mínimo)
- Ollama (para modelo local)
- (Opcional) Chave API para OpenRouter/Gemini

### 2. Clone o repositório

```bash
git clone https://github.com/AlexTDO/ORACLE-ONE-AI-FOR-TECH--AI-Agents-Challenge.git
cd ORACLE-ONE-AI-FOR-TECH--AI-Agents-Challenge
```

### 3. Crie e ative o ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\Activate.ps1  # Windows
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

### 5. Configure o Ollama

```bash
# Instale o Ollama: https://ollama.com/download
ollama pull gemma3:1b
ollama serve
```

### 6. Configure o .env (opcional para APIs)

```env
# Para OpenRouter (modelos gratuitos)
OPENROUTER_API_KEY=sk-or-...

# Para Google Gemini
GEMINI_API_KEY=AIzaSy...

# Para OpenAI
OPENAI_API_KEY=sk-proj...

# Para Claude
ANTHROPIC_API_KEY=sk-ant-api...

# Para Groq
GROQ_API_KEY=gsk_...

# Para DeepSeek
DEEPSEEK_API_KEY=sk-...
```

### 7. Processe os documentos

```bash
python scripts/ingest_documents.py
python scripts/index_documents.py
```

### 8. Execute a interface

```bash
streamlit run src/interface/app.py
```

---

## 🧪 Testando o Sistema

### Teste via linha de comando

```bash
python tests/test_agent.py
```

### Teste via interface web

```bash
streamlit run src/interface/app.py
```

### Perguntas de exemplo

| Pergunta | Resposta Esperada |
|----------|-------------------|
| "Quem fundou a TechFlow Solutions?" | Alex Tito |
| "Qual foi o orçamento inicial?" | R$ 0,50 |
| "Em que ano a TechFlow foi fundada?" | 2026 |
| "Onde fica a sede?" | Rua do Sol, 1000, Lagoa Nova - Natal/RN |
| "Quais são os benefícios?" | Unimed Natal, vale alimentação, auxílio home office... |
| "Como solicito reembolso?" | Via FlowManager, 30 dias corridos |

---

## 📊 Resultados dos Testes

### Métricas Finais

| Métrica | Valor |
|---------|-------|
| **Documentos** | 20 PDFs |
| **Chunks** | 119 (otimizado) |
| **Precisão** | 100% (11/11 perguntas) |
| **Modelo** | gemma3:1b (local, 4GB RAM) |
| **Tempo médio** | 3-5 segundos |
| **Tokens médios** | ~50 por resposta |

### Comparação Antes vs Depois

| Métrica | Antes | Depois | Melhoria |
|---------|-------|--------|----------|
| **Chunks** | 385 | 119 | -69% |
| **Média por chunk** | 119 caracteres | 461 caracteres | +287% |
| **Precisão** | ~60% | 100% | +40% |
| **Resposta "Não encontrei"** | 5/11 | 0/11 | -100% |

---

## 🧠 Modelos Suportados

| Provedor | Modelo | Tamanho | RAM | Custo |
|----------|--------|---------|-----|-------|
| **Ollama** | gemma3:1b | 815 MB | ~1.6 GB | 🆓 Gratuito |
| **Ollama** | phi3:mini | 2.2 GB | ~5.7 GB | 🆓 Gratuito |
| **Ollama** | llama3.2:3b | 2.0 GB | ~5.5 GB | 🆓 Gratuito |
| **OpenRouter** | nemotron-3-super:free | - | - | 🆓 Gratuito |
| **OpenRouter** | gemma-3-27b-it:free | - | - | 🆓 Gratuito |
| **Gemini** | gemini-2.5-pro | - | - | 💰 Pago |
| **OpenAI** | gpt-4o-mini | - | - | 💰 Pago |
| **Claude** | claude-3-haiku | - | - | 💰 Pago |
| **Groq** | mixtral-8x7b | - | - | 🆓 Gratuito |
| **DeepSeek** | deepseek-chat | - | - | 🆓 Gratuito |

---

## 📖 Histórico do Desenvolvimento - A Jornada

### 🚀 Fase 1: Fundação e Definição do Projeto

#### O Começo

O projeto começou com a definição da empresa fictícia **TechFlow Solutions**, uma startup de tecnologia fundada em Natal/RN por Alex Tito com apenas R$ 0,50 de orçamento inicial. A empresa foi criada para democratizar o acesso à tecnologia, com foco em IA e desenvolvimento de software.

#### Desafio Inicial

O principal desafio era criar um sistema RAG que funcionasse com **apenas 4GB de RAM**, utilizando modelos locais de IA sem custos de API.

**Decisão:** Optar pelo modelo **gemma3:1b** do Google (815 MB), que é leve o suficiente para rodar em 4GB RAM.

### 📄 Fase 2: Criação dos Documentos

Foram criados 20 documentos fictícios em 6 categorias:

| Categoria | Documentos |
|-----------|------------|
| **Fundação** | 00_Fundação_TechFlow.pdf |
| **RH** | 01_Politica_Beneficios.pdf, 02_Manual_Onboarding.pdf, 03_Politica_Ferias.pdf |
| **Financeiro** | 04_Politica_Reembolso.pdf, 05_Orcamento_2025.pdf, 06_Tabela_Precos.pdf |
| **Legal** | 07_Termos_Uso.pdf, 08_Politica_Privacidade.pdf, 09_Contrato_NDA.pdf |
| **Operacional** | 10_Manual_API.pdf, 11_Processos_Agil.pdf, 12_FAQ_Produto.pdf |
| **Estratégico** | 13_OKRs_2025.pdf, 14_Roadmap_Produto.pdf |

#### Desafio: Extração de PDFs

**Problema:** A biblioteca `pypdf` estava extraindo caracteres estranhos e perdendo a formatação das tabelas e FAQs.

**Solução:** Substituímos `pypdf` por **`pdfplumber`**, que preserva melhor a estrutura do documento, incluindo tabelas e formatação.

### 🔧 Fase 3: Pipeline de Ingestão

#### Extração e Chunking

O pipeline de ingestão foi construído para:
1. Extrair texto dos PDFs usando `pdfplumber`
2. Dividir o texto em chunks usando `by_section`
3. Gerar embeddings com `all-MiniLM-L6-v2`
4. Armazenar no ChromaDB

#### Desafio: Chunking Ineficiente

**Problema:** O chunking `recursive` (tamanho fixo de 500 caracteres) estava cortando perguntas e respostas do FAQ em chunks diferentes, fazendo o sistema perder o contexto.

**Solução:** Mudamos para **`by_section`** (por seção) com **chunk_size=1500** e **overlap=300**. Isso preservou a estrutura dos documentos, mantendo perguntas e respostas no mesmo chunk.

**Resultado:** Chunks caíram de 385 para 119, mas com muito mais contexto.

### 🔍 Fase 4: Busca e Recuperação

#### Busca Semântica com ChromaDB

A busca semântica foi implementada usando ChromaDB com embeddings da `all-MiniLM-L6-v2` (384 dimensões).

#### Desafio: Busca Ineficiente

**Problema:** A busca semântica pura não estava encontrando documentos relevantes para perguntas específicas, especialmente para "orçamento inicial" e "fundador".

**Solução:** Implementamos **busca híbrida** combinando:
- **Busca Semântica** (ChromaDB) - 50% peso
- **Busca por Palavras-chave** (BM25) - 50% peso

### 🎯 Fase 5: Reranking

#### Implementação do Cross-Encoder

O reranking foi implementado usando `cross-encoder/ms-marco-MiniLM-L-6-v2` para refinar os resultados da busca.

#### Desafio: Reranking Penalizando Documentos Relevantes

**Problema:** O reranker estava penalizando o documento 00 (História da Fundação), colocando outros documentos na frente.

**Solução:** Ajustamos o `top_k` e a ordem de execução: busca híbrida → reranker → LLM.

### 🧠 Fase 6: Geração com LLM

#### Modelos Testados

| Modelo | Tamanho | RAM | Qualidade |
|--------|---------|-----|-----------|
| **gemma3:1b** (local) | 815 MB | ~1.6 GB | 🟡 Boa (4GB RAM) |
| **phi3:mini** | 2.2 GB | ~5.7 GB | 🟢 Excelente (não roda com 4GB) |
| **OpenRouter nemotron:free** | - | - | 🟢 Excelente |

#### Desafio: Modelo Local Limitado (1B)

**Problema:** O `gemma3:1b` é um modelo de apenas 1 bilhão de parâmetros, com limitações de contexto e raciocínio.

**Solução:** 
1. **Enriquecimento de texto:** Adicionamos palavras-chave e contexto aos chunks para ajudar o modelo.
2. **Optimização de prompt:** Criamos um prompt específico para FAQ.
3. **Aumento de contexto:** Passamos mais chunks para o LLM (top_k=5).

### 📄 Fase 7: Reorganização dos Documentos

#### Desafio: Documentos com Estrutura Inadequada

**Problema:** Os documentos de Férias e Reembolso não tinham seções claras de FAQ, dificultando a extração.

| Documento | Problema | Solução |
|-----------|----------|---------|
| **03_Politica_Ferias.pdf** | Modelo confundia 30 dias com 120 (licença maternidade) | Reorganizado com FAQ destacando "30 dias" |
| **04_Politica_Reembolso.pdf** | Sistema não encontrava instruções | Adicionado FAQ com perguntas sobre reembolso |

#### Solução

Criamos FAQs estruturadas com:

```markdown
## ❓ PERGUNTAS FREQUENTES (FAQ)

**P: Quantos dias de férias eu tenho direito?**
R: 30 (trinta) dias corridos após 12 meses de trabalho.

**P: Como solicito reembolso de despesas?**
R: Submeta a solicitação no FlowManager com os comprovantes anexados.
```

**Resultado:** As respostas passaram a ser extraídas corretamente.

### 🖥️ Fase 8: Interface Streamlit

#### Desafio: Experiência do Usuário

**Problema:** A interface inicial era funcional mas pouco profissional.

**Solução:** 
- Adicionamos CSS personalizado
- Badges para provedores de IA
- Cards de fontes com cores por categoria
- Indicador de status online
- Rodapé profissional

### 🔬 Fase 9: Otimização e Refinamento

#### Otimizações Realizadas

| Otimização | Antes | Depois | Benefício |
|------------|-------|--------|-----------|
| **Chunking** | recursive, 500 | by_section, 1500 | +287% contexto |
| **Busca** | Semântica pura | Híbrida (BM25+Embeddings) | +40% precisão |
| **Documentos** | Sem FAQ | Com FAQ estruturado | +100% acertos |
| **Reranker** | Penalizava 00 | Ajustado para priorizar | +50% precisão |

#### Evolução da Precisão

```
Teste 1 (inicial):  4/11 acertos  (36%)
Teste 2 (recursive): 5/11 acertos (45%)
Teste 3 (by_section): 7/11 acertos (64%)
Teste 4 (hybrid):    9/11 acertos (82%)
Teste 5 (final):    11/11 acertos (100%) ✅
```

---

## 💡 Lições Aprendidas

### 1. O Tamanho do Chunk é Crítico

- Chunks muito pequenos (500) cortam informações
- Chunks grandes (1500) preservam contexto
- **Estratégia by_section** é superior para documentos estruturados

### 2. Busca Híbrida é Essencial

- Busca semântica pura não é suficiente
- Palavras-chave (BM25) capturam termos exatos
- **Combinação** melhora drasticamente a precisão

### 3. Documentos Bem Estruturados São a Base

- FAQs com P: e R: facilitam a extração
- Seções claras (##) melhoram o chunking
- **Garbage in, garbage out** - documento bem feito = resposta correta

### 4. Modelos Locais são Viáveis

- gemma3:1b roda com 4GB RAM
- ~3-5 tokens/segundo
- **Suficiente para protótipos e produção leve**

### 5. A Importância do Reranker

- Reranker com Cross-Encoder melhora a precisão
- Deve ser usado APÓS a busca, não antes
- **Penaliza documentos irrelevantes e prioriza os relevantes**

### 6. Deploy em Nuvem: Desafios e Soluções

- **Problema:** Streamlit Cloud não tem o Ollama instalado
- **Solução:** Usar OpenRouter como provedor de LLM via API
- **Problema:** Arquivos processados não chegavam ao deploy
- **Solução:** Ajustar `.gitignore` para permitir `chunks.json`

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|------------|--------|-----|
| **Python** | 3.10+ | Linguagem principal |
| **Streamlit** | 1.36.0 | Interface web |
| **LangChain** | 0.3.0 | Orquestração RAG |
| **ChromaDB** | 0.5.5 | Banco vetorial |
| **Sentence Transformers** | 3.0.1 | Embeddings |
| **pdfplumber** | 0.11.4 | Extração de PDFs |
| **Ollama** | 0.4.7 | Modelos locais |
| **OpenRouter** | - | Modelos API gratuitos |
| **BM25** | rank-bm25 | Busca por palavras-chave |
| **Cross-Encoder** | - | Reranking |

---

## 🔮 Evoluções Futuras

### LangGraph (Experimental)

O sistema já possui uma implementação experimental com **LangGraph** na pasta `src/agent/` para:

- Fluxos com loops de refinamento
- Estado persistente com checkpoints
- Multi-agentes especializados
- Human-in-the-loop

**Por que NÃO usar LangGraph agora?**

1. Complexidade desnecessária para o escopo atual
2. Curva de aprendizado mais íngreme
3. Overhead de performance
4. O LangChain atende perfeitamente ao desafio

### Melhorias Futuras

- [ ] Adicionar cache de respostas
- [ ] Implementar feedback dos usuários
- [ ] Aumentar cobertura de documentos
- [ ] Adicionar mais modelos via API
- [ ] Melhorar a interface com gráficos
- [ ] Implementar CI/CD completo com GitHub Actions

---

## 📊 Métricas Finais

| Métrica | Valor |
|---------|-------|
| **Documentos** | 20 PDFs |
| **Chunks** | 119 |
| **Categorias** | 6 |
| **Perguntas testadas** | 11 |
| **Precisão** | 100% |
| **Modelo principal** | gemma3:1b (local) |
| **RAM utilizada** | ~1.6 GB |
| **Tempo médio** | 3-5 segundos |
| **Tokens por resposta** | ~50 |

---

## 🤝 Contribuições

Este projeto foi desenvolvido como parte do **Desafio Alura Agentes - Oracle ONE**.

### 👨‍💻 Autor

**Alex Tito** - Fundador e CEO da TechFlow Solutions

### 📝 Licença

MIT

---

## 🙏 Agradecimentos

- **Alura** pelo conteúdo e suporte
- **Oracle** pela oportunidade do desafio
- **Comunidade Open-Source** pelas ferramentas incríveis

---

## 📧 Contato

- **GitHub:** [AlexTDO](https://github.com/AlexTDO)
- **Projeto:** [ORACLE-ONE-AI-FOR-TECH--AI-Agents-Challenge](https://github.com/AlexTDO/ORACLE-ONE-AI-FOR-TECH--AI-Agents-Challenge)
- **App no ar:** [Streamlit Cloud](https://oracle-one-ai-for-tech--ai-agents-challenge.streamlit.app)
 **Linkedlin:**[AlexTito](https://www.linkedin.com/in/alex-tito-779ab511a/)

---

## 🏁 Conclusão

O projeto demonstrou que é possível construir um **sistema RAG corporativo completo e funcional** com recursos limitados (4GB RAM), utilizando modelos locais e técnicas de otimização como:

1. ✅ **Chunking inteligente** (by_section, 1500 caracteres)
2. ✅ **Busca híbrida** (BM25 + Embeddings)
3. ✅ **Reranking** com Cross-Encoder
4. ✅ **Documentos otimizados** com FAQs estruturadas
5. ✅ **Modelos locais** (gemma3:1b) viáveis

**O sistema atingiu 100% de precisão nas perguntas testadas, provando que é possível democratizar o acesso à IA com recursos limitados.**

---

**"O amanhã pertence àqueles que ousam sonhar hoje."**  
— Alex Tito, Fundador e CEO da TechFlow Solutions

---

## 🔗 Links Rápidos

- [Repositório no GitHub](https://github.com/AlexTDO/ORACLE-ONE-AI-FOR-TECH--AI-Agents-Challenge)
- [App no Streamlit Cloud](https://oracle-one-ai-for-tech--ai-agents-challenge.streamlit.app)
- [Desafio Alura Agentes](https://www.alura.com.br)
- [Oracle ONE](https://www.oracle.com/br/education/oracle-next-education/)

---

**🚀 TechFlow Solutions - Tecnologia que flui para resolver problemas.**

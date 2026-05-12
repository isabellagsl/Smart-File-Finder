#  SmartFile Finder Pro

**SmartFile Finder Pro** utiliza Processamento de Linguagem Natural (NLP) para transformar a maneira como pesquisamos em documentos. Ao contrário dos filtros tradicionais baseados em correspondência exata de palavras, este sistema utiliza **Embeddings** para compreender o contexto e o significado real da sua pesquisa.

##  Tecnologias Utilizadas

- **Python 3.9+**
- **Interface Web**: `streamlit`
- **NLP & Embeddings**: `sentence-transformers`, `transformers`
- **Processamento de Documentos**: `PyMuPDF` (PDF), `python-docx` (DOCX), `langchain-text-splitters`
- **Manipulação de Dados**: `pandas`

##  Funcionalidades do MVP

- **Suporte Multi-Formato**: Extração de texto de arquivos PDF, DOCX e TXT.
- **Busca Contextual**: Encontra informações relevantes pelo significado e não apenas por palavras exatas (ex: retorna resultados de "receitas" quando se pesquisa por "comida").
- **Similaridade de Cosseno**: Ranquia os documentos de forma inteligente, ordenando do mais relevante para o menos relevante com barra de progresso visual.
- **Interface Web**: Uma interface simples, intuitiva e rápida, construída inteiramente com Streamlit.
- **Processamento em Tempo Real**: Gera embeddings instantaneamente e divide o texto em chunks semânticos para indexação.

##  Arquitetura Técnica

O projeto tira proveito do modelo pré-treinado `'paraphrase-multilingual-mpnet-base-v2'` da biblioteca Sentence-Transformers. Este modelo oferece alta precisão semântica para múltiplos idiomas. Para o processamento de texto, emprega-se o `RecursiveCharacterTextSplitter` alinhado com o tokenizer do modelo, garantindo o melhor aproveitamento do contexto.

###  Fluxo de Dados

1. **Input**: O usuário faz o upload de múltiplos arquivos (PDF, DOCX, TXT) através da barra lateral.
2. **Extração e Chunking**: O texto é extraído dos documentos e dividido em partes menores (chunks de 500 tokens) mantendo a estrutura dos parágrafos.
3. **Vetorização**: O modelo converte cada chunk de texto num vetor matemático de embeddings.
4. **Query**: A pesquisa do usuário é convertida simultaneamente para o mesmo espaço vetorial.
5. **Match**: O sistema calcula a Similaridade de Cosseno para encontrar as correspondências contextuais mais próximas, retornando os trechos exatos com pontuação de relevância.

##  Como Executar

Para testar este MVP na sua máquina local, siga os passos abaixo:

### 1. Clone o repositório
```bash
git clone https://github.com/isabellagsl/Smart-File-Finder.git
cd Smart-File-Finder
```

### 2. Crie e ative um ambiente virtual (Recomendado)
```bash
python -m venv venv

# No Linux / macOS:
source venv/bin/activate

# No Windows:
venv\Scripts\activate
```

### 3. Instale as dependências
O projeto já conta com um arquivo `requirements.txt` gerado com todas as dependências (como `streamlit`, `sentence-transformers`, `PyMuPDF`, `python-docx`, `langchain-text-splitters`, etc). Instale com:
```bash
pip install -r requirements.txt
```

### 4. Inicie a aplicação Streamlit
```bash
streamlit run app.py
```
*(Nota: Certifique-se de ter o arquivo principal Python criado, como `app.py`, para iniciar o servidor do Streamlit).*

---
Desenvolvido por isabella Gonçalves, projeto pessoal de estudo.
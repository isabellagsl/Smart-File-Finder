#  Smart File Finder

**Smart File Finder** utiliza Processamento de Linguagem Natural (NLP) para transformar a maneira como pesquisamos em documentos. Ao contrário dos filtros tradicionais baseados em correspondência exata de palavras, este sistema utiliza **Embeddings** para compreender o contexto e o significado real da sua pesquisa.

##  Tecnologias Utilizadas

- **Python 3.9+**
- **NLP & Embeddings**: `sentence-transformers`
- **Interface Web**: `streamlit`

##  Funcionalidades do MVP

- **Busca Contextual**: Encontra informações relevantes pelo significado e não apenas por palavras exatas (ex: retorna resultados de "receitas" quando se pesquisa por "comida").
- **Similaridade de Cosseno**: Ranquia os documentos de forma inteligente, ordenando do mais relevante para o menos relevante.
- **Interface Web**: Uma interface simples, intuitiva e rápida, construída inteiramente com Streamlit.
- **Processamento em Tempo Real**: Gera embeddings instantaneamente, sendo ideal para explorar pequenas e médias coleções de dados de forma ágil.

##  Arquitetura Técnica

O projeto tira proveito do modelo pré-treinado 'paraphrase-multilingual-MiniLM-L12-v2' da biblioteca Sentence-Transformers. Este modelo foi escolhido por ser altamente otimizado para velocidade e eficiência, sendo perfeito para rodar localmente em ambientes de MVP, mesmo sem o uso de GPUs dedicadas.

###  Fluxo de Dados

1. **Input**: O utilizador carrega ficheiros de texto ou insere uma lista de textos através da interface.
2. **Vetorização**: O modelo converte cada documento (ou linha de texto) num vetor matemático de 384 dimensões.
3. **Query**: A pesquisa (pergunta) do utilizador é convertida simultaneamente para o mesmo espaço vetorial.
4. **Match**: O sistema calcula o ângulo entre os vetores (Similaridade de Cosseno) para encontrar as correspondências contextuais mais próximas, retornando os melhores resultados.

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
Crie um arquivo `requirements.txt` (se ainda não existir) com `streamlit`, `sentence-transformers` e `pandas`, e instale:
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
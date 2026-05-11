import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# 1. Configuração Inicial
st.set_page_config(page_title="SmartFile Finder Pro", page_icon="📂", layout="wide")
st.title("SmartFile Finder Pro 📂")
st.markdown("Busque **dentro** dos seus documentos usando Inteligência Artificial.")

# 2. Carregamento do Modelo
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="cpu")

model = load_model()

# 3. Funções de Extração de Texto
def extract_text(file):
    text = ""
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    elif file_extension == 'docx':
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file_extension == 'txt':
        text = file.read().decode("utf-8")
    
    return text

# 4. Interface de Upload
uploaded_files = st.sidebar.file_uploader(
    "Suba seus arquivos (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    all_chunks = []
    metadata = []

    # Processamento e Chunking
    # Usamos o tokenizer padrão da HuggingFace para manter o texto alinhado com o que o modelo entende (tokens)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=500,  # O limite do modelo mpnet é geralmente 512 tokens
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    with st.spinner("Processando e indexando arquivos..."):
        for uploaded_file in uploaded_files:
            content = extract_text(uploaded_file)
            chunks = text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "nome": uploaded_file.name,
                    "chunk_id": i
                })

        # Geração de Embeddings
        embeddings_base = model.encode(all_chunks, convert_to_tensor=True)

    # 5. Busca Semântica
    query = st.text_input("O que você deseja encontrar dentro dos arquivos?", placeholder="Ex: cláusula de rescisão, meta de faturamento...")

    if query:
        embedding_busca = model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embedding_busca, embeddings_base)[0]

        # Criar DataFrame de resultados
        resultados = pd.DataFrame({
            "Arquivo": [m["nome"] for m in metadata],
            "Trecho": all_chunks,
            "Score": cosine_scores.tolist()
        }).sort_values(by="Score", ascending=False)

        st.subheader("Resultados Relevantes:")
        
        # Filtro de relevância
        top_resultados = resultados[resultados["Score"] > 0.30].head(5)

        if not top_resultados.empty:
            for _, row in top_resultados.iterrows():
                with st.expander(f" {row['Arquivo']} (Score: {row['Score']:.2f})"):
                    st.write(f"\"{row['Trecho']}...\"")
                    st.progress(float(row['Score']))
        else:
            st.warning("Nenhum trecho relevante encontrado.")
else:
    st.info("Comece fazendo o upload de alguns arquivos na barra lateral.")
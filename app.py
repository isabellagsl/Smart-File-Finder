import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# 1. Configuração inicial da página e título
st.set_page_config(page_title="SmartFile Finder", page_icon="📂")

st.title("SmartFile Finder 📂")
st.markdown("Busque arquivos pelo **significado**, não apenas pelo nome!")

# 2. Carregamento do modelo de linguagem
# Utilização de cache para evitar o recarregamento do modelo a cada atualização da interface.
@st.cache_resource
def load_model():
    # O modelo multilingue foi selecionado para garantir suporte otimizado ao idioma Português.
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

model = load_model()

# 3. Base de dados simulada para testes
arquivos_exemplo = [
    "Manual de instruções da impressora térmica",
    "Relatório financeiro do segundo trimestre de 2023",
    "Receita de bolo de cenoura com cobertura de chocolate",
    "Contrato de prestação de serviços de limpeza",
    "Guia de boas práticas para desenvolvimento em Python",
    "Documento de identidade e CPF digital",
    "Planilha de gastos mensais e orçamento doméstico"
]

# 4. Geração dos embeddings da base de dados
# O processamento em cache evita o recálculo dos vetores a cada nova requisição.
@st.cache_data
def get_database_embeddings(lista_arquivos):
    return model.encode(lista_arquivos, convert_to_tensor=True)

embeddings_base = get_database_embeddings(arquivos_exemplo)

# 5. Campo de entrada para a busca
query = st.text_input("O que você está procurando?", placeholder="Ex: manual, finanças, comida...")

if query:
    # Conversão da string de busca em representação vetorial.
    embedding_busca = model.encode(query, convert_to_tensor=True)

    # Comparação vetorial entre a busca e a base de dados.
    # O cálculo de similaridade de cosseno define a relevância com base na proximidade dos vetores.
    cosine_scores = util.cos_sim(embedding_busca, embeddings_base)[0]

    # Organização dos resultados em ordem decrescente de pontuação (Score).
    resultados = pd.DataFrame({
        "Arquivo": arquivos_exemplo,
        "Score": cosine_scores.tolist()
    }).sort_values(by="Score", ascending=False)

    # Exibição dos resultados encontrados
    st.subheader("Resultados encontrados:")
    
    for index, row in resultados.iterrows():
        score = row['Score']
        
        # Atribuição de indicadores visuais baseados no nível de confiança do resultado.
        if score > 0.65:
            confianca = "✅ Alta"
            cor = "green"
        elif score > 0.35:
            confianca = "⚠️ Média"
            cor = "orange"
        else:
            confianca = "❌ Baixa"
            cor = "red"

        # Filtro opcional para ocultar resultados com relevância considerada insuficiente (< 20%).
        if score > 0.20:
            st.write(f"**{row['Arquivo']}**")
            st.caption(f"Relevância: :{cor}[{score:.2f}] | Confiança: {confianca}")
            st.divider()
        
    # Aviso emitido caso nenhum documento atinja o critério mínimo de relevância.
    if resultados['Score'].max() < 0.20:
        st.warning("Nenhum arquivo com relevância mínima encontrado para os termos pesquisados.")

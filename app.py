import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# 1. SETUP (App Title and Icon)
st.set_page_config(page_title="PlantCare Assistant", page_icon="🌿")
st.title("🌿 Indoor Houseplant Care Assistant")
st.subheader("A simple guide for happy plants")
st.divider() 

# 2. DATA: 10 Specific Plant Care Documents
with open("documents.txt","r") as file:
    text=file.read()
    DOCUMENTS = text.split("\n\n")

# 3. SETTINGS: User Controls in the Sidebar
st.sidebar.header("⚙️ RAG Configuration")

c_size = st.sidebar.number_input("Chunk Size", min_value=50, max_value=1000, value=500)
c_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=200, value=50)

st.sidebar.divider()
st.sidebar.info("The system uses these values to process the knowledge base in real-time.")
top_k = st.sidebar.slider("Show top K results",1,5,3)

# 4. THE BRAIN
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def build_vectorstore(chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents(DOCUMENTS)
    vectorstore = Chroma.from_documents(docs, load_embeddings())
    return vectorstore, len(docs)

with st.spinner("Building index..."):
    vectorstore, chunk_count = build_vectorstore(c_size, c_overlap)

# 5. THE SEARCH: 
query = st.text_input("What is your topic of choice?", placeholder="e.g., Light, Water, or Pests")

if query:
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    st.write("### Found Information:")
    for i, (result,score) in enumerate(results,1):
        with st.container(border=True):
            st.caption(f"match: #{i} - distance: {score:.3f}")
            st.write(result.page_content)

# 6. SYSTEM STATS
st.divider()
st.write(f"**Number of chunks:** {chunk_count}")

import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from datetime import datetime

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("🔴 **Error**: Gemini API Key not found. Please add it to your Streamlit secrets (`.streamlit/secrets.toml`).")
    st.stop()

@st.cache_resource
def load_models_and_data():
    """Loads embedding model and FAISS vector store."""
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    index_path = "./vector_store/docs.index"
    docs_path = "./vector_store/docs.pkl"
    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        return None, None, None
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    return embedding_model, index, docs

embedding_model, index, docs = load_models_and_data()
gemini_model = genai.GenerativeModel('gemini-1.5-flash')



def transform_query(chat_history, latest_question):
    """
    Uses Gemini to transform a follow-up question into a standalone query
    based on the chat history.
    """
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    prompt = f"""
    Based on the chat history below, rewrite the user's latest question to be a clear, standalone question that can be understood without the context of the conversation.

    Chat History:
    {history_str}

    Latest User Question: "{latest_question}"

    Standalone Question:
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def get_context(question, top_k=5):
    """Retrieves context from the vector store."""
    query_embedding = embedding_model.encode([question]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [docs[i].page_content for i in indices[0]]
    retrieved_metadata = [docs[i].metadata for i in indices[0]]
    return retrieved_docs, retrieved_metadata

def generate_answer_with_gemini(chat_history, context_docs, latest_question):
    """Generates a contextual answer using Gemini."""
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    context_str = "\n\n---\n\n".join(context_docs)
    
    prompt = f"""
    You are a helpful assistant. Answer the user's question based on the provided chat history and the retrieved document context.
    - Your answer must be based **only** on the information in the `Retrieved Document Context`.
    - Be conversational and refer to the `Chat History` to understand the flow of the conversation.
    - If the context does not contain the answer, state that the information is not available in the provided documents.
    
    Chat History:
    {history_str}

    Retrieved Document Context:
    {context_str}

    Latest User Question: "{latest_question}"

    Answer:
    """
    response = gemini_model.generate_content(prompt)
    return response.text



st.title("🧠 Chatbot — Retrieval-Augmented Generation")


if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = None


with st.sidebar:
    st.header("Histori Obrolan")
    if st.button("➕ Mulai Obrolan Baru"):
        session_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        st.session_state.chat_sessions[session_name] = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait dokumen Anda?"}]
        st.session_state.current_session = session_name
        st.rerun()

    if st.session_state.chat_sessions:
        for session in sorted(st.session_state.chat_sessions.keys(), reverse=True):
            if st.button(session, key=session, use_container_width=True):
                st.session_state.current_session = session
                st.rerun()


if embedding_model is None:
    st.error("Vector store belum dibuat! 🔴 Jalankan `build_index.py` terlebih dahulu.")
    st.stop()

if not st.session_state.current_session:
    st.info("Mulai obrolan baru atau pilih salah satu dari histori di sidebar.")
    st.stop()


current_messages = st.session_state.chat_sessions[st.session_state.current_session]
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen Anda..."):
    
    current_messages.append({"role": "user", "content": prompt})

    
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban... 🧠"):
            
            standalone_query = transform_query(current_messages[:-1], prompt)
            st.info(f"🔎 Mencari dokumen untuk: *{standalone_query}*")

            
            context_docs, context_metadata = get_context(standalone_query)

            
            answer = generate_answer_with_gemini(current_messages, context_docs, prompt)
            
            
            st.markdown(answer)

            
            with st.expander("📚 Lihat Sumber"):
                for i, meta in enumerate(context_metadata):
                    source_file = os.path.basename(meta['source'])
                    st.write(f"**Sumber {i+1}:** {source_file}, Halaman: {meta['page'] + 1}")

    
    current_messages.append({"role": "assistant", "content": answer})
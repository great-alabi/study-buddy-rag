import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai
from pypdf import PdfReader
from authlib.integrations.requests_client import OAuth2Session

# Safely load local .env if it exists (for local development)
try:
    load_dotenv()
except ImportError:
    pass

client_id = st.secrets["auth"]["auth0"]["client_id"]
client_secret = st.secrets["auth"]["auth0"]["client_secret"]
server_metadata_url = st.secrets["auth"]["server_metadata_url"]
auth0_domain = server_metadata_url.split("/")[2]

# --- AUTHENTICATION CHECK ---
if not st.user.is_logged_in:
    st.title("Welcome to Study Buddy RAG 📚")
    st.write("Please log in using your preferred platform to continue.")

    if st.button("Log in with Auth0", type="primary"):
        st.login("auth0")
    
    st.stop()  # Halts execution here until the user logs in

# --- LOGGED-IN USER VIEW ---
st.sidebar.markdown(f"**Welcome, {st.user.name}!**")
st.sidebar.text(st.user.email)

if st.sidebar.button("Log out"):
    st.logout()

# Safe key retrieval for both Local & Cloud environments
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

# Page config
st.set_page_config(page_title="My Study Buddy", layout="centered")
st.title("📚 My Study Buddy")

# Initialize clients (cached so they don't reload on every click)
@st.cache_resource
def init_rag_clients():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("research-papers-index")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    return index, model, ai_client

index, model, ai_client = init_rag_clients()

# Custom CSS styling for standard cards and active green source highlighting
st.markdown("""
    <style>
    .file-card {
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
        margin-bottom: 8px;
    }
    .file-card-active {
        padding: 8px 12px;
        border-radius: 6px;
        border: 2px solid #28a745;
        background-color: #e8f5e9;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for tracking active sources used in the latest response
if "active_sources" not in st.session_state:
    st.session_state.active_sources = []

# Sidebar for uploading documents and viewing library
with st.sidebar:
    st.header("📚 Document Library")
    
    papers_folder = "research_papers"
    os.makedirs(papers_folder, exist_ok=True)
    existing_files = os.listdir(papers_folder)
    
    if existing_files:
        st.subheader("Existing Files")
        for file in existing_files:
            is_active = file in st.session_state.active_sources
            card_class = "file-card-active" if is_active else "file-card"
            
            with st.container():
                st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                col1, col2 = st.columns([0.75, 0.25])
                with col1:
                    prefix_icon = "🟢 📄" if is_active else "📄"
                    st.markdown(f"{prefix_icon} **{file}**")
                with col2:
                    with st.popover("⋮"):
                        st.write(f"Manage **{file}**")
                        if st.button("Delete File", key=f"del_{file}", type="primary"):
                            try:
                                file_path = os.path.join(papers_folder, file)
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                
                                index.delete(filter={"source_paper": {"$eq": file}})
                                potential_ids = [f"{file}-{idx}" for idx in range(500)]
                                index.delete(ids=potential_ids)
                                    
                                if file in st.session_state.active_sources:
                                    st.session_state.active_sources.remove(file)
                                    
                                st.success(f"Permanently purged {file}!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting file: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No documents in local folder yet.")
    
    st.divider()
    
    st.header("Upload New Documents")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process & Index Documents"):
            with st.spinner("Processing and indexing documents..."):
                for uploaded_file in uploaded_files:
                    bytes_data = uploaded_file.read()
                    save_path = os.path.join(papers_folder, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(bytes_data)
                    
                    full_text = ""
                    if uploaded_file.name.endswith(".pdf"):
                        reader = PdfReader(save_path)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                full_text += text + "\n"
                    else:
                        full_text = bytes_data.decode("utf-8", errors="ignore")
                    
                    chunk_size = 500
                    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
                    
                    vectors_to_upsert = []
                    for idx, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50:  
                            chunk_vector = model.encode(chunk).tolist()
                            vector_id = f"{uploaded_file.name}-{idx}"
                            vectors_to_upsert.append({
                                "id": vector_id,
                                "values": chunk_vector,
                                "metadata": {
                                    "source_paper": uploaded_file.name,
                                    "text": chunk
                                }
                            })
                    
                    if vectors_to_upsert:
                        index.upsert(vectors=vectors_to_upsert)
                        st.success(f"Indexed {uploaded_file.name} ({len(vectors_to_upsert)} chunks).")
                    else:
                        st.warning(f"Could not extract enough text from {uploaded_file.name}.")
                
                st.rerun()

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about your library..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching library..."):
            try:
                query_vector = model.encode(user_query).tolist()
                search_results = index.query(
                    vector=query_vector, 
                    top_k=4, 
                    include_metadata=True,
                    timeout=15 
                )
                
                context_chunks = []
                retrieved_sources = set()
                
                for match in search_results.get('matches', []):
                    source = match.get('metadata', {}).get('source_paper', 'Unknown')
                    text = match.get('metadata', {}).get('text', '')
                    context_chunks.append(f"Source: {source}\nSnippet: {text}")
                    retrieved_sources.add(source)
                    
                combined_context = "\n\n---\n\n".join(context_chunks)
                
                prompt = f"""
                You are an expert research assistant. Answer the user's question using only the context chunks below. 
                Explicitly mention the source document filename when referencing information from that document.
                
                Context Chunks:
                {combined_context}
                
                User Question: {user_query}
                """
                
                response = ai_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                answer_text = response.text
                
                st.session_state.active_sources = list(retrieved_sources)
                
                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Connection error while searching Pinecone: {e}")
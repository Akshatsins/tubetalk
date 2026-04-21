import streamlit as st
import time
from transcribe import get_transcript, delete_file

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TubeTalk Pro",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STYLES ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(-45deg, #000000, #0a0a0a, #1a0505, #000000);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #e0e0e0;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stTextInput > div > div, .stChatInput > div > div {
        background-color: #1a1a1a !important;
        border: 1px solid #333;
        color: white;
        border-radius: 10px;
    }
    .stTextInput > div > div:focus-within, .stChatInput > div > div:focus-within {
        border-color: #FF0000;
        box-shadow: 0 0 10px rgba(255, 0, 0, 0.2);
    }

    .stButton button {
        background: linear-gradient(90deg, #CC0000 0%, #FF0000 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(255, 0, 0, 0.4);
    }

    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    div[data-testid="stChatMessageContent"] {
        color: #e0e0e0;
    }

    .timestamp-tag {
        background: rgba(255, 0, 0, 0.15);
        color: #FF5555;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        margin-top: 8px;
        margin-right: 5px;
        border: 1px solid rgba(255, 0, 0, 0.3);
    }

    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- CONFIG ---
GROQ_MODEL = "llama3-8b-8192"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- STATE ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- HELPERS ---
def get_api_key():
    """Get Groq API key from secrets or session state."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return st.session_state.get("groq_api_key", "")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def create_vector_db(segments):
    documents = [
        Document(page_content=s['text'], metadata={"start": s['start']})
        for s in segments
    ]
    embeddings = get_embeddings()
    return FAISS.from_documents(documents, embeddings)

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ▶️ TubeTalk Pro")

    # API Key input (for when not using secrets)
    if not get_api_key():
        st.markdown("#### 🔑 Groq API Key")
        key_input = st.text_input("Enter your Groq API Key", type="password", placeholder="gsk_...")
        if key_input:
            st.session_state.groq_api_key = key_input
            st.success("Key saved!")
        st.markdown("[Get free key →](https://console.groq.com)", unsafe_allow_html=True)
        st.divider()

    if st.button("New Session", type="primary"):
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        st.session_state.video_url = ""
        st.rerun()

    if st.session_state.vector_db:
        st.markdown("#### 📊 Stats")
        st.write(f"Model: `{GROQ_MODEL}`")
        st.write(f"Embeddings: `{EMBEDDING_MODEL}`")
        try:
            st.write(f"Vectors: `{st.session_state.vector_db.index.ntotal}`")
        except Exception:
            pass

# --- MAIN UI ---
if not st.session_state.vector_db:
    # LANDING PAGE
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<h1 style='text-align: center; font-size: 60px; font-weight: 800;"
            "background: -webkit-linear-gradient(#fff, #666);"
            "-webkit-background-clip: text; -webkit-text-fill-color: transparent;'>TubeTalk</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color: #888; font-size: 18px; margin-bottom: 30px;'>"
            "The AI Video Intelligence Engine</p>",
            unsafe_allow_html=True
        )

        api_key = get_api_key()
        if not api_key:
            st.warning("⚠️ Add your Groq API key in the sidebar to get started.")
        else:
            url_input = st.text_input(
                "YouTube URL",
                placeholder="Paste link here...",
                label_visibility="collapsed"
            )

            if st.button("Analyze Video", disabled=not api_key):
                if url_input:
                    with st.status("Processing video...", expanded=True) as status:
                        try:
                            st.write("📋 Fetching transcript from YouTube...")
                            segments = get_transcript(url_input)

                            st.write("🧠 Building knowledge base...")
                            st.session_state.vector_db = create_vector_db(segments)
                            st.session_state.video_url = url_input
                            status.update(label="✅ Ready to chat!", state="complete", expanded=False)
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            status.update(label="❌ Error", state="error")
                            st.error(f"Error: {e}")

else:
    # WORKSPACE
    col_video, col_chat = st.columns([1.5, 2], gap="large")

    with col_video:
        st.markdown("""
        <div style="border-radius: 15px; overflow: hidden;
        box-shadow: 0 0 20px rgba(255,0,0,0.3);
        border: 1px solid #222; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        st.video(st.session_state.video_url)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chat:
        st.markdown("### 💬 Chat Stream")

        # Render history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask about the video..."):
            api_key = get_api_key()
            if not api_key:
                st.error("No API key found. Please add it in the sidebar.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    ph = st.empty()

                    # Retrieval
                    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 4})
                    docs = retriever.invoke(prompt)
                    context = "\n".join([d.page_content for d in docs])

                    # Groq streaming
                    llm = ChatGroq(
                        model=GROQ_MODEL,
                        api_key=api_key,
                        streaming=True
                    )

                    full_res = ""
                    try:
                        system_prompt = (
                            "You are a helpful assistant that answers questions about YouTube videos. "
                            "Use the provided transcript context to answer accurately. "
                            "Be concise and direct."
                        )
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                        ]

                        for chunk in llm.stream(messages):
                            full_res += chunk.content
                            ph.markdown(full_res + " 🔴")

                        # Timestamps
                        sources_html = "<br><div style='margin-top:15px;'>"
                        for d in docs:
                            ts = format_time(d.metadata['start'])
                            sources_html += f"<span class='timestamp-tag'>▶️ {ts}</span>"
                        sources_html += "</div>"

                        ph.markdown(full_res + sources_html, unsafe_allow_html=True)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": full_res + sources_html
                        })

                    except Exception as e:
                        st.error(f"Groq Error: {e}")

import streamlit as st
import time
from transcribe import get_transcript

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
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    :root {
        --bg-base:      #080808;
        --bg-panel:     #0f0f0f;
        --bg-card:      #161616;
        --bg-hover:     #1e1e1e;
        --accent:       #ff2d2d;
        --accent-dim:   rgba(255, 45, 45, 0.12);
        --accent-glow:  rgba(255, 45, 45, 0.25);
        --border:       rgba(255,255,255,0.06);
        --border-light: rgba(255,255,255,0.10);
        --text-primary: #f0f0f0;
        --text-secondary: #888;
        --text-muted:   #4a4a4a;
        --radius-sm: 8px;
        --radius-md: 14px;
        --radius-lg: 20px;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; font-size: 15px; }

    .stApp { background-color: var(--bg-base); color: var(--text-primary); }
    .stApp::before {
        content: '';
        position: fixed; inset: 0;
        background:
            radial-gradient(ellipse 60% 50% at 80% 10%, rgba(255,30,30,0.055) 0%, transparent 60%),
            radial-gradient(ellipse 40% 40% at 10% 90%, rgba(255,30,30,0.035) 0%, transparent 60%);
        pointer-events: none; z-index: 0;
    }

    #MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
    .stDeployButton { display: none; }

    [data-testid="stSidebar"] { background: var(--bg-panel); border-right: 1px solid var(--border); }

    .stTextInput > div > div > input {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        padding: 14px 18px !important;
        transition: border 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-dim) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: var(--text-muted) !important; }

    [data-testid="stChatInput"] > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-lg) !important;
        padding: 4px 8px !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-dim) !important;
    }
    [data-testid="stChatInput"] textarea { color: var(--text-primary) !important; font-family: 'DM Sans', sans-serif !important; }

    .stButton button {
        background: linear-gradient(135deg, #cc0000 0%, #ff2d2d 100%) !important;
        color: #fff !important; border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important; font-size: 13px !important;
        letter-spacing: 0.5px !important; padding: 10px 22px !important;
        transition: all 0.2s !important; text-transform: uppercase !important;
    }
    .stButton button:hover { box-shadow: 0 4px 20px var(--accent-glow) !important; transform: translateY(-1px) !important; }
    .stButton button[kind="secondary"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        color: var(--text-secondary) !important;
    }
    .stButton button[kind="secondary"]:hover {
        border-color: var(--accent) !important; color: var(--text-primary) !important;
        box-shadow: none !important; transform: none !important;
    }

    [data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 6px 0 !important; }
    div[data-testid="stChatMessageContent"] { color: var(--text-primary) !important; font-size: 14px !important; line-height: 1.7 !important; }

    .tt-logo {
        font-family: 'Syne', sans-serif; font-size: 72px; font-weight: 800;
        letter-spacing: -3px;
        background: linear-gradient(135deg, #ffffff 30%, #ff2d2d 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1; margin-bottom: 4px;
    }
    .tt-tagline {
        color: var(--text-muted); font-size: 14px; font-weight: 300;
        letter-spacing: 3px; text-transform: uppercase; margin-bottom: 48px;
    }
    .tt-feature-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 99px; padding: 6px 14px; font-size: 12px;
        color: var(--text-secondary); margin: 4px;
    }
    .tt-pill-dot { width: 6px; height: 6px; background: var(--accent); border-radius: 50%; }

    .timestamp-chip {
        display: inline-flex; align-items: center; gap: 5px;
        background: var(--accent-dim); color: #ff6b6b;
        padding: 4px 12px; border-radius: 99px; font-size: 11px;
        font-weight: 600; margin: 3px 3px 0 0;
        border: 1px solid rgba(255,45,45,0.2); letter-spacing: 0.5px;
        font-family: 'Syne', monospace;
    }
    .sources-header {
        font-size: 10px; color: var(--text-muted);
        text-transform: uppercase; letter-spacing: 2px;
        margin: 16px 0 8px; font-family: 'Syne', sans-serif;
    }
    .ai-badge {
        display: inline-flex; align-items: center; gap: 5px;
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 99px; padding: 3px 10px 3px 6px;
        font-size: 11px; color: var(--text-muted); margin-bottom: 10px;
    }
    .ai-dot {
        width: 8px; height: 8px; background: var(--accent);
        border-radius: 50%; animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } }

    .transcript-segment {
        display: flex; gap: 12px; padding: 10px 12px;
        border-radius: var(--radius-sm); border-left: 2px solid transparent;
        transition: background 0.15s;
    }
    .transcript-segment:hover { background: var(--bg-hover); border-left-color: var(--accent); }
    .transcript-time { font-family: 'Syne', monospace; font-size: 11px; color: var(--accent); min-width: 40px; padding-top: 2px; }
    .transcript-text { font-size: 13px; color: var(--text-secondary); line-height: 1.6; }

    .panel-header {
        font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 700;
        letter-spacing: 2.5px; text-transform: uppercase; color: var(--text-muted);
        margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid var(--border);
    }
    .summary-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 20px; margin-bottom: 16px; }
    .summary-title { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px; }
    .summary-text { font-size: 14px; color: var(--text-secondary); line-height: 1.7; }

    .video-wrapper {
        border-radius: var(--radius-lg); overflow: hidden;
        border: 1px solid var(--border-light);
        box-shadow: 0 0 40px rgba(255,45,45,0.1), 0 20px 60px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 2px; border-bottom: 1px solid var(--border) !important; }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; color: var(--text-muted) !important;
        font-family: 'Syne', sans-serif !important; font-size: 12px !important;
        font-weight: 700 !important; letter-spacing: 1px !important;
        text-transform: uppercase !important; border-bottom: 2px solid transparent !important;
        padding: 10px 16px !important; border-radius: 0 !important;
    }
    .stTabs [aria-selected="true"] { color: var(--text-primary) !important; border-bottom-color: var(--accent) !important; }
    .stTabs [data-baseweb="tab-panel"] { padding: 16px 0 0 0 !important; }

    [data-testid="stMetric"] { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px 20px; }
    [data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 12px !important; }
    [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: 'Syne', sans-serif !important; }

    .main .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
    [data-testid="stChatMessage"] { margin-top: 2px !important; margin-bottom: 2px !important; }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3a3a; }
</style>
""", unsafe_allow_html=True)

# --- CONFIG ---
GROQ_MODEL = "llama3-8b-8192"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- STATE ---
defaults = {
    "vector_db": None,
    "video_url": "",
    "chat_history": [],
    "transcript_segments": [],
    "video_summary": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- HELPERS ---
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return st.session_state.get("groq_api_key", "")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def create_vector_db(segments):
    documents = [Document(page_content=s['text'], metadata={"start": s['start']}) for s in segments]
    return FAISS.from_documents(documents, get_embeddings())

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def generate_summary(segments, api_key):
    full_text = " ".join([s['text'] for s in segments[:80]])
    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key)
    result = llm.invoke(
        f"Summarize this YouTube video transcript in 2-3 concise sentences. No fluff.\n\n{full_text[:3000]}"
    )
    return result.content

def get_quick_prompts():
    return [
        "🧠 What are the key takeaways?",
        "📌 Summarize the main points",
        "❓ What questions does this answer?",
        "🔍 What are the most important moments?",
    ]

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 20px;'>
        <div style='font-family: Syne, sans-serif; font-weight: 800; font-size: 22px; letter-spacing: -1px; color: #f0f0f0;'>▶ TubeTalk</div>
        <div style='font-size: 11px; color: #444; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px;'>AI Video Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # API key input if not in secrets
    if not get_api_key():
        st.markdown("#### 🔑 Groq API Key")
        key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        if key_input:
            st.session_state.groq_api_key = key_input
            st.success("Key saved!")
        st.markdown("[Get free key →](https://console.groq.com)", unsafe_allow_html=True)
        st.divider()

    if st.session_state.vector_db:
        seg_count = len(st.session_state.transcript_segments)
        duration = format_time(st.session_state.transcript_segments[-1]['start']) if seg_count else "—"
        st.metric("Segments", seg_count)
        st.metric("Duration", duration)
        st.metric("Messages", len(st.session_state.chat_history))
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📋 Copy Transcript", type="secondary"):
            full = "\n".join([f"[{format_time(s['start'])}] {s['text']}" for s in st.session_state.transcript_segments])
            st.code(full, language=None)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("＋ New Video", type="primary"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# --- LANDING PAGE ---
if not st.session_state.vector_db:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2.2, 1])
    with col2:
        st.markdown("<div class='tt-logo'>TubeTalk</div>", unsafe_allow_html=True)
        st.markdown("<div class='tt-tagline'>AI Video Intelligence Engine</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='margin-bottom: 32px;'>
            <span class='tt-feature-pill'><span class='tt-pill-dot'></span>Transcript</span>
            <span class='tt-feature-pill'><span class='tt-pill-dot'></span>Semantic Search</span>
            <span class='tt-feature-pill'><span class='tt-pill-dot'></span>AI Chat</span>
            <span class='tt-feature-pill'><span class='tt-pill-dot'></span>Timestamps</span>
            <span class='tt-feature-pill'><span class='tt-pill-dot'></span>Summary</span>
        </div>
        """, unsafe_allow_html=True)

        api_key = get_api_key()
        if not api_key:
            st.warning("⚠️ Add your Groq API key in the sidebar to get started.")
        
        url_input = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            analyze = st.button("Analyze Video →", type="primary", use_container_width=True, disabled=not api_key)

        if analyze:
            if url_input:
                with st.status("Preparing your video...", expanded=True) as status:
                    try:
                        st.write("📋 Fetching transcript from YouTube...")
                        segments = get_transcript(url_input)
                        st.session_state.transcript_segments = segments

                        st.write("🧠 Building semantic index...")
                        st.session_state.vector_db = create_vector_db(segments)
                        st.session_state.video_url = url_input

                        st.write("✨ Generating video summary...")
                        try:
                            st.session_state.video_summary = generate_summary(segments, api_key)
                        except Exception:
                            st.session_state.video_summary = ""

                        status.update(label="Video ready!", state="complete", expanded=False)
                        time.sleep(0.8)
                        st.rerun()
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(f"Error: {e}")
            else:
                st.error("Please enter a YouTube URL first.")

# --- WORKSPACE ---
else:
    left_col, right_col = st.columns([1.4, 2], gap="large")

    with left_col:
        st.markdown("<div class='video-wrapper'>", unsafe_allow_html=True)
        st.video(st.session_state.video_url)
        st.markdown("</div>", unsafe_allow_html=True)

        tab_sum, tab_trans = st.tabs(["Summary", "Transcript"])

        with tab_sum:
            if st.session_state.video_summary:
                st.markdown(f"""
                <div class='summary-card'>
                    <div class='summary-title'>AI Summary</div>
                    <div class='summary-text'>{st.session_state.video_summary}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#444;font-size:13px;padding:12px 0;'>No summary available.</div>", unsafe_allow_html=True)

            segs = st.session_state.transcript_segments
            if segs:
                total_secs = segs[-1]['start']
                word_count = sum(len(s['text'].split()) for s in segs)
                c1, c2 = st.columns(2)
                c1.metric("Duration", format_time(total_secs))
                c2.metric("Word Count", f"{word_count:,}")

        with tab_trans:
            st.markdown("<div class='panel-header'>Full Transcript</div>", unsafe_allow_html=True)
            search_term = st.text_input("Search transcript", placeholder="Filter segments...", label_visibility="collapsed")
            filtered = [s for s in st.session_state.transcript_segments if not search_term or search_term.lower() in s['text'].lower()]
            transcript_html = ""
            for seg in filtered[:120]:
                highlight = "background: var(--accent-dim); border-left-color: var(--accent);" if search_term and search_term.lower() in seg['text'].lower() else ""
                transcript_html += f"<div class='transcript-segment' style='{highlight}'><span class='transcript-time'>{format_time(seg['start'])}</span><span class='transcript-text'>{seg['text']}</span></div>"
            st.markdown(f"<div style='max-height: 340px; overflow-y: auto;'>{transcript_html}</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='panel-header'>Chat</div>", unsafe_allow_html=True)

        if not st.session_state.chat_history:
            st.markdown("<div style='color:#555;font-size:11px;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;'>Try asking</div>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            for idx, qp in enumerate(get_quick_prompts()):
                col = col_a if idx % 2 == 0 else col_b
                with col:
                    if st.button(qp, key=f"qp_{qp}", type="secondary", use_container_width=True):
                        st.session_state._quick_prompt = qp
            st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

        for i, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    st.markdown("<div class='ai-badge'><span class='ai-dot'></span> TubeTalk AI</div>", unsafe_allow_html=True)
                st.markdown(msg["content"], unsafe_allow_html=True)
                if msg["role"] == "assistant" and msg.get("plain_text"):
                    if st.button("Copy", key=f"copy_{i}", type="secondary"):
                        st.code(msg["plain_text"], language=None)

        injected = st.session_state.pop("_quick_prompt", None)
        user_input = st.chat_input("Ask anything about the video...")
        prompt = injected or user_input

        if prompt:
            api_key = get_api_key()
            if not api_key:
                st.error("No API key found. Please add it in the sidebar.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    st.markdown("<div class='ai-badge'><span class='ai-dot'></span> TubeTalk AI</div>", unsafe_allow_html=True)
                    ph = st.empty()

                    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.invoke(prompt)
                    context = "\n".join([f"[{format_time(d.metadata['start'])}] {d.page_content}" for d in docs])

                    system_prompt = f"""You are TubeTalk, a smart AI assistant helping users understand YouTube videos.

Below are the most relevant excerpts from the video transcript (with timestamps):

---
{context}
---

Use this as context to answer accurately. Be concise and helpful. If the question can't be answered from the transcript, say so briefly."""

                    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, streaming=True)
                    full_res = ""

                    try:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                        for chunk in llm.stream(messages):
                            full_res += chunk.content
                            ph.markdown(full_res + " ▌")

                        sources_html = "<div class='sources-header'>Referenced moments</div><div style='margin-bottom: 4px;'>"
                        for d in docs:
                            ts = format_time(d.metadata['start'])
                            snippet = d.page_content[:55] + "..." if len(d.page_content) > 55 else d.page_content
                            sources_html += f"<span class='timestamp-chip' title='{snippet}'>▶ {ts}</span>"
                        sources_html += "</div>"

                        final_html = full_res + "<br>" + sources_html
                        ph.markdown(final_html, unsafe_allow_html=True)
                        st.session_state.chat_history.append({"role": "assistant", "content": final_html, "plain_text": full_res})

                    except Exception as e:
                        error_msg = f"**Groq Error:** `{e}`"
                        ph.markdown(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        if st.session_state.chat_history:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

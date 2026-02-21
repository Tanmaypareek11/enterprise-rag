import streamlit as st
from rag_app import generate_answer_logic
from pypdf import PdfReader

# -------------------------
# Page Config & Setup
# -------------------------
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Dark Mode & Interactive CSS
# -------------------------
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363d;
    }
    
    /* Sidebar Text Fixes */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #E6EDF3 !important;
    }

    /* Custom Hero Card (Dark Gradient) */
    .hero-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 60px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-top: 50px;
        border: 1px solid #30363d;
    }
    .hero-card h1 { font-size: 3rem; margin-bottom: 15px; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    .hero-card p { font-size: 1.3rem; opacity: 0.95; }

    /* Chat Bubbles */
    div[data-testid="stChatMessageContent"] {
        border-radius: 15px;
        padding: 15px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    /* User Message (Blue/Purple) */
    div[data-testid="stChatMessageContainer"] > div:first-child {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        color: white;
        border: none;
    }

    /* Assistant Message (Dark Gray Card) */
    div[data-testid="stChatMessageContainer"] > div:last-child {
        background-color: #21262D;
        color: #E6EDF3;
        border: 1px solid #30363d;
    }

    /* Input Area */
    div.stChatInputContainer {
        background-color: #21262D;
        border: 1px solid #30363d;
        border-radius: 15px;
    }
    
    /* Input Text Color */
    .stChatInput textarea {
        color: white !important;
    }

    /* Metrics & Cards */
    .metric-card {
        background: #21262D;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        text-align: center;
        border: 1px solid #30363d;
    }
    .metric-value { font-size: 1.4rem; font-weight: bold; color: #58A6FF; }
    .metric-label { font-size: 0.9rem; color: #8B949E; margin-top: 5px; }

    /* Buttons */
    .stButton > button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #2EA043;
    }

    /* Info Boxes */
    .stAlert {
        background-color: #21262D;
        color: #E6EDF3;
        border: 1px solid #30363d;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #F0F6FC !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State Initialization
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_processed" not in st.session_state:
    st.session_state.file_processed = None

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("### 📂 Document Hub")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"], label_visibility="collapsed")
    
    st.markdown("---")
    
    # File Info Area in Sidebar
    if uploaded_file:
        try:
            reader = PdfReader(uploaded_file)
            num_pages = len(reader.pages)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.1rem;">📄 {uploaded_file.name[:25]}</div>
                <div class="metric-label">{num_pages} Pages • {round(uploaded_file.size / 1024, 1)} KB</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.markdown("""
        <div style="text-align: center; color: #8B949E; padding: 20px;">
            Waiting for file...
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# Main Logic
# -------------------------
st.title("Enterprise RAG Assistant")

# --- STATE 1: NO FILE UPLOADED (HERO SCREEN) ---
if uploaded_file is None:
    st.markdown("""
    <div class="hero-card">
        <h1>👋 Welcome</h1>
        <p>Upload a PDF in the sidebar to activate the AI.</p>
        <p style="margin-top: 20px; font-size: 1rem; opacity: 0.8;">Powered by FLAN-T5 & FAISS</p>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion Chips
    st.markdown("### 💡 Try asking")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("What is the summary?")
    with col2:
        st.info("Who are the authors?")
    with col3:
        st.info("Key takeaways?")

# --- STATE 2: FILE UPLOADED (CHAT INTERFACE) ---
else:
    # Check for new file
    if st.session_state.file_processed != uploaded_file.name:
        st.session_state.file_processed = uploaded_file.name

    # Document Details Expander
    with st.expander("📜 Document Details", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Filename:** {uploaded_file.name}")
        with c2:
            st.write(f"**Size:** {round(uploaded_file.size / 1024, 2)} KB")

    # --- Chat History Display ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input Area ---
    if prompt := st.chat_input(f"Ask about {uploaded_file.name}..."):
        
        # 1. Add User Message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Generate Response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing document..."):
                try:
                    response = generate_answer_logic(prompt, uploaded_file)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
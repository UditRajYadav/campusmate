import streamlit as st
from load_docs import load_all_pdfs, chunk_text
from load_web import load_web_pages
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import re
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(page_title="CampusMate | BBDU", page_icon="🎓", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

* { font-family: 'Poppins', sans-serif !important; }

#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 900px; }

body {
    background: radial-gradient(circle at 20% 20%, #1e293b 0%, #0f172a 60%);
}

.hero {
    text-align: center;
    padding: 1.5rem 0 2rem 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    color: #38bdf8;
    padding: 5px 16px;
    border-radius: 30px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
    border: 1px solid rgba(56,189,248,0.3);
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 8px;
}

.chat-card {
    border-radius: 18px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    backdrop-filter: blur(10px);
    line-height: 1.55;
}
.user-card {
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: white;
    margin-left: 15%;
    border-bottom-right-radius: 4px;
}
.bot-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e2e8f0;
    margin-right: 15%;
    border-bottom-left-radius: 4px;
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    opacity: 0.7;
    margin-bottom: 6px;
}

div[data-testid="stChatInput"] textarea {
    border-radius: 16px !important;
    border: 1.5px solid rgba(56,189,248,0.4) !important;
    background: rgba(255,255,255,0.04) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">🎓 AI-POWERED &nbsp;·&nbsp; RAG CHATBOT</div>
    <div class="hero-title">CampusMate</div>
    <div class="hero-subtitle">Your instant answer engine for everything BBDU — fees, calendar, notices & more</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🎓 CampusMate")
    st.markdown("An AI RAG chatbot built for **Babu Banarasi Das University**, combining live web data and official PDFs.")
    st.markdown("---")
    st.markdown("**📚 Data Sources**")
    st.markdown("- Fee Structure PDF\n- Academic Handbook\n- Holiday Calendar\n- BBDU Website (live)")
    st.markdown("---")
    st.markdown("**⚙️ Tech Stack**")
    st.markdown("RAG · ChromaDB · Sentence-Transformers · Groq LLaMA 3.1 · Streamlit")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

WEB_URLS = [
    "https://www.bbdu.ac.in/",
    "https://www.bbdu.ac.in/category/notices",
    "https://www.bbdu.ac.in/category/academic-calendar",
    "https://www.bbdu.ac.in/category/examination-notices",
    "https://www.bbdu.ac.in/anti-ragging",
    "https://www.bbdu.ac.in/iqac/policy-document",
]

@st.cache_resource
def build_index():
    pdf_docs = load_all_pdfs()
    web_docs = load_web_pages(WEB_URLS)
    all_docs = pdf_docs + web_docs

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="campusmate_docs"
)

    chunk_id = 0
    for source_name, text in all_docs:
        if source_name in ("fee_summary.txt", "general_info.txt"):
            chunks = [line.strip() for line in text.split("\n") if line.strip()]
        else:
            chunks = chunk_text(text)

        embeddings = model.encode(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            collection.add(
                ids=[f"chunk_{chunk_id}"],
                embeddings=[embedding.tolist()],
                documents=[chunk],
                metadatas=[{"source": source_name}]
            )
            chunk_id += 1
    return model, collection

with st.spinner("🔧 Loading knowledge base..."):
    model, collection = build_index()

def load_lines(filename):
    path = os.path.join("documents", filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

FEE_LINES = load_lines("fee_summary.txt")
GENERAL_LINES = load_lines("general_info.txt")

def normalize(text):
    return re.sub(r'[^a-z0-9\s]', '', text.lower())

def search_lines(question, lines):
    """Find lines that contain keywords from the question (punctuation-insensitive)."""
    question_normalized = normalize(question)
    question_words = [w for w in question_normalized.split() if len(w) > 2]
    matches = []
    for line in lines:
        line_normalized = normalize(line)
        if any(word in line_normalized for word in question_words):
            matches.append(line)
    return matches

def get_answer(question):
    general_keywords = ["pincode", "pin code", "address", "location", "contact",
                         "phone", "email", "helpline", "situated", "where"]
    fee_keywords = ["fee", "fees", "tuition", "cost", "charges", "payment"]

    q_lower = question.lower()
    is_general_question = any(word in q_lower for word in general_keywords)
    is_fee_question = any(word in q_lower for word in fee_keywords)

    if is_general_question and GENERAL_LINES:
        matched_lines = search_lines(question, GENERAL_LINES)
        if matched_lines:
            combined_context = "\n".join(matched_lines)
            source_file = "general_info.txt"
        else:
            query_embedding = model.encode([question])
            results = collection.query(query_embeddings=query_embedding.tolist(), n_results=8)
            combined_context = "\n\n".join(results["documents"][0])
            source_file = results["metadatas"][0][0]["source"]

    elif is_fee_question:
        matched_lines = search_lines(question, FEE_LINES)
        if matched_lines:
            combined_context = "\n".join(matched_lines)
            source_file = "fee_summary.txt"
        else:
            query_embedding = model.encode([question])
            results = collection.query(query_embeddings=query_embedding.tolist(), n_results=5,
                                        where={"source": "fee_summary.txt"})
            combined_context = "\n\n".join(results["documents"][0])
            source_file = "fee_summary.txt"

    else:
        query_embedding = model.encode([question])
        results = collection.query(query_embeddings=query_embedding.tolist(), n_results=8)
        combined_context = "\n\n".join(results["documents"][0])
        source_file = results["metadatas"][0][0]["source"]

    prompt = f"""You are CampusMate, a helpful AI assistant for BBDU (Babu Banarasi Das University).

Answer the student's question using ONLY the information given below.

Reasoning rules:
- If a fee is stated as "per semester" without naming a specific semester number, it applies equally to EVERY semester of that course's duration.
- Course name abbreviations should be matched flexibly (e.g., "Mtech", "M.Tech", "M. Tech" all refer to the same course).
- If the answer can be reasonably inferred from the context, infer it (e.g., if the context contains "Lucknow 226028", the pincode is 226028; if it contains an address, use that for location questions).

Only say "I don't know" if the answer is genuinely absent from the context even after this reasoning.

Context:
{combined_context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content, source_file

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("##### 💡 Try asking:")
    cols = st.columns(4)
    quick_qs = ["What is the fee for B.Tech CSE?", "When do holidays start?", "What is the pincode of BBDU?", "Anti-ragging policy?"]
    for col, q in zip(cols, quick_qs):
        if col.button(q, use_container_width=True):
            st.session_state.pending_question = q

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-card user-card">
            <div class="card-label">You</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-card bot-card">
            <div class="card-label">🎓 CampusMate</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

question = st.chat_input("Ask CampusMate anything about BBDU...")
pending = st.session_state.pop("pending_question", None)
final_question = pending or question

if final_question:
    st.session_state.messages.append({"role": "user", "content": final_question})
    with st.spinner("Thinking..."):
        answer, source_file = get_answer(final_question)
    st.session_state.messages.append({"role": "assistant", "content": answer, "source": source_file})
    st.rerun()
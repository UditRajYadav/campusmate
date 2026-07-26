import streamlit as st
from load_docs import load_all_pdfs, chunk_text
from load_web import load_web_pages
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(page_title="CampusMate", page_icon="🎓")
st.title("🎓 CampusMate")
st.caption("Ask me anything about your college documents and website")

WEB_URLS = [
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
    collection = chroma_client.create_collection(name="campusmate_docs")

    chunk_id = 0
    for source_name, text in all_docs:
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

with st.spinner("Loading documents and website content..."):
    model, collection = build_index()

def get_answer(question):
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=5)

    context_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    combined_context = "\n\n".join(context_chunks)
    source_file = sources[0]["source"]

    prompt = f"""You are a helpful college assistant. Answer the student's question 
using ONLY the information given below.

Important reasoning rule: If a fee is stated as "per semester" or "per Sem" without 
naming a specific semester number, that fee applies equally to EVERY semester of that 
course's duration (e.g., "Rs. 70,250/- per Sem" for a 4-year B.Tech course means 
Rs. 70,250/- for semester 1, 2, 3, 4, 5, 6, 7, and 8 alike) — unless the context 
explicitly states a different amount for a specific semester.

If multiple sources give related information, prioritize a clean summary source 
(like fee_summary.txt) over a raw table extraction if both are present.

If the answer genuinely isn't in the context even after this reasoning, say you don't know.

Context:
{combined_context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    return answer, source_file

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Type your question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, source_file = get_answer(question)
            st.write(answer)
            st.caption(f"📄 Source: {source_file}")
    st.session_state.messages.append({"role": "assistant", "content": answer})
import streamlit as st
from load_docs import load_all_pdfs, chunk_text
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(page_title="CampusMate", page_icon="🎓")
st.title("🎓 CampusMate")
st.caption("Ask me anything about your college documents")

@st.cache_resource
def build_index():
    docs = load_all_pdfs()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="campusmate_docs")

    chunk_id = 0
    for filename, text in docs:
        chunks = chunk_text(text)
        embeddings = model.encode(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            collection.add(
                ids=[f"chunk_{chunk_id}"],
                embeddings=[embedding.tolist()],
                documents=[chunk],
                metadatas=[{"source": filename}]
            )
            chunk_id += 1
    return model, collection

with st.spinner("Loading documents..."):
    model, collection = build_index()

def get_answer(question):
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=1)
    best_chunk = results["documents"][0][0]
    source_file = results["metadatas"][0][0]["source"]

    prompt = f"""You are a helpful college assistant. Answer the student's question 
using ONLY the information given below. If the answer isn't in the context, say you don't know.

Context:
{best_chunk}

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
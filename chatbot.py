from load_docs import load_all_pdfs, chunk_text
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
from groq import Groq

# Setup
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Loading document and building search index...")

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
            documents=[chunk]
        )
        chunk_id += 1
print("Ready! Ask me anything about the document. Type 'quit' to exit.\n")

# Chat loop
def get_answer(question):
    query_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=1
    )
    best_chunk = results["documents"][0][0]

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
    return response.choices[0].message.content

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = get_answer(question)
    print("CampusMate:", answer, "\n")
from load_docs import load_all_pdfs, chunk_text
from load_web import load_web_pages
from sentence_transformers import SentenceTransformer
import chromadb

WEB_URLS = [
    "https://www.bbdu.ac.in/",
    "https://www.bbdu.ac.in/category/notices",
    "https://www.bbdu.ac.in/category/academic-calendar",
    "https://www.bbdu.ac.in/category/examination-notices",
    "https://www.bbdu.ac.in/anti-ragging",
    "https://www.bbdu.ac.in/iqac/policy-document",
]

pdf_docs = load_all_pdfs()
web_docs = load_web_pages(WEB_URLS)
all_docs = pdf_docs + web_docs

print("\n=== All loaded files ===")
for name, text in all_docs:
    print(f"{name}: {len(text)} characters")

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="debug_docs")

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

question = "Mtech fee"
query_embedding = model.encode([question])
results = collection.query(query_embeddings=query_embedding.tolist(), n_results=5)

print(f"\n=== Top 5 chunks retrieved for: '{question}' ===\n")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"--- Result {i+1} | Source: {meta['source']} ---")
    print(doc[:300])
    print()
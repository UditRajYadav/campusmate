from load_docs import load_pdf_text, chunk_text
from sentence_transformers import SentenceTransformer
import chromadb

# Load and chunk the document
text = load_pdf_text("test_doc.pdf")
chunks = chunk_text(text)

# Load the embedding model (downloads once, then cached)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert chunks into embeddings
print("Creating embeddings...")
embeddings = model.encode(chunks)

# Set up ChromaDB (a local database for embeddings)
client = chromadb.Client()
collection = client.create_collection(name="campusmate_docs")

# Store each chunk with its embedding
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding.tolist()],
        documents=[chunk]
    )

print(f"Stored {len(chunks)} chunks in the database.")

# Test: search for something
query = "What are the library hours?"
query_embedding = model.encode([query])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=1
)

print("\n--- Question ---")
print(query)
print("\n--- Best matching chunk ---")
print(results["documents"][0][0])
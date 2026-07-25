from pypdf import PdfReader
import os

def load_all_pdfs(folder_path="documents"):
    """Reads every PDF in the folder and returns a list of (filename, text) pairs."""
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            all_docs.append((filename, full_text))
    return all_docs

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    docs = load_all_pdfs()
    print(f"Found {len(docs)} PDF(s): {[name for name, _ in docs]}")
    for name, text in docs:
        chunks = chunk_text(text)
        print(f"{name}: {len(chunks)} chunks")
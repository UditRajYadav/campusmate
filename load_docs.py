from pypdf import PdfReader
import os
import time
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

CACHE_FILE = "cleaned_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)

def clean_with_llm(raw_text, cache_key, cache):
    if cache_key in cache:
        return cache[cache_key]  # already cleaned before, skip API call

    if len(raw_text.strip()) < 50:
        return raw_text

    prompt = f"""The following text was extracted from a PDF and may contain jumbled 
table data (numbers separated from their labels). Rewrite it as clear, complete sentences.

STRICT RULES:
- Do NOT invent, guess, or add any number or fact not present in the original text.
- Do NOT skip or omit any number or fact present in the original text.
- Keep every number exactly as written.
- If something is genuinely unclear or ambiguous, keep it as-is rather than guessing.

Text to rewrite:
{raw_text}

Rewritten version:"""

    time.sleep(2)  # small delay to avoid hitting rate limits
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    cleaned = response.choices[0].message.content
    cache[cache_key] = cleaned
    return cleaned

def load_all_pdfs(folder_path="documents"):
    cache = load_cache()
    all_docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):
            reader = PdfReader(path)
            full_text = ""
            for i, page in enumerate(reader.pages):
                raw_page_text = page.extract_text() or ""
                cache_key = f"{filename}_page{i}"
                cleaned = clean_with_llm(raw_page_text, cache_key, cache)
                full_text += cleaned + "\n"
            all_docs.append((filename, full_text))

        elif filename.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                full_text = f.read()
            all_docs.append((filename, full_text))

    save_cache(cache)
    return all_docs

def chunk_text(text, chunk_size=200, overlap=80):
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
    print(f"Found {len(docs)} file(s): {[name for name, _ in docs]}")
    for name, text in docs:
        chunks = chunk_text(text)
        print(f"{name}: {len(chunks)} chunks")
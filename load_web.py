import requests
from bs4 import BeautifulSoup

def load_web_pages(urls):
    """Fetches each URL, strips HTML, and returns a list of (url, text) pairs."""
    all_docs = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts/styles/nav clutter
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            all_docs.append((url, text))
            print(f"Loaded: {url} ({len(text)} characters)")
        except Exception as e:
            print(f"Failed to load {url}: {e}")
    return all_docs

if __name__ == "__main__":
    test_urls = [
        "https://www.bbdu.ac.in/category/notices",
        "https://www.bbdu.ac.in/category/academic-calendar",
        "https://www.bbdu.ac.in/category/examination-notices",
        "https://www.bbdu.ac.in/anti-ragging",
        "https://www.bbdu.ac.in/iqac/policy-document",
        "https://collegedunia.com/lucknow-colleges",
    ]
    docs = load_web_pages(test_urls)
    for url, text in docs:
        print(f"\n--- {url} preview ---\n{text[:300]}")
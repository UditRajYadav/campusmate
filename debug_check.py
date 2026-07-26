from load_docs import load_all_pdfs

docs = load_all_pdfs()
for filename, text in docs:
    if "fee_structure" in filename:
        # Find the part mentioning MBA
        idx = text.find("MBA")
        print(text[idx:idx+1000])
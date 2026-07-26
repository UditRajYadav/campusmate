def get_answer(question):
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)
    
    # Combine multiple chunks instead of just one
    context_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    combined_context = "\n\n".join(context_chunks)
    source_file = sources[0]["source"]  # just show the top source

    prompt = f"""You are a helpful college assistant. Answer the student's question 
using ONLY the information given below. If the answer isn't in the context, say you don't know.

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
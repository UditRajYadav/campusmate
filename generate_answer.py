from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_answer(question, context_chunk):
    prompt = f"""You are a helpful college assistant. Answer the student's question 
using ONLY the information given below. If the answer isn't in the context, say you don't know.

Context:
{context_chunk}
Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    test_context = "Library Hours: Monday-Friday 8:00 AM to 10:00 PM, Saturday 9:00 AM to 6:00 PM, Sunday 10:00 AM to 4:00 PM."
    test_question = "What time does the library close on Saturday?"
    answer = generate_answer(test_question, test_context)
    print("Answer:", answer)
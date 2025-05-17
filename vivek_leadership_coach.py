
import streamlit as st
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import os

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

book_chunks = [
    {"text": "Level 5 Leadership is about humility + fierce resolve. This concept is central to Good to Great by Jim Collins.", "source": "Good to Great"},
    {"text": "The Golden Circle starts with WHY, then HOW, then WHAT. Simon Sinek explains this in Start With Why.", "source": "Start With Why"},
    {"text": "Trust is built through credibility and behavior. The Speed of Trust by Stephen M.R. Covey defines 13 behaviors to build trust.", "source": "The Speed of Trust"},
    {"text": "Atomic Habits teaches that identity-based habits are stronger than outcome-based ones. Small habits compound.", "source": "Atomic Habits"},
    {"text": "Leaders Eat Last emphasizes the importance of the Circle of Safety in building team trust.", "source": "Leaders Eat Last"},
]

texts = [chunk["text"] for chunk in book_chunks]
sources = [chunk["source"] for chunk in book_chunks]

@st.cache_resource
def embed_texts(texts: List[str]):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings, model

embeddings, model = embed_texts(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

st.title("Vivek Leadership Coach 🧠")
st.markdown("Ask a leadership question based on insights from 15 top books.")

query = st.text_input("Enter your leadership question:")

if query:
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k=3)

    retrieved_chunks = [texts[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)

    prompt = f"You are Vivek's Leadership Coach. Use the following excerpts from leadership books to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    with st.spinner("Thinking like a Level 5 Leader..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a leadership coach trained on 15 classic books."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5
        )

        answer = response.choices[0].message.content
        st.markdown("### Answer")
        st.write(answer)

        st.markdown("---")
        st.markdown("**Sources used:**")
        for i in I[0]:
            st.write(f"- {sources[i]}")

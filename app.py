import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
# ============================
# 0️⃣ LOAD PARQUET
# ============================
df = pd.read_parquet("embeddings.parquet")
df.to_csv("embeddings.csv", index=False)


# ============================
# 1️⃣ EMBEDDING FUNCTION
# ============================

def create_embedding(text_list):
    r = requests.post(
        "http://127.0.0.1:11434/api/embed",
        json={"model": "bge-m3", "input": text_list}
    )
    r.raise_for_status()
    return r.json()["embeddings"]


# ============================
# 2️⃣ LLM INFERENCE
# ============================



genai.configure(api_key="SECRET API KEY")
model = genai.GenerativeModel("models/gemini-2.0-flash-001")

def inference(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"


# ============================
# 3️⃣ TIME FORMATTER
# ============================
def format_timestamp(seconds: float) -> str:
    sec = int(seconds)
    return f"{sec//60:02d}:{sec%60:02d}"

# ============================
# 4️⃣ CONTEXT BUILDER
# ============================
def build_context(top_df: pd.DataFrame) -> str:
    blocks = []
    for _, r in top_df.iterrows():
        block = f"""
Video Title: {r['title']}
Video Number: {r['number']}
Timestamp: {format_timestamp(r['start'])} → {format_timestamp(r['end'])}
Subtitle: {r['text']}
""".strip()
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)

# ============================
# 5️⃣ RAG PIPELINE
# ============================
def answer_query(query: str):
    # Embed question
    q_emb = create_embedding([query])[0]

    # Similarity search
    sim = cosine_similarity(np.vstack(df["embedding"]), [q_emb]).flatten()
    top_idx = sim.argsort()[::-1][:5]
    top_df = df.loc[top_idx]

    # Build human-readable context
    context = build_context(top_df)

    # LLM Prompt
    prompt = f"""
You are a tutor for the "Web Development" course.
ONLY use the chunks below to answer.

Relevant video chunks:
{context}

------------------------------------
User question: "{query}"

Your task:
- Tell EXACT video number
- Mention video title
- Mention timestamp shown above
- Explain briefly what is taught there
- If question is unrelated to the course, say you cannot answer.
- DO NOT make up any new video numbers or timestamps.
- Only use information from the given chunks.
"""

    return inference(prompt)

# ============================
# 6️⃣ STREAMLIT UI
# ============================

st.set_page_config(page_title="Sigma WebDev Tutor", page_icon="💬")

# Custom CSS
st.markdown("""
<style>
.chat-bubble-user {
    background-color: #1a1a1a;
    padding: 12px 18px;
    border-radius: 12px;
    margin-bottom: 12px;
    text-align: right;
    color: white;
}
.chat-bubble-bot {
    background-color: #262626;
    padding: 12px 18px;
    border-radius: 12px;
    margin-bottom: 12px;
    color: #e5e5e5;
}
.title {
    font-size: 30px;
    font-weight: 700;
    text-align: center;
    padding-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "bot",
        "content": "👋 Hello! Ask any question about the Web Development course. Feel free to ask me about any topic from "
        "the course and I will try to give you the exact video numbers and time stamps where you can find it.Happy Learning"
    })

st.markdown("<div class='title'>💬 Web Development Course Tutor Assistant</div>", unsafe_allow_html=True)

# Display chat messages
for msg in st.session_state.messages:
    bubble_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
    st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# Input
query = st.text_input("Ask your question about the course:")

# Button
if st.button("Ask"):
    if query.strip() != "":
        st.session_state.messages.append({"role": "user", "content": query})
        answer = answer_query(query)
        st.session_state.messages.append({"role": "bot", "content": answer})
        st.rerun()

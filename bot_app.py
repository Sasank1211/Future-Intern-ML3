import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------
# Load Dataset
# -------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\LENOVO\Downloads\tech_support_dataset.csv")

    # Lowercase all column names
    df.columns = df.columns.str.lower()

    # Auto-map typical column names
    if "customer_issue" in df.columns and "tech_response" in df.columns:
        df = df.rename(columns={"customer_issue": "question", "tech_response": "answer"})
    elif "query" in df.columns and "response" in df.columns:
        df = df.rename(columns={"query": "question", "response": "answer"})

    # Check again
    if "question" not in df.columns or "answer" not in df.columns:
        st.error("Dataset must have 'question' and 'answer' columns (or similar like Customer_Issue & Tech_Response).")
        st.stop()

    return df[["question", "answer"]]

# -------------------
# Create Embeddings & Index
# -------------------
@st.cache_resource
def build_index(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embeddings = model.encode(df["question"].tolist(), show_progress_bar=True)

    d = question_embeddings.shape[1]  # embedding dimensions
    index = faiss.IndexFlatL2(d)
    index.add(np.array(question_embeddings).astype("float32"))
    return model, index

# -------------------
# Search Function
# -------------------
def get_answer(user_query, model, index, df, top_k=1):
    query_embedding = model.encode([user_query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), top_k)
    best_answer = df.iloc[indices[0][0]]["answer"]
    return best_answer

# -------------------
# Streamlit UI
# -------------------
def main():
    st.set_page_config(page_title="Customer Support Chatbot", page_icon="🤖")
    st.title("🤖 Tech Support Chatbot")

    df = load_data()
    model, index = build_index(df)

    if "history" not in st.session_state:
        st.session_state.history = []

    # Chat input
    user_input = st.chat_input("Ask me your tech support question...")

    if user_input:
        answer = get_answer(user_input, model, index, df)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", answer))

    # Display chat history
    for role, text in st.session_state.history:
        with st.chat_message("user" if role == "You" else "assistant"):
            st.write(text)

if __name__ == "__main__":
    main()

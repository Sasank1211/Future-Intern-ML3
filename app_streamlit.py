import streamlit as st
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------
# Load Dataset
# -------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\LENOVO\Downloads\Future_IN_ML_1\tech_support_dataset.csv")
    df = df.rename(columns={
        "Customer_Issue": "question",
        "Tech_Response": "answer"
    })
    return df[["question", "answer"]]

# -------------------
# Build Backend (TF-IDF index)
# -------------------
@st.cache_resource
def build_backend(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["question"])
    return vectorizer, X

# -------------------
# Find Best Answer
# -------------------
def get_answer(user_query, vectorizer, X, df):
    query_vec = vectorizer.transform([user_query])
    sims = cosine_similarity(query_vec, X).flatten()
    idx = sims.argmax()
    return df.iloc[idx]["answer"]

# -------------------
# Streamlit App
# -------------------
def main():
    st.set_page_config(page_title="Tech Support Chatbot", page_icon="🤖")
    st.title("🤖 Tech Support Chatbot")

    df = load_data()
    vectorizer, X = build_backend(df)

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.chat_input("Ask me your tech support question...")

    if user_input:
        answer = get_answer(user_input, vectorizer, X, df)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", answer))

    for role, msg in st.session_state.history:
        with st.chat_message("user" if role == "You" else "assistant"):
            st.write(msg)

if __name__ == "__main__":
    main()




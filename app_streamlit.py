import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import math
# -------------------
def tokenize(text):
    # simple tokenizer: keep word characters
    return re.findall(r"\w+", str(text).lower())

def build_backend_manual(docs):
    """
    docs: list of strings
    returns: vocab (dict term->idx), tfidf_norm (N x V numpy array), idf (numpy array), docs list
    """
    tokenized = [tokenize(d) for d in docs]
    # build vocab
    vocab = {}
    for tokens in tokenized:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    V = len(vocab)
    N = len(docs)
    # term frequency matrix
    tf = np.zeros((N, V), dtype=np.float32)
    df_counts = np.zeros(V, dtype=np.float32)
    for i, tokens in enumerate(tokenized):
        c = Counter(tokens)
        for term, cnt in c.items():
            j = vocab[term]
            tf[i, j] = cnt
        for term in set(tokens):
            df_counts[vocab[term]] += 1.0
    # idf (smooth)
    idf = np.log((1.0 + N) / (1.0 + df_counts)) + 1.0
    # tf-idf
    tfidf = tf * idf[np.newaxis, :]
    # normalize rows
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf_norm = tfidf / norms
    return vocab, tfidf_norm, idf, docs

def query_to_vec(query, vocab, idf):
    tokens = tokenize(query)
    vec = np.zeros(len(vocab), dtype=np.float32)
    c = Counter(tokens)
    for term, cnt in c.items():
        if term in vocab:
            vec[vocab[term]] = cnt * idf[vocab[term]]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def get_answer_manual(user_query, vocab, tfidf_norm, idf, docs, df):
    qvec = query_to_vec(user_query, vocab, idf)
    if qvec.sum() == 0:
        return "Sorry, I don't have a matching answer for that. Please rephrase or provide more details."
    sims = tfidf_norm.dot(qvec)
    idx = int(np.argmax(sims))
    # optional: if top similarity is tiny, say unclear
    top_sim = float(sims[idx])
    if top_sim < 0.1:
        return "I found a weak match; could you give more details or try rephrasing?"
    return df.iloc[idx]["answer"]

# -------------------
# Load Dataset (ensure the CSV is in the same directory as the app)
# -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("tech_support_dataset.csv")
    # rename if columns differ
    rename_map = {}
    if "Customer_Issue" in df.columns and "Tech_Response" in df.columns:
        rename_map = {"Customer_Issue": "question", "Tech_Response": "answer"}
    elif "question" in df.columns and "answer" in df.columns:
        rename_map = {}
    else:
        # attempt best-guess rename: first two text columns
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if len(text_cols) >= 2:
            rename_map = {text_cols[0]: "question", text_cols[1]: "answer"}
    if rename_map:
        df = df.rename(columns=rename_map)
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    return df[["question", "answer"]]

@st.cache_resource
def build_backend(df):
    docs = df["question"].tolist()
    vocab, tfidf_norm, idf, docs_out = build_backend_manual(docs)
    return vocab, tfidf_norm, idf, docs_out

# -------------------
# Streamlit UI
# -------------------
def main():
    st.set_page_config(page_title="Tech Support Chatbot", page_icon="🤖")
    st.title("🤖 Tech Support Chatbot (no sklearn)")
    st.write("This app uses a small built-in TF-IDF implementation so we don't depend on scikit-learn.")

    df = load_data()
    vocab, tfidf_norm, idf, docs = build_backend(df)

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.chat_input("Ask a tech support question...")

    if user_input:
        answer = get_answer_manual(user_input, vocab, tfidf_norm, idf, docs, df)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", answer))

    for role, msg in st.session_state.history:
        with st.chat_message("user" if role == "You" else "assistant"):
            st.write(msg)

    st.markdown("---")
    st.subheader("Dataset stats")
    st.write(f"Number of examples: **{len(df)}**")
    # show top 10 most common question words
    all_tokens = []
    for q in df["question"].sample(min(2000, len(df))).tolist():
        all_tokens.extend(tokenize(q))
    top = Counter(all_tokens).most_common(15)
    st.write("Top tokens in sample questions:")
    st.dataframe(pd.DataFrame(top, columns=["token", "count"]).set_index("token"))

if __name__ == "__main__":
    main()







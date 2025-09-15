# support_backend.py
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import uuid
import datetime

DB_PATH = "support_tickets.db"
DATA_PATH = r"C:\Users\LENOVO\Downloads\tech_support_dataset.csv"

# ---------------------------
# Dataset loader / normalizer
# ---------------------------
def load_dataset(csv_path=DATA_PATH):
    df = pd.read_csv(csv_path)
    # Normalize column names (case-insensitive)
    cols = {c: c.strip() for c in df.columns}
    df.columns = list(cols.keys())
    # Map expected columns
    rename_map = {}
    if "Customer_Issue" in df.columns:
        rename_map["Customer_Issue"] = "question"
    if "Tech_Response" in df.columns:
        rename_map["Tech_Response"] = "answer"
    # fallback guesses
    for c in df.columns:
        low = c.lower()
        if low in ["question", "query", "issue", "customer_issue"] and "question" not in rename_map.values():
            rename_map[c] = "question"
        if low in ["answer", "response", "tech_response"] and "answer" not in rename_map.values():
            rename_map[c] = "answer"

    df = df.rename(columns=rename_map)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain question and answer columns (mapped from Customer_Issue/Tech_Response).")
    df = df[["question", "answer"]].dropna().reset_index(drop=True)
    return df

# ---------------------------
# Retriever: TF-IDF
# ---------------------------
class Retriever:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
        self.X = self.vectorizer.fit_transform(self.df["question"].values.astype("U"))

    def get_most_similar(self, user_query, top_k=1):
        qv = self.vectorizer.transform([user_query])
        sims = cosine_similarity(qv, self.X).flatten()
        if sims.max() == 0:
            return []  # no match
        idxs = sims.argsort()[-top_k:][::-1]
        results = []
        for idx in idxs:
            results.append({
                "question": self.df.iloc[idx]["question"],
                "answer": self.df.iloc[idx]["answer"],
                "score": float(sims[idx])
            })
        return results

# ---------------------------
# Simple ticket DB
# ---------------------------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            customer TEXT,
            issue_text TEXT,
            category TEXT,
            status TEXT,
            assigned_to TEXT,
            note TEXT
        )
    """)
    conn.commit()
    return conn

def create_ticket(conn, customer, issue_text, category="General", note=""):
    ticket_id = str(uuid.uuid4())[:8]
    created_at = datetime.datetime.utcnow().isoformat()
    c = conn.cursor()
    c.execute("""
        INSERT INTO tickets (id, created_at, customer, issue_text, category, status, assigned_to, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticket_id, created_at, customer, issue_text, category, "open", None, note))
    conn.commit()
    return ticket_id

def get_ticket(conn, ticket_id):
    c = conn.cursor()
    c.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    row = c.fetchone()
    if not row:
        return None
    keys = ["id","created_at","customer","issue_text","category","status","assigned_to","note"]
    return dict(zip(keys, row))

def list_open_tickets(conn, limit=50):
    c = conn.cursor()
    c.execute("SELECT * FROM tickets WHERE status != 'closed' ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    keys = ["id","created_at","customer","issue_text","category","status","assigned_to","note"]
    return [dict(zip(keys, r)) for r in rows]

# ---------------------------
# Initialization helper
# ---------------------------
def build_backend():
    df = load_dataset()
    retriever = Retriever(df)
    conn = init_db()
    return df, retriever, conn

# If executed directly, quick check
if __name__ == "__main__":
    df, r, conn = build_backend()
    print("Loaded dataset rows:", len(df))

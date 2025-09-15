import streamlit as st
from support_backend import build_backend, create_ticket, list_open_tickets
import os
import openai  # optional; only if you want fallback generation

# Optional: set OPENAI_API_KEY environment variable for LLM fallback:
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# Build backend
df, retriever, conn = build_backend()

st.set_page_config(page_title="Tech Support Chatbot", layout="wide")
st.title("🤖 Tech Support Chatbot")

# Sidebar: ticket list for support agents
with st.sidebar:
    st.header("Support Agent Panel")
    if st.button("Refresh tickets"):
        pass
    tickets = list_open_tickets(conn, limit=100)
    st.write(f"Open tickets: {len(tickets)}")
    for t in tickets[:10]:
        st.markdown(f"**{t['id']}** — {t['customer']} — {t['status']}")
        st.caption(f"{t['issue_text'][:80]}")

# Main chat area
if "history" not in st.session_state:
    st.session_state.history = []

# Kick-off quick greeting + category buttons on first load
if not st.session_state.history:
    st.session_state.history.append(("bot", "Hello! I'm your Tech Support assistant. How can I help today?"))
    st.session_state.history.append(("bot", "Choose an option or type your question:"))
    st.session_state.quick_buttons = ["Can't connect to Wi-Fi", "Software/install issue", "Forgot password", "Other / Human"]

# Display history
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

# Quick option buttons
cols = st.columns(4)
options = st.session_state.get("quick_buttons", [])
for i, opt in enumerate(options):
    if cols[i].button(opt):
        # treat as a user input
        st.session_state.history.append(("user", opt))
        user_query = opt
        # auto-run retrieval below by adding to session_state variable
        st.session_state.last_input = user_query

# Chat input
user_input = st.chat_input("Type your problem (or pick a button)...")
if user_input:
    st.session_state.history.append(("user", user_input))
    st.session_state.last_input = user_input

# If there is an input to handle:
if "last_input" in st.session_state:
    user_query = st.session_state.pop("last_input")
    # 1) Try retrieval
    results = retriever.get_most_similar(user_query, top_k=1)
    if results:
        best = results[0]
        # threshold: only use if score reasonably high (tune as needed)
        if best["score"] > 0.2:
            reply = f"I found this in our knowledge base:\n\n**Q:** {best['question']}\n\n**A:** {best['answer']}\n\nIf this doesn't solve it, reply 'human' to create a support ticket."
            st.session_state.history.append(("bot", reply))
        else:
            results = []  # treat as no-match to trigger fallback
    # 2) If no retriever results, optional LLM fallback
    if not results:
        if OPENAI_KEY:
            # Use OpenAI to generate a helpful reply (simple prompt)
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini" if False else "gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful tech support assistant. Keep answers concise."},
                        {"role": "user", "content": user_query}
                    ],
                    max_tokens=200,
                    temperature=0.2,
                )
                gen = resp["choices"][0]["message"]["content"].strip()
                st.session_state.history.append(("bot", gen + "\n\nIf you'd like to escalate to a human agent, reply 'human' or click the 'Create Ticket' button."))
            except Exception as e:
                st.session_state.history.append(("bot", "Apologies — I couldn't generate an answer right now. Please type 'human' to create a ticket."))
        else:
            st.session_state.history.append(("bot", "I couldn't find a direct KB answer. Reply 'human' to create a support ticket, or provide more details."))

# Escalation controls: if user wants human
st.markdown("---")
if st.button("Create Ticket (escalate)"):
    # Ask for brief details
    customer_name = st.text_input("Your name or email (for follow-up):", key="ticket_name")
    category = st.selectbox("Category", ["General", "Network", "Software", "Account", "Hardware", "Other"], key="ticket_cat")
    details = st.text_area("Describe the issue (required):", key="ticket_details")
    if st.button("Submit Ticket"):
        if not customer_name or not details:
            st.error("Please provide your name/email and describe the issue.")
        else:
            tid = create_ticket(conn, customer_name, details, category=category)
            st.success(f"Ticket created — ID: {tid}. Our team will contact you.")
            st.session_state.history.append(("bot", f"Ticket created successfully. Your ticket ID is **{tid}**."))

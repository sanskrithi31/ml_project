import streamlit as st
import json
import os
import hashlib
from datetime import datetime
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import plotly.express as px  # <-- For calendar mood timeline

# ------------------ Constants ------------------

USERS_FILE = "users.json"
MOOD_LOG_FILE = "mood_log.csv"
JOURNAL_LOG_FILE = "journal_log.csv"
MODEL_FILE = "model.pkl"

SELF_CARE_TIPS = {
    "joy": "Celebrate your wins, share your happiness, and savor the moment!",
    "sadness": "Talk to a friend, write down your thoughts, or take a relaxing walk.",
    "anger": "Try deep breathing, journaling, or physical activity like walking.",
    "fear": "Focus on grounding techniques, like 5-4-3-2-1 or talk to someone you trust.",
    "surprise": "Reflect on the moment—was it good or unsettling? Accept and explore.",
    "neutral": "Take time to check in with yourself. Try mindfulness or light reading.",
}

# ------------------ Load Model & Embedder ------------------

model = joblib.load(MODEL_FILE)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ Auth Functions ------------------

def load_users():
    if os.path.exists(USERS_FILE) and os.path.getsize(USERS_FILE) > 0:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    users = load_users()
    return username in users and users[username] == hash_password(password)

def signup(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = hash_password(password)
    save_users(users)
    return True, "Signup successful! Please log in."

# ------------------ Mood Logging ------------------

def log_mood(text, emotion, username):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[timestamp, username, text, emotion]], columns=["timestamp", "username", "text", "emotion"])
    if os.path.exists(MOOD_LOG_FILE):
        df = pd.read_csv(MOOD_LOG_FILE)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data
    df.to_csv(MOOD_LOG_FILE, index=False)

def load_user_logs(username):
    if not os.path.exists(MOOD_LOG_FILE):
        return pd.DataFrame(columns=["timestamp", "username", "text", "emotion"])
    df = pd.read_csv(MOOD_LOG_FILE)
    return df[df["username"] == username]

# ------------------ Journal Logging ------------------

def log_journal(username, title, content):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[timestamp, username, title, content]], columns=["timestamp", "username", "title", "content"])
    if os.path.exists(JOURNAL_LOG_FILE):
        df = pd.read_csv(JOURNAL_LOG_FILE)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry
    df.to_csv(JOURNAL_LOG_FILE, index=False)

def load_journal(username):
    if not os.path.exists(JOURNAL_LOG_FILE):
        return pd.DataFrame(columns=["timestamp", "username", "title", "content"])
    df = pd.read_csv(JOURNAL_LOG_FILE)
    return df[df["username"] == username]

# ------------------ Visualization ------------------

def plot_emotion_pie_chart(df):
    if df.empty:
        st.info("No emotion data to show.")
        return
    emotion_counts = df["emotion"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.subheader("🧠 Emotion Distribution")
    st.pyplot(fig)

def plot_emotion_frequency(df):
    if df.empty:
        return
    freq = df["emotion"].value_counts()
    st.subheader("📊 Emotion Frequency")
    st.bar_chart(freq)

# def plot_mood_calendar(df):
#     if df.empty:
#         st.info("No mood data available to generate a timeline.")
#         return
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df["date"] = df["timestamp"].dt.date
#     calendar_df = df.groupby("date")["emotion"].apply(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
#     calendar_df["date"] = pd.to_datetime(calendar_df["date"])
#     calendar_df["emotion"] = calendar_df["emotion"].astype(str)

#     fig = px.scatter(
#         calendar_df,
#         x="date",
#         y=["emotion"] * len(calendar_df),
#         color="emotion",
#         title="🗓️ Emotion Timeline (Calendar View)",
#         labels={"emotion": "Emotion", "date": "Date"},
#         height=400
#     )
#     fig.update_traces(marker=dict(size=10))
#     fig.update_layout(showlegend=True)
#     st.plotly_chart(fig)


def plot_mood_timeline_detailed(df):
    if df.empty:
        st.info("No mood logs to display.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_sorted = df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df_sorted["timestamp"], df_sorted["emotion"], c="skyblue", s=100, edgecolors="k")
    ax.set_title("🕒 Mood Timeline (Detailed)", fontsize=14)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Emotion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.subheader("🕒 Mood Timeline (Detailed View)")
    st.pyplot(fig)

import plotly.express as px

def plot_emotion_calendar_view(df):
    if df.empty:
        st.info("No mood logs to display.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.strftime("%a")  # "Mon", "Tue", ...
    df["time"] = df["timestamp"].dt.strftime("%H:%M")

    fig = px.scatter(
        df,
        x="day",            # Weekday name
        y="emotion",        # Emotion category
        color="emotion",    # Color by emotion
        hover_data=["timestamp", "text"],  # 👈 FIXED: was 'note', now 'text'
        title="📅 Emotion Timeline (Calendar View)",
        labels={"day": "Day of Week", "emotion": "Emotion"},
        height=400
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    )

    st.plotly_chart(fig, use_container_width=True)




# ------------------ Streamlit App ------------------

st.set_page_config(page_title="AI Emotion Bot", page_icon="🧠")
st.title("🧠 AI Mental Health Emotion Classifier + Support Bot")

# ------------------ Auth System ------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""

if not st.session_state.authenticated:
    auth_tab = st.sidebar.radio("Account", ["Login", "Signup"])

    if auth_tab == "Signup":
        st.sidebar.subheader("Create a New Account")
        new_user = st.sidebar.text_input("Username", key="signup_user")
        new_pass = st.sidebar.text_input("Password", type="password", key="signup_pass")
        if st.sidebar.button("Sign up"):
            if not new_user or not new_pass:
                st.sidebar.warning("Please enter both username and password.")
            else:
                success, msg = signup(new_user, new_pass)
                st.sidebar.success(msg) if success else st.sidebar.error(msg)

    elif auth_tab == "Login":
        st.sidebar.subheader("Login")
        user = st.sidebar.text_input("Username", key="login_user")
        pwd = st.sidebar.text_input("Password", type="password", key="login_pass")
        if st.sidebar.button("Login"):
            if authenticate(user, pwd):
                st.session_state.authenticated = True
                st.session_state.current_user = user
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials.")
else:
    st.sidebar.write(f"🔓 Logged in as: **{st.session_state.current_user}**")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.current_user = ""
        st.success("You have been logged out.")
        st.rerun()

# ------------------ Main App ------------------

if st.session_state.authenticated:
    st.header(f"Welcome, {st.session_state.current_user} 👋")
    user_input = st.text_area("🗣️ How are you feeling today?")

    if st.button("Analyze Emotion"):
        if user_input.strip():
            embedding = embedder.encode([user_input])
            pred = model.predict(embedding)[0]
            st.success(f"🧠 Detected Emotion: **{pred}**")
            tip = SELF_CARE_TIPS.get(pred, "Take care of yourself today!")
            st.info(f"💡 Self-Care Tip: {tip}")
            log_mood(user_input, pred, st.session_state.current_user)
            st.rerun()
        else:
            st.warning("Please enter a message first.")

    user_df = load_user_logs(st.session_state.current_user)
    plot_emotion_pie_chart(user_df)
    plot_emotion_frequency(user_df)
    plot_mood_timeline_detailed(user_df)
    plot_emotion_calendar_view(user_df)
    #plot_mood_calendar(user_df)  # 👈 Add this line here


    st.subheader("📓 Personal Journal")

    with st.expander("➕ Write a Journal Entry"):
        title = st.text_input("Title")
        note = st.text_area("Write your thoughts here...")
        if st.button("Save Entry"):
            if title and note:
                log_journal(st.session_state.current_user, title, note)
                st.success("Journal entry saved!")
                st.rerun()
            else:
                st.warning("Please fill in both the title and note.")

    with st.expander("📖 View Your Journal Entries"):
        journal_df = load_journal(st.session_state.current_user)
        if journal_df.empty:
            st.info("No journal entries found.")
        else:
            for _, row in journal_df.iterrows():
                st.markdown(f"### 📝 {row['title']} ({row['timestamp']})")
                st.markdown(f"{row['content']}")
                st.markdown("---")

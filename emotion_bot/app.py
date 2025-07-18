import streamlit as st
import joblib
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Force CPU device to avoid "meta tensor" error
device = torch.device("cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Load trained model
model = joblib.load("model.pkl")

# Load support responses for each emotion
with open("support_responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)

# Emotion classes in same order as during training
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

# Streamlit App
st.set_page_config(page_title="AI Mental Health Support Bot", layout="centered")
st.title("🧠 AI Mental Health Support Bot")
st.markdown("Type how you feel, and the bot will understand your emotion and offer support.")

# Input box
user_input = st.text_area("💬 How are you feeling today?", height=150)

# Button
if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Generate BERT embedding
        with st.spinner("🔍 Analyzing..."):
            emb = embedder.encode([user_input])
            pred = model.predict(emb)[0]

        # Show result
        st.success(f"**Detected Emotion:** `{pred}`")

        # Supportive message
        st.markdown("### 💡 Support Response")
        st.info(responses.get(pred, "Stay strong. You're not alone."))

        # Optional: emoji
        emojis = {
            "anger": "😠",
            "disgust": "🤢",
            "fear": "😨",
            "joy": "😊",
            "sadness": "😢",
            "surprise": "😲"
        }
        st.markdown(f"### {emojis.get(pred, '🤖')}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, BERT & ML. Ready for placement.")

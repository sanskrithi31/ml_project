import streamlit as st
import joblib
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


device = torch.device("cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2")



model = joblib.load("model.pkl")


with open("support_responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)


emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


st.set_page_config(page_title="AI Mental Health Support Bot", layout="centered")
st.title("AI Mental Health Support Bot")
st.markdown("Type how you feel, and the bot will understand your emotion and offer support.")


user_input = st.text_area("How are you feeling today?", height=150)


if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        
        with st.spinner("Analyzing..."):
            emb = embedder.encode([user_input])
            pred = model.predict(emb)[0]

        
        st.success(f"**Detected Emotion:** `{pred}`")

        
        st.markdown("###  Support Response")
        st.info(responses.get(pred, "Stay strong. You're not alone."))

        
        # emojis = {
        #     "anger": "😠",
        #     "disgust": "🤢",
        #     "fear": "😨",
        #     "joy": "😊",
        #     "sadness": "😢",
        #     "surprise": "😲"
        # }
        # st.markdown(f"### {emojis.get(pred, '🤖')}")



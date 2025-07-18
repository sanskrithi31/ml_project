# preprocess_goemotions.py

import pandas as pd
from datasets import load_dataset

# Load dataset from HuggingFace
dataset = load_dataset("go_emotions", split="train")

# Original label ID → name mapping
label_map = dataset.features["labels"].feature.names

# Define 6 emotion buckets
emotion_map = {
    "joy": ["joy", "excitement", "amusement", "gratitude", "love", "pride"],
    "sadness": ["grief", "remorse", "disappointment", "sadness"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "disgust": ["disgust", "embarrassment"],
    "surprise": ["surprise", "realization", "curiosity"]
}

# Reverse label lookup for id → name
def get_label_name(label_id):
    return label_map[label_id]

# Map original label to 6-core class
def map_to_core_emotion(label_ids):
    if len(label_ids) != 1:
        return None  # Skip multi-label samples
    emotion = get_label_name(label_ids[0])
    for core_emotion, sub_labels in emotion_map.items():
        if emotion in sub_labels:
            return core_emotion
    return None

# Process dataset
texts = []
emotions = []

for entry in dataset:
    mapped = map_to_core_emotion(entry["labels"])
    if mapped:
        text = entry["text"].lower().strip()
        texts.append(text)
        emotions.append(mapped)

# Create DataFrame
df = pd.DataFrame({"text": texts, "emotion": emotions})

# Shuffle and save
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("goemotions_filtered.csv", index=False)

print("✅ Saved preprocessed dataset to goemotions_filtered.csv")
print(df["emotion"].value_counts())
# preprocess_goemotions.py


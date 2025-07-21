
import pandas as pd
from datasets import load_dataset


dataset = load_dataset("go_emotions", split="train")


label_map = dataset.features["labels"].feature.names

emotion_map = {
    "joy": ["joy", "excitement", "amusement", "gratitude", "love", "pride"],
    "sadness": ["grief", "remorse", "disappointment", "sadness"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "disgust": ["disgust", "embarrassment"],
    "surprise": ["surprise", "realization", "curiosity"]
}

def get_label_name(label_id):
    return label_map[label_id]


def map_to_core_emotion(label_ids):
    if len(label_ids) != 1:
        return None  
    emotion = get_label_name(label_ids[0])
    for core_emotion, sub_labels in emotion_map.items():
        if emotion in sub_labels:
            return core_emotion
    return None


texts = []
emotions = []

for entry in dataset:
    mapped = map_to_core_emotion(entry["labels"])
    if mapped:
        text = entry["text"].lower().strip()
        texts.append(text)
        emotions.append(mapped)


df = pd.DataFrame({"text": texts, "emotion": emotions})


df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("goemotions_filtered.csv", index=False)

print("Saved preprocessed dataset to goemotions_filtered.csv")
print(df["emotion"].value_counts())


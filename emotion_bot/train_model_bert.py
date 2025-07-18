

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer


df = pd.read_csv("goemotions_filtered.csv")


embedder = SentenceTransformer("all-MiniLM-L6-v2")



X = embedder.encode(df["text"].tolist(), show_progress_bar=True)
y = df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)



clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


print("Evaluation:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(clf, "model.pkl")
print("Model saved to model.pkl")

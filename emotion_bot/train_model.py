

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

data = {
    "text": [
        "I feel so happy today!", "I'm really anxious about my exams.",
        "Everything is frustrating me", "I love spending time with my friends",
        "I'm scared of what might happen tomorrow", "I’m feeling really down",
        "I'm super excited!", "I'm angry at how things turned out"
    ],
    "emotion": [
        "joy", "anxiety", "anger", "joy", "fear", "sadness", "joy", "anger"
    ]
}
df = pd.DataFrame(data)


X = df['text']
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))


joblib.dump(pipeline, 'model.pkl')


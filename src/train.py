import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Baca file
cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin-1', names=cols)
df['sentiment'] = df['sentiment'].replace(4, 1)

import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# df = df.sample(100000, random_state=42)

X = df['clean_text']
y = df['sentiment']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs('model', exist_ok=True)

joblib.dump(model, 'model/sentiment_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')


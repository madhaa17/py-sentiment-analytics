import sys
import joblib

# 1. Load model dan vectorizer
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# 2. Ambil input dari argumen terminal
if len(sys.argv) < 2:
    print("Masukkan teks review sebagai argumen")
    sys.exit()

input_text = " ".join(sys.argv[1:])

# 3. Transform teks
input_vec = vectorizer.transform([input_text])

# 4. Prediksi
pred = model.predict(input_vec)[0]
sentiment = "POSITIF" if pred == 1 else "NEGATIF"

print(f"\nReview: {input_text}")
print(f"Sentimen: {sentiment}")

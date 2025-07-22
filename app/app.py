import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re, string

# Load model dan vectorizer
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Konfigurasi Halaman
st.set_page_config(page_title="Sentiment Predictor", page_icon="💬", layout="wide")
st.title("💬 Sentiment Analysis")
st.markdown("Masukkan teks atau upload file CSV untuk menganalisis sentimen (positif atau negatif) dari ulasan.")

# ========== Data & Preprocessing ==========
cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
# df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin-1', names=cols)
df = pd.read_csv('../sample/sample_reviews.csv', encoding='latin-1', names=cols)
df['sentiment'] = df['sentiment'].replace(4, 1)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# ========== Layout ==========
tab1, tab2, tab3 = st.tabs(["🧾 Prediksi Manual", "📂 Batch Prediksi", "📊 Eksplorasi Data"])

# ======= TAB 1 - Prediksi Manual ========
with tab1:
    st.subheader("Masukkan Teks Ulasan")
    user_input = st.text_area("Tulis ulasan di bawah ini:", "")
    
    if st.button("🔍 Prediksi"):
        if user_input.strip() == "":
            st.warning("Tolong isi teks terlebih dahulu.")
        else:
            input_vec = vectorizer.transform([clean_text(user_input)])
            prediction = model.predict(input_vec)[0]
            sentiment = "✅ POSITIF" if prediction == 1 else "❌ NEGATIF"
            st.success(f"**Sentimen:** {sentiment}")

# ======= TAB 2 - Batch Prediksi =========
with tab2:
    st.subheader("Upload File CSV")
    st.markdown("File harus memiliki kolom bernama `review`.")

    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if 'review' not in df_upload.columns:
                st.error("❌ Kolom 'review' tidak ditemukan.")
            else:
                df_upload['clean_review'] = df_upload['review'].astype(str).apply(clean_text)
                X_input = vectorizer.transform(df_upload['clean_review'])
                preds = model.predict(X_input)
                df_upload['sentiment'] = preds
                df_upload['sentiment_label'] = df_upload['sentiment'].map({1: '✅ POSITIF', 0: '❌ NEGATIF'})

                st.success("✅ Prediksi berhasil!")
                st.dataframe(df_upload[['review', 'sentiment_label']])

                csv = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Hasil", csv, file_name='hasil_prediksi.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Error saat membaca file: {e}")

# ======= TAB 3 - Eksplorasi Data ========
with tab3:
    st.subheader("Eksplorasi Data Latih")
    with st.expander("📌 WordCloud"):
        sentiment_option = st.radio("Pilih Sentimen", ["Positif", "Negatif"], horizontal=True)
        selected_sentiment = 1 if sentiment_option == "Positif" else 0
        text = " ".join(df[df['sentiment'] == selected_sentiment]['clean_text'].dropna().tolist()[:5000])

        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    with st.expander("📌 Distribusi Sentimen"):
        st.bar_chart(df['sentiment'].value_counts().rename({0: 'Negatif', 1: 'Positif'}))

# 💬 Sentiment Analysis Web App

A machine learning web application for predicting the sentiment (positive or negative) from user reviews using **Logistic Regression** and **TF-IDF Vectorization**, built with **Streamlit**. This project utilizes the popular **Sentiment140** dataset, containing 1.6 million labeled tweets.


---

## 🚀 Features

- 🔎 Predict sentiment for a single text input
- 📂 Batch prediction from CSV file
- 🌥 Generate WordCloud for positive/negative sentiments
- 📊 Sentiment distribution visualization
- 🎯 Model trained on 320,000 sample tweets
- 🧠 Logistic Regression with TF-IDF vectorizer

---

## 🧰 Tech Stack

- Python 3.12
- Streamlit
- scikit-learn
- pandas
- matplotlib, wordcloud
- joblib

---

## 📦 Installation (Local)

### 1. Clone Repository

```bash
git clone https://github.com/madhaa17/py-sentiment-analytics.git
cd py-sentiment-analytics
```
### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run Streamlit App
```bash
streamlit run app.py
```

# (Optional) Install kaggle CLI
```bash
pip install kaggle
```

# Let’s say your kaggle.json API key is ready:
```bash
mkdir -p ~/.kaggle
cp path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

# Download the dataset directly to /data
```bash
kaggle datasets download -d kazanova/sentiment140 -p data/
unzip data/sentiment140.zip -d data/
```


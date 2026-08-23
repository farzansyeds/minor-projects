# IMDb Sentiment Analysis using PyTorch RNN

An end-to-end NLP pipeline for binary sentiment classification (Positive/Negative) on the IMDb 50k movie reviews dataset.

---

## 📌 Architecture & Pipeline
* **Preprocessing**: HTML stripping, URL removal, punctuation filtering, stopword elimination, and Porter Stemming with NLTK.
* **Feature Extraction**: Scikit-Learn `TfidfVectorizer` (top 5,000 features).
* **Model**: Single-layer `nn.RNN` with a linear classification head (`nn.Linear`).
* **Optimization**: `BCELoss` with `Adam` optimizer.
* **Accuracy**: **85.16%** on unseen test data after 10 epochs.

---

## 🚀 How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
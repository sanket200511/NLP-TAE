# train_models.py
import sys
import io
import pandas as pd
import numpy as np
from indic_preprocessor import clean_indic_text, tokenize_indic
from model_engine import IndiToxModelEngine, LABEL_COLS, compute_multilabel_metrics

# Set utf-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading Indian comments dataset...")
df = pd.read_csv("data/indian_toxic_comments.csv")
print(f"Loaded {len(df)} comments.")

print("Preprocessing comments...")
df["cleaned_text"] = df["comment_text"].apply(clean_indic_text)
texts = df["cleaned_text"].tolist()
y = df[LABEL_COLS].values

engine = IndiToxModelEngine()

print("Training TF-IDF Vectorizer...")
vec, X_tfidf = engine.train_tfidf(texts)

print("Training FastText subword model...")
sentences = [tokenize_indic(t) for t in texts]
ft = engine.train_fasttext(sentences, vector_size=100, epochs=15)

print("Training Logistic Regression and Random Forest on TF-IDF...")
lr_tfidf, rf_tfidf = engine.train_models(X_tfidf, y, feature_mode="tfidf")

# FastText embeddings
X_ft = np.vstack([engine.embed_comment_fasttext(t) for t in texts])
print("Training Logistic Regression and Random Forest on FastText...")
lr_ft, rf_ft = engine.train_models(X_ft, y, feature_mode="fasttext")

print("Evaluating TF-IDF models:")
pred_lr = lr_tfidf.predict(X_tfidf)
pred_rf = rf_tfidf.predict(X_tfidf)
print("LR Metrics:", compute_multilabel_metrics(y, pred_lr))
print("RF Metrics:", compute_multilabel_metrics(y, pred_rf))

print("Testing hybrid prediction on sample Indian comments:")
samples = [
    "नमस्ते सर, बहुत अच्छा वीडियो था।",
    "Tu bilkul pagal hai kya? dimag nahi hai.",
    "Chup bsdk, teri maa chod dunga.",
    "I will track your IP and kill you.",
    "Super anna, chala bagundi video!"
]
for s in samples:
    res = engine.predict_hybrid(s, model_type="Logistic Regression", feature_mode="tfidf")
    print(f"\nText: {s}")
    print(f"Detected Lang: {res['detected_language']} | Toxic: {res['is_toxic']} | Score: {res['overall_score']:.2f}")
    print(f"Predictions: {res['binary_predictions']}")

print("\nModel training & verification complete!")

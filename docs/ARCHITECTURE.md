# 🧠 IndiTox 2.0: System Architecture & Implementation

This document describes the engineering pipeline, text representations, and classifiers driving **IndiTox 2.0**.

---

## 🏗️ End-to-End Pipeline

```
[User Input Comment]
       │
       ▼
[Indic Preprocessor] ──> Unicode Normalization & Character Elongation Reduction
       │
       ▼
[Language Detector] ──> Auto-identifies script (Devanagari/Dravidian) vs. Latin
       │
       ▼
[Feature Extraction] ──> Multi-Ngram TF-IDF & FastText Subwords
       │
       ▼
[Classifier Ensemble] ──> OvR Logistic Regression + Random Forest + Lexicon Blender
       │
       ▼
[Response Output] ──> Severity score, highlighted text, and polite rewrite options
```

---

## 🧼 1. Indic Preprocessor Engine

Located in [`indic_preprocessor.py`](file:///d:/Projects/Toxic-Comment-Detector/indic_preprocessor.py), this component handles noisy multilingual social media comments:

1. **Character Elongation Compression**:
   - Compresses repeated characters (e.g., `paaaaagaaaal` $\rightarrow$ `pagal`, `kuttaaa` $\rightarrow$ `kutta`).
2. **Obfuscation Mapping**:
   - Replaces common masked abusive slurs (e.g., `b*ch` $\rightarrow$ `bitch`, `b@kwas` $\rightarrow$ `bakwas`, `bsdk`).
3. **Unicode Script Filtering**:
   - Preserves alphanumeric characters, common symbols, and native Unicode ranges:
     - Devanagari (Hindi/Marathi): `\u0900-\u097F`
     - Tamil: `\u0B80-\u0BFF`
     - Telugu: `\u0C00-\u0C7F`
     - Kannada: `\u0C80-\u0CFF`
     - Malayalam: `\u0D00-\u0D7F`
4. **Multilingual Stopword Removal**:
   - Filters out high-frequency non-semantic terms from [`data/stopwords_indic.txt`](file:///d:/Projects/Toxic-Comment-Detector/data/stopwords_indic.txt).

---

## 📊 2. Feature Representation

We capture textual signals through two complementary methods:

1. **Multi-Granular TF-IDF**:
   - Word $1$-$3$ ngrams capture semantic phrase contexts.
   - Built with sublinear term-frequency scaling to prevent long spam comments from skewing weights.
2. **Subword FastText Embeddings**:
   - Uses subword n-gram matrices to capture vocabulary variations.
   - Ensures robust vector representation even for typo-filled or unseen slang.

---

## 🤖 3. Model Engine & Hybrid Inference

Implemented in [`model_engine.py`](file:///d:/Projects/Toxic-Comment-Detector/model_engine.py):

1. **OneVsRest Logistic Regression**:
   - Configured with balanced class weights to compensate for label sparsity.
   - Optimized regularization ($C=3.5$) for peak multi-label macro F1 performance.
2. **Ensemble Random Forest**:
   - Uses balanced sub-sampling across trees to handle skewed distributions.
3. **Lexicon-Sensitivity Blender**:
   - Blends machine learning probabilities with rule-based scores from [`indic_lexicon.py`](file:///d:/Projects/Toxic-Comment-Detector/indic_lexicon.py) to achieve high recall on zero-day slurs and threats.

# 🇮🇳 IndiTox 2.0: Indian Multilingual Toxic Comment Detector

**Author**: Sanket Kurve (USN: CS23121)  
**Project Type**: NLP TAE / Natural Language Processing  
**Reference & Foundation**: [SaiVivek7/Toxic-comment-classification](https://github.com/SaiVivek7/Toxic-comment-classification.git) (Mahindra University NLP Course)

---

## 🌟 Executive Summary

**IndiTox 2.0** is an end-to-end NLP and Machine Learning system specifically engineered for detecting **toxic comments in Indian social media spaces**. Unlike traditional English-centric models, IndiTox 2.0 handles the linguistic complexities of the Indian digital sphere, including **Native Indic Scripts** (Devanagari, Tamil, Telugu, Malayalam, Kannada), **Romanized Hinglish / Code-Mixed text**, character elongations, masked vulgarities, and regional profanities.

### 🌐 Supported Indian Languages
1. **Hindi (हिन्दी)** — Devanagari script processing & stopword filtering
2. **Hinglish (Code-Mixed)** — Romanized Hindi slang, abusive phrases, and conversational discourse
3. **Telugu (తెలుగు / Tenglish)** — Native script and Romanized Dravidian comments
4. **Tamil (தமிழ் / Tanglish)** — Native script and Romanized Tamil insults & feedback
5. **Malayalam (മലയാളം / Manglish)** — Regional comments & abusive terminology
6. **Kannada (ಕನ್ನಡ / Kanglish)** — Regional comments & moderation
7. **Indian English** — Socio-political discussions, cyberbullying, and hate speech

---

## 🛡️ Multi-Label Toxicity Taxonomy
IndiTox 2.0 classifies comments across **6 standard multi-label categories**:
- **Toxic (`toxic`)**: General hostility, rudeness, trolling, or aggressive tone.
- **Severe Toxic (`severe_toxic`)**: Extremely vulgar attacks, high-severity offensive slurs.
- **Obscene (`obscene`)**: Sexual vulgarity, anatomical profanity, and abusive slang.
- **Threat (`threat`)**: Direct threats of violence, physical harm, or murder.
- **Insult (`insult`)**: Derogatory personal name-calling and mocking.
- **Identity Hate (`identity_hate`)**: Communal, religious, casteist, regional, or ethnic hate speech.

---

## 🧠 System Architecture & NLP Pipeline

```
[User Comment] (Hindi / Hinglish / Telugu / Tamil / Kannada / Malayalam / English)
       │
       ▼
[Indic Preprocessing Engine]
 ├── Unicode Script Normalization (Devanagari, Dravidian, Latin)
 ├── De-obfuscation (Masked profanities: b@kwas, b*ch, bsdk)
 ├── Elongation Compression (bhaaaai -> bhai, kuttaaa -> kutta)
 └── Indic & Hinglish Stopword Filtering
       │
       ▼
[Language & Script Detection] ──> Auto-identifies source language
       │
       ▼
[Feature Extraction & Embeddings]
 ├── FastText Subword Embeddings (char n-grams 2-6 for typo resilience)
 └── Multi-lingual TF-IDF Vectorizer (word 1-3 & char n-grams)
       │
       ▼
[Classification Models & Hybrid Inference]
 ├── Logistic Regression (OneVsRest with Balanced Class Weights)
 ├── Random Forest Ensemble Classifier
 └── Indic Toxic Lexicon Sensitivity Blending
       │
       ▼
[Explainability & Polite Rephrase Engine]
 ├── Token-level Toxic Span Highlighting
 └── Constructive Civil Rephrase Suggestions
```

---

## 🚀 Key Features in the Streamlit Web Application

1. **🏠 Home & Overview**: Project highlights, taxonomy breakdown, and pipeline flowchart.
2. **📊 Indian Multilingual EDA**: Language distribution charts, clean vs. toxic proportions, label correlation heatmaps, and multilingual n-gram explorer.
3. **🧼 Indic Preprocessing Studio**: Real-time inspection of Unicode preservation, character elongation reduction, and stopword removal.
4. **🤖 Model Benchmarks & Training**: Interactive performance evaluation with Macro/Micro F1 scores, Hamming Loss, 2x2 confusion matrices, and ROC-AUC curves.
5. **🔍 Live Comment Detector**:
   - Quick-try presets for 7 Indian languages.
   - Dynamic Language & Script badge.
   - Real-time overall toxicity gauge and per-category confidence pills.
   - **Explainability Span Highlighter**: Visualizes toxic trigger words.
   - **Polite Rephrase Suggester**: Suggests respectful, constructive alternatives.
6. **📁 Batch CSV Moderation**: Upload any comment feed (YouTube/Twitter/Instagram) to generate exportable moderation reports (`inditox_moderation_report.csv`).

---

## 💻 Installation & Usage

### 1. Prerequisites & Virtual Environment
```bash
# Clone or navigate to the repository
cd Toxic-Comment-Detector

# Activate the virtual environment (Windows)
.\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Pre-train the Models
```bash
python train_models.py
```

### 4. Launch the Interactive Web Application
```bash
streamlit run main.py
```

---

## 📊 Performance & Evaluation Highlights

| Model Architecture | Macro F1 | Micro F1 | Weighted F1 | Hamming Loss |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression (OvR)** | **0.992** | **0.994** | **0.993** | **0.003** |
| **FastText Subwords + Random Forest** | **0.943** | **0.972** | **0.969** | **0.018** |

---

## 📜 Acknowledgments
- Inspired by the research and repository of **Vivek et al.** ([Toxic-comment-classification](https://github.com/SaiVivek7/Toxic-comment-classification.git)) at Mahindra University.
- Supported by the **Jigsaw & Kaggle** toxic comment challenge standards adapted for Indian regional language spaces.

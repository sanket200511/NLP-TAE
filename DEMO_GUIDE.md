# 🇮🇳 IndiTox 2.0: Complete Testing & Live Demo Guide

**Project**: Indian Multilingual Toxic Comment Detector  
**Author**: Sanket Kurve (USN: CS23121)  
**Repository**: [https://github.com/sanket200511/NLP-TAE](https://github.com/sanket200511/NLP-TAE)

---

## ⚡ Quick Start: How to Run the Project

### 1. Open Terminal in the Project Directory
```powershell
cd d:\Projects\Toxic-Comment-Detector
```

### 2. Activate Virtual Environment
```powershell
.\venv\Scripts\activate
```

### 3. (Optional) Re-train Models on Indian Dataset
```powershell
python train_models.py
```

### 4. Launch the Streamlit Web Application
```powershell
streamlit run main.py
```
> The application will open automatically in your default browser at **`http://localhost:8501`**.

---

## 🎬 Step-by-Step Live Demo & Presentation Script

Follow this structured order when presenting to evaluators, professors, or team members:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Home & Overview      ➜ Introduce problem & architecture  │
│ 2. Preprocessing Studio ➜ Show Indian slang/script cleaning │
│ 3. Multilingual EDA     ➜ Showcase Indian comment analytics │
│ 4. Model Benchmarks     ➜ Present F1, ROC-AUC & Confusions  │
│ 5. Live Detector        ➜ Test Hindi/Hinglish/Regional text │
│ 6. Batch Moderation     ➜ Upload CSV & download report      │
└─────────────────────────────────────────────────────────────┘
```

---

### 📍 Stage 1: Home & Overview (`🏠 Home & Overview`)
- **Key Talking Point**: Explain that traditional toxicity models fail on Indian social media because they only process ASCII English and break on Devanagari, Dravidian scripts, and Hinglish slang.
- **Showcase**:
  - The **6 Multi-label Categories** (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).
  - The **7 Supported Languages Matrix** (Hindi, Hinglish, Telugu, Tamil, Malayalam, Kannada, Indian English).
  - The end-to-end architecture pipeline card.

---

### 📍 Stage 2: Indic Preprocessing Studio (`🧼 Indic Preprocessing Studio`)
- **Key Talking Point**: Show the NLP innovations used to handle noisy Indian text before it reaches ML models.
- **Live Demo Test 1**: Masked Profanity & Elongations
  - **Input Text**: `Arre bhaaaai tu kitna b@kwaas video banata hai, bilkul pagal hai kya??? 😡😡`
  - **Highlight**:
    - Automatic **Hinglish** language detection badge.
    - De-obfuscation of `b@kwaas` $\rightarrow$ `bakwas`.
    - Elongation compression of `bhaaaai` $\rightarrow$ `bhai`.
    - Script preservation keeping Indic characters intact without stripping them.

---

### 📍 Stage 3: Indian Multilingual EDA (`📊 Indian Multilingual EDA`)
- **Key Talking Point**: Demonstrate deep data analysis of Indian social media commentary.
- **Showcase**:
  - **Tab 1 (Overview)**: Total rows, clean vs. toxic ratio metrics.
  - **Tab 2 (Languages)**: Distribution across Hindi, Hinglish, Telugu, Tamil, Malayalam, Kannada.
  - **Tab 3 (Labels & Correlations)**: Correlation between `insult` and `obscene` (~0.7).
  - **Tab 4 (N-Gram Explorer)**: Switch between unigrams, bigrams, and trigrams in Indian languages.

---

### 📍 Stage 4: Model Benchmarks (`🤖 Model Benchmarks & Training`)
- **Key Talking Point**: Compare TF-IDF and FastText subword representations with Logistic Regression (OvR) and Random Forest classifiers.
- **Showcase**:
  - **Macro F1 & Hamming Loss** metrics cards.
  - **Per-Label Classification Report**: Precision, Recall, and F1 across all 6 labels.
  - **2x2 Confusion Matrices Heatmaps** for each toxicity label.
  - **Interactive Altair ROC-AUC Curves**.

---

### 📍 Stage 5: Live Comment Detector (`🔍 Live Comment Detector`)

Use these curated test cases to demonstrate real-time classification across languages:

#### Test Case 1: Hinglish Violent Threat + Abuse
- **Input**: `Tere ghar aake tujhe jaan se maar dalunga kutte, police bhi nahi bacha payegi.`
- **Expected Result**:
  - Status: **⚠️ TOXIC DETECTED (100% Severity)**
  - Detected Language: `🌐 Hinglish`
  - Active Labels: `TOXIC`, `SEVERE TOXIC`, `THREAT`, `INSULT`
  - Span Highlighter: Flags `jaan se maar dalunga` and `kutte`
  - Polite Rephrase: Suggests constructive alternative phrasing.

#### Test Case 2: Hindi (Devanagari Script) Obscene Insult
- **Input**: `चुप कर गधे, तुझे कुछ नहीं पता, बकवास बंद कर अपनी।`
- **Expected Result**:
  - Status: **⚠️ TOXIC DETECTED (88% Severity)**
  - Detected Language: `🌐 Hindi`
  - Active Labels: `TOXIC`, `INSULT`
  - Span Highlighter: Flags `गधे` and `बकवास`.

#### Test Case 3: South Indian Regional Insult (Tamil)
- **Input**: `Loose madhiri pesadha, unakku onnum theriyaadhu muttal.`
- **Expected Result**:
  - Status: **⚠️ TOXIC DETECTED**
  - Detected Language: `🌐 Tamil (Romanized)`
  - Active Labels: `TOXIC`, `INSULT`
  - Span Highlighter: Flags `muttal` (fool).

#### Test Case 4: Positive / Clean Indian Comment (Hinglish)
- **Input**: `Bhai aapka explanation bohot badhiya tha, sab samajh aa gaya!`
- **Expected Result**:
  - Status: **✅ CLEAN / SAFE (38% Severity)**
  - Detected Language: `🌐 Hinglish`
  - Active Labels: All `CLEAR`.

#### Test Case 5: Positive / Clean Hindi Devanagari Comment
- **Input**: `नमस्ते सर, आपका यह वीडियो बहुत ज्ञानवर्धक और उपयोगी था। धन्यवाद!`
- **Expected Result**:
  - Status: **✅ CLEAN / SAFE (38% Severity)**
  - Detected Language: `🌐 Hindi`
  - Active Labels: All `CLEAR`.

---

### 📍 Stage 6: Batch CSV Moderation (`📁 Batch CSV Moderation`)
- **Key Talking Point**: Show production readiness for automated YouTube/Instagram comment moderation feeds.
- **Showcase**:
  - Click **🚀 Run Batch Moderation**.
  - Show summary metrics (Total rows, % Toxic, Avg severity).
  - Show per-row prediction columns added (`detected_language`, `is_toxic`, `severity_score`, `prob_threat`, etc.).
  - Click **⬇️ Download Moderation Report (CSV)** to download the annotated CSV.

---

## 🎯 Common Q&A for Academic / Viva Defense

### Q1: How does IndiTox handle words it has never seen before (Out-Of-Vocabulary)?
> **Answer**: IndiTox uses **FastText Subword Embeddings** (character n-grams 2–6) and **Character TF-IDF**. When an unfamiliar or misspelled Indian slang word appears (e.g. `kuttaa`, `bhaaaai`), the model decomposes it into character n-gram subwords, retaining semantic meaning.

### Q2: Why is multi-label classification necessary instead of binary classification?
> **Answer**: Online comments often contain multiple distinct forms of toxicity simultaneously (e.g., a comment can be simultaneously an `insult`, `obscene`, and a `threat`). Multi-label classification (using OneVsRest Logistic Regression and Multi-Output Random Forest) evaluates each dimension independently.

### Q3: How did you solve class imbalance (e.g., rare threats vs common insults)?
> **Answer**: We applied **Balanced Class Weighting** (`class_weight="balanced"`) in Logistic Regression and tuned prediction probability thresholds with hybrid lexicon sensitivity.

---

## 📁 File Structure Reference
```
Toxic-Comment-Detector/
├── .vscode/settings.json       # IDE Python interpreter path
├── data/
│   ├── indian_toxic_comments.csv # Multi-lingual Indian dataset
│   └── stopwords_indic.txt     # Indic & Hinglish stopwords
├── models/                     # Saved ML models & vectorizers
├── indic_preprocessor.py       # Multi-script cleaning & language detection
├── indic_lexicon.py            # Toxic lexicon, highlighter & rephraser
├── model_engine.py             # FastText, TF-IDF, LR, RF & hybrid inference
├── generate_dataset.py         # Dataset generator script
├── train_models.py             # Pre-training and evaluation script
├── main.py                     # Streamlit interactive web dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview & documentation
└── DEMO_GUIDE.md               # Complete testing & presentation script
```

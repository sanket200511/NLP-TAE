# 🇮🇳 IndiTox 2.0: Indian Multilingual Toxic Comment Detector

**Author**: Sanket Kurve (USN: CS23121)  
**Course Project**: NLP TAE / Natural Language Processing  
**Repository**: [https://github.com/sanket200511/NLP-TAE](https://github.com/sanket200511/NLP-TAE)

---

## 🌟 Executive Summary

**IndiTox 2.0** is an end-to-end Natural Language Processing (NLP) system designed to detect toxic comments across Indian social media spaces. Traditional moderation tools fail because they only support English text and break on native Indian scripts or code-mixed Hinglish slang.

IndiTox 2.0 natively supports **8 Indian languages and scripts**, filters out regional slurs, and recommends polite rephrasing alternatives in real-time.

---

## 🌐 Supported Languages & Scripts

1. **Hindi (हिन्दी)** — Devanagari script processing and stopwords.
2. **Hinglish (Code-Mixed)** — Romanized Hindi slang and conversational triggers.
3. **Marathi (मराठी)** — Marathi Devanagari script and Romanized Marathi slang.
4. **Telugu (తెలుగు)** — Telugu native script and Romanized Tenglish insults.
5. **Tamil (தமிழ்)** — Tamil native script and Romanized Tanglish insults.
6. **Malayalam (മലയാളം)** — Malayalam native script and Manglish terms.
7. **Kannada (ಕನ್ನಡ)** — Kannada native script and Kanglish terms.
8. **Indian English** — Socio-political discussions and cyberbullying text.

---

## 🛡️ Multi-Label Toxicity Classification

Comments are categorized across **6 standard labels**:
- **Toxic (`toxic`)**: General hostility, trolling, or aggressive tone.
- **Severe Toxic (`severe_toxic`)**: Extremely vulgar insults or slurs.
- **Obscene (`obscene`)**: Sexual profanity or anatomical vulgarities.
- **Threat (`threat`)**: Direct threats of violence, physical harm, or murder.
- **Insult (`insult`)**: Derogatory personal attacks and mocking.
- **Identity Hate (`identity_hate`)**: Hate speech targeting caste, religion, region, or gender.

---

## 💻 Quick Start & Running Instructions

To run the web application locally:

### 1. Open Terminal & Navigate to Project
```powershell
cd d:\Projects\Toxic-Comment-Detector
```

### 2. Activate Virtual Environment
```powershell
.\venv\Scripts\activate
```

### 3. Launch Flask Web Application
```powershell
python app.py
```

### 4. Visit in Browser
Open **`http://localhost:8080`** in your browser.

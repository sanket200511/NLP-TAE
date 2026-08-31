# 📊 IndiTox 2.0: Dataset Specifications

This document outlines the properties, schema, and generation specifications of the benchmark dataset used in **IndiTox 2.0**.

---

## 📈 Dataset Overview

The dataset is saved at [`data/indian_toxic_comments.csv`](file:///d:/Projects/Toxic-Comment-Detector/data/indian_toxic_comments.csv) and contains **546 comments** representing real-world social media discourse in India:
- **Clean Comments**: Positive feedback, greetings, technical discussions, and constructive queries in native scripts and Romanized text.
- **Toxic Comments**: Hostile remarks, vulgarities, hate speech, threats, and insults.

---

## 📋 Schema Columns

| Column | Type | Description |
| :--- | :---: | :--- |
| `comment_text` | String | Raw text content of the social media comment. |
| `language` | String | Labeled primary script/language (e.g. Hindi, Hinglish, Marathi, Telugu, Tamil, Malayalam, Kannada, Indian_English). |
| `toxic` | Integer | Binary classification for general toxicity ($1$ = Yes, $0$ = No). |
| `severe_toxic` | Integer | Binary classification for extreme vulgarity or slurs. |
| `obscene` | Integer | Binary classification for obscene/sexual terms. |
| `threat` | Integer | Binary classification for threats of violence/murder. |
| `insult` | Integer | Binary classification for personal mockery or insults. |
| `identity_hate` | Integer | Binary classification for casteist, communal, or ethnic hate speech. |

---

## 🛠️ Data Generation & Augmentation

The dataset is generated via [`generate_dataset.py`](file:///d:/Projects/Toxic-Comment-Detector/generate_dataset.py) using seed comments representing native and Romanized code-mixed expressions. 

### Augmentation Rules:
1. **Salutations**: Appends prefixes like `"Hey "`, `"Sir, "` to clean comments to verify model calibration on formal and informal openings.
2. **Politeness Markers**: Appends suffixes like `" please"`, `" 🙏"` to confirm the model does not trigger false positives on respectful language.
3. **Class Balancing**: Multiplies toxic classes by balanced subsets to ensure the ensemble classifier trains effectively on sparse categories.

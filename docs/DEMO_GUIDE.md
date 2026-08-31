# 🎬 IndiTox 2.0: Live Demo & Testing Script Guide

This guide provides steps for running live demo tests and evaluating predictions across native scripts and Romanized code-mixed comments.

---

## 🚀 Launching the Demo

1. **Activate Environment**:
   ```powershell
   .\venv\Scripts\activate
   ```
2. **Start Flask Server**:
   ```powershell
   python app.py
   ```
3. Open **`http://localhost:8080`** in your browser.

---

## 🧪 Single Comment Test Scenarios

Copy and paste the following sample comments into the input field to test different languages:

### 1. Hindi (Devanagari)
- **Safe Input**: `नमस्ते सर, बहुत ज्ञानवर्धक वीडियो था। धन्यवाद!`
  - *Expected Result*: Clean (0% severity, status green).
- **Toxic Input**: `ये मुल्ले देशद्रोही हैं, इनको खत्म करो ये देश के दुश्मन हैं।`
  - *Expected Result*: Flagged Toxic (`toxic`, `threat`, `identity_hate`).

### 2. Hinglish (Romanized Hindi)
- **Safe Input**: `bhai aapka explanation bohot clear aur easy tha.`
  - *Expected Result*: Clean (0% severity).
- **Toxic Input**: `Tu bilkul pagal hai kya? dimag nahi hai bsdk.`
  - *Expected Result*: Flagged Toxic (`toxic`, `obscene`, `insult`). Renders polite rephrasing suggestion cards.

### 3. Marathi (Devanagari & Romanized)
- **Safe Input**: `नवीन व्हिडिओ खूप छान आहे भाऊ, खूप माहिती मिळाली.`
  - *Expected Result*: Clean (0% severity). Detected as `Marathi`.
- **Toxic Input**: `tu murkha aahes ka? kasa boltoy samajhta ka.`
  - *Expected Result*: Flagged Toxic (`toxic`, `insult`). Detected as `Marathi (Romanized)`.

### 4. Telugu (Dravidian Script)
- **Safe Input**: `Super anna, chala bagundi video!`
  - *Expected Result*: Clean (0% severity).
- **Toxic Input**: `chala vedhava video idi, lanja deng.`
  - *Expected Result*: Flagged Toxic (`toxic`, `obscene`, `insult`).

---

## 📁 Batch Moderation Demo

To test batch processing:
1. Switch to the **Batch Moderation** tab.
2. Drag and drop a sample comment CSV file (or upload from `data/indian_toxic_comments.csv`).
3. Select the comment column from the dropdown (e.g., `comment_text`).
4. Click **Process Batch**.
5. Inspect the live summary cards (Total rows, Flagged toxic, Clean count) and preview rows.
6. Click **Download Moderation Report** to retrieve the annotated CSV.

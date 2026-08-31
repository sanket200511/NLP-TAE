# main.py
"""
🇮🇳 IndiTox 2.0: Indian Multilingual Toxic Comment Detector
Tailored for Indian Social Media Comments across Hindi, Hinglish, Telugu, Tamil, Malayalam, Kannada, and Indian English.
Features: FastText Subwords + TF-IDF + Logistic Regression / Random Forest + Lexicon Explainability & Polite Rephraser.
"""

import sys
import io
import re
import string
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# Suppress Altair/narwhals version deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="altair")
warnings.filterwarnings("ignore", module="narwhals")

import altair as alt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Custom modules
from indic_preprocessor import (
    clean_indic_text,
    tokenize_indic,
    detect_indic_language,
    get_indic_stopwords,
    deobfuscate_text,
    normalize_repeated_chars
)
from indic_lexicon import (
    highlight_toxic_spans,
    suggest_polite_alternatives,
    find_toxic_terms,
    INDIC_TOXIC_LEXICON
)
from model_engine import (
    IndiToxModelEngine,
    LABEL_COLS,
    compute_multilabel_metrics,
    compute_detailed_evaluations
)

# Optional wordcloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="IndiTox: Indian Toxic Comment Detector",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/sanket200511/NLP-TAE",
        "Report a bug": "mailto:sanketkurve.2005@gmail.com",
        "About": "🇮🇳 **IndiTox 2.0**: Multilingual Toxic Comment Detection for Indian Social Media.\n\nAuthor: Sanket Kurve (CS23121)\nSupports Hindi, Hinglish, Telugu, Tamil, Malayalam, Kannada, and Indian English."
    }
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "indian_toxic_comments.csv"

# ==========================================
# Custom CSS (Theme-Adaptive Sleek Glassmorphism)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero Header */
.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    color: white;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.15);
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 12px;
    background: linear-gradient(90deg, #ff9933, #ffffff, #138808);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.05rem;
    color: #e2e8f0;
    margin-bottom: 14px;
}
.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.hero-tag {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #f8fafc;
}

/* Feature & Step Cards - Theme Adaptive */
.step-card {
    background: rgba(125, 125, 125, 0.07);
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(125, 125, 125, 0.2);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    height: 100%;
    transition: all 0.2s ease-in-out;
}
.step-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    border-color: #6366f1;
}
.step-num {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    color: #818cf8;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.step-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.step-desc {
    font-size: 0.9rem;
    opacity: 0.85;
    line-height: 1.4;
}

/* Toxicity Indicators & Badges */
.badge-clean {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.badge-toxic {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.badge-lang {
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
}

/* Result box */
.res-box {
    background: rgba(125, 125, 125, 0.08);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(125, 125, 125, 0.2);
    margin-top: 14px;
}

/* Footer */
footer { visibility: hidden; }
.inditox-footer {
    position: fixed; left:0; bottom:0; width:100%;
    background: #0f172a; color:#94a3b8; text-align:center;
    padding: 8px 12px; font-size: 13px; z-index: 9999;
    border-top: 1px solid rgba(255,255,255,0.1);
}
.inditox-footer strong { color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Initialize Session State
# ==========================================
def init_app_state():
    if "engine" not in st.session_state:
        st.session_state.engine = IndiToxModelEngine()
    if "df_indian" not in st.session_state:
        if DATA_PATH.exists():
            st.session_state.df_indian = pd.read_csv(DATA_PATH)
        else:
            st.session_state.df_indian = None

init_app_state()


# ==========================================
# Sidebar Navigation
# ==========================================
def render_sidebar():
    st.sidebar.markdown("### 🇮🇳 IndiTox Navigation")
    pages = [
        "🏠 Home & Overview",
        "📊 Indian Multilingual EDA",
        "🧼 Indic Preprocessing Studio",
        "🤖 Model Benchmarks & Training",
        "🔍 Live Comment Detector",
        "📁 Batch CSV Moderation"
    ]
    choice = st.sidebar.radio("Go to Section:", pages, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ⚙️ Moderation Settings")
    feature_mode = st.sidebar.selectbox(
        "Feature Representation:",
        ["tfidf", "fasttext"],
        format_func=lambda x: "🔤 TF-IDF (N-Grams)" if x == "tfidf" else "🧠 FastText Subwords"
    )
    model_type = st.sidebar.selectbox(
        "Classifier Architecture:",
        ["Logistic Regression", "Random Forest"],
        format_func=lambda x: f"⚡ {x}"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌐 Supported Indian Languages")
    st.sidebar.markdown("""
    - 🇮🇳 **Hindi (हिन्दी)**
    - 🇮🇳 **Hinglish (Code-Mixed)**
    - 🇮🇳 **Telugu (తెలుగు)**
    - 🇮🇳 **Tamil (தமிழ்)**
    - 🇮🇳 **Malayalam (മലയാളം)**
    - 🇮🇳 **Kannada (ಕನ್ನಡ)**
    - 🇮🇳 **Indian English**
    """)
    return choice, feature_mode, model_type


# ==========================================
# TAB 1: Home & Overview
# ==========================================
def render_home_page():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🇮🇳 IndiTox 2.0</div>
        <div class="hero-sub">Multilingual Toxic Comment Detection for Indian Social Media & Code-Mixed Discourse</div>
        <div class="hero-tags">
            <span class="hero-tag">🌟 Multi-Script Unicode Support</span>
            <span class="hero-tag">🇮🇳 7 Indian Languages</span>
            <span class="hero-tag">🛡️ 6 Multi-label Categories</span>
            <span class="hero-tag">⚡ FastText + TF-IDF Hybrid</span>
            <span class="hero-tag">💡 Polite Rephrase Suggester</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 Project Architecture & Workflow")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">Step 1</div>
            <div class="step-title">📥 Multi-Script Ingestion</div>
            <div class="step-desc">Ingests comments in Devanagari, Dravidian scripts, Romanized Hinglish, and slang with elongation reduction.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">Step 2</div>
            <div class="step-title">🧼 Indic Preprocessing</div>
            <div class="step-desc">De-obfuscates masked profanities (<code>b@kwas</code>, <code>b*ch</code>), filters Indian stopwords, and detects source language.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">Step 3</div>
            <div class="step-title">🤖 Subword & ML Models</div>
            <div class="step-desc">FastText subwords and character n-gram TF-IDF feeding Logistic Regression (OvR) and Random Forest ensemble.</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">Step 4</div>
            <div class="step-title">🔍 Explain & Rephrase</div>
            <div class="step-desc">Token-level toxic span highlighting and constructive civil rewrite suggestions for healthy online discourse.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Multi-Label Toxicity Taxonomy")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.info("**1. Toxic (`toxic`)**\n\nGeneral hostility, rudeness, trolling, or aggressive behavior.")
        st.error("**2. Severe Toxic (`severe_toxic`)**\n\nExtremely vulgar, highly abusive profanities and offensive attacks.")
    with t2:
        st.warning("**3. Obscene (`obscene`)**\n\nVulgar sexual references, anatomical slurs, and explicit profanity.")
        st.error("**4. Threat (`threat`)**\n\nDirect statements of violence, murder, physical harm, or harassment.")
    with t3:
        st.warning("**5. Insult (`insult`)**\n\nDisparaging, derogatory personal name-calling and mocking.")
        st.error("**6. Identity Hate (`identity_hate`)**\n\nCommunal, religious, casteist, regional, or ethnic hate speech.")

    st.markdown("---")
    st.markdown("### 👤 Author & Project Overview")
    st.markdown("""
    <div style="background: rgba(125, 125, 125, 0.08); border: 1px solid rgba(125, 125, 125, 0.2); border-radius: 12px; padding: 20px;">
        <p style="margin: 6px 0; font-size: 1.05rem;"><strong>Project:</strong> Indian Multilingual Toxic Comment Detection System (IndiTox 2.0)</p>
        <p style="margin: 6px 0;"><strong>Author:</strong> Sanket Kurve (USN: CS23121)</p>
        <p style="margin: 6px 0;"><strong>Project Type:</strong> NLP TAE / Natural Language Processing</p>
        <p style="margin: 6px 0;"><strong>Repository:</strong> <a href="https://github.com/sanket200511/NLP-TAE" target="_blank" style="color: #818cf8; font-weight: 600;">github.com/sanket200511/NLP-TAE</a></p>
        <p style="margin: 6px 0; opacity: 0.9;"><strong>Core Implementation:</strong> Native multi-script Unicode tokenization, Hinglish elongation compressor, Subword FastText embeddings, Multilingual TF-IDF, and Hybrid Sensitivity ML inference.</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# TAB 2: Indian Multilingual EDA
# ==========================================
def render_eda_page():
    st.markdown("## 📊 Indian Multilingual Dataset & EDA")
    df = st.session_state.df_indian

    if df is None:
        st.warning("Dataset not loaded. Please ensure data/indian_toxic_comments.csv exists.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Dataset Overview",
        "🌐 Language & Script Analysis",
        "🏷️ Toxicity Labels & Correlations",
        "🔤 N-Gram & Word Analytics"
    ])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Comments", f"{len(df):,}")
        c2.metric("Languages Covered", f"{df['language'].nunique()}")
        clean_count = (df[LABEL_COLS].sum(axis=1) == 0).sum()
        toxic_count = (df[LABEL_COLS].sum(axis=1) > 0).sum()
        c3.metric("Clean Comments", f"{clean_count} ({clean_count/len(df)*100:.1f}%)")
        c4.metric("Toxic Comments", f"{toxic_count} ({toxic_count/len(df)*100:.1f}%)")

        st.markdown("#### Sample Dataset Entries")
        st.dataframe(
            df[["comment_text", "language"] + LABEL_COLS].head(15),
            use_container_width=True,
            height=300
        )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Distribution by Language")
            lang_counts = df["language"].value_counts().reset_index()
            lang_counts.columns = ["Language", "Count"]
            chart_lang = alt.Chart(lang_counts).mark_bar(cornerRadius=6).encode(
                x=alt.X("Count:Q", title="Number of Comments"),
                y=alt.Y("Language:N", sort="-x", title=""),
                color=alt.Color("Language:N", scale=alt.Scale(scheme="category10"), legend=None),
                tooltip=["Language", "Count"]
            ).properties(height=280)
            st.altair_chart(chart_lang, use_container_width=True)

        with c2:
            st.markdown("#### Clean vs Toxic Proportion")
            pie_data = pd.DataFrame({
                "Category": ["Clean (Non-Toxic)", "Toxic (Any Label)"],
                "Count": [clean_count, toxic_count]
            })
            chart_pie = alt.Chart(pie_data).mark_arc(outerRadius=110, innerRadius=50).encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color("Category:N", scale=alt.Scale(domain=["Clean (Non-Toxic)", "Toxic (Any Label)"], range=["#10b981", "#ef4444"])),
                tooltip=["Category", "Count"]
            ).properties(height=280)
            st.altair_chart(chart_pie, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Toxicity Label Distribution")
            label_sums = df[LABEL_COLS].sum().reset_index()
            label_sums.columns = ["Label", "Count"]
            chart_labels = alt.Chart(label_sums).mark_bar(cornerRadius=6).encode(
                x=alt.X("Count:Q", title="Occurrences"),
                y=alt.Y("Label:N", sort="-x", title=""),
                color=alt.Color("Label:N", scale=alt.Scale(scheme="reds")),
                tooltip=["Label", "Count"]
            ).properties(height=280)
            st.altair_chart(chart_labels, use_container_width=True)

        with c2:
            st.markdown("#### Label Correlation Heatmap")
            corr = df[LABEL_COLS].corr().stack().reset_index()
            corr.columns = ["Label A", "Label B", "Correlation"]
            chart_corr = alt.Chart(corr).mark_rect().encode(
                x=alt.X("Label A:O", title=""),
                y=alt.Y("Label B:O", title=""),
                color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="purples", domain=(-0.2, 1))),
                tooltip=["Label A", "Label B", alt.Tooltip("Correlation:Q", format=".2f")]
            ).properties(height=280)
            st.altair_chart(chart_corr, use_container_width=True)

    with tab4:
        st.markdown("#### Multilingual N-Gram Explorer")
        ngram_type = st.radio("Select N-Gram Order:", ["Unigrams (1)", "Bigrams (2)", "Trigrams (3)"], horizontal=True)
        n_val = 1 if "1" in ngram_type else (2 if "2" in ngram_type else 3)

        from sklearn.feature_extraction.text import CountVectorizer
        cleaned_series = df["comment_text"].apply(lambda x: clean_indic_text(x, remove_stopwords=True))
        vec = CountVectorizer(ngram_range=(n_val, n_val), min_df=2, max_df=0.95)
        try:
            mat = vec.fit_transform(cleaned_series)
            sum_words = mat.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:25]
            ngram_df = pd.DataFrame(words_freq, columns=["N-Gram Phrase", "Frequency"])

            chart_ngram = alt.Chart(ngram_df).mark_bar(cornerRadius=4, color="#6366f1").encode(
                x=alt.X("Frequency:Q", title="Frequency"),
                y=alt.Y("N-Gram Phrase:N", sort="-x", title=""),
                tooltip=["N-Gram Phrase", "Frequency"]
            ).properties(height=380)
            st.altair_chart(chart_ngram, use_container_width=True)
        except Exception as e:
            st.info("Not enough vocabulary for selected n-gram setting.")


# ==========================================
# TAB 3: Indic Preprocessing Studio
# ==========================================
def render_preprocessing_page():
    st.markdown("## 🧼 Indic Preprocessing Studio")
    st.markdown("Explore how raw, noisy Indian social media text (with elongations, masked slurs, and multi-script Unicode) is normalized.")

    raw_sample = st.text_area(
        "Input Raw Indian Comment:",
        value="Arre bhaaaai tu kitna b@kwaas video banata hai, bilkul pagal hai kya??? 😡😡",
        height=100
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1. Language & Script Detection")
        lang = detect_indic_language(raw_sample)
        st.markdown(f'<span class="badge-lang">🌐 Detected: <strong>{lang}</strong></span>', unsafe_allow_html=True)

        st.markdown("#### 2. De-obfuscation (Leetspeak / Masking)")
        deobf = deobfuscate_text(raw_sample)
        st.code(deobf)

    with c2:
        st.markdown("#### 3. Elongation Compression")
        elong = normalize_repeated_chars(deobf)
        st.code(elong)

        st.markdown("#### 4. Final Cleaned Output (Script-Preserving)")
        cleaned = clean_indic_text(raw_sample, remove_stopwords=False)
        st.success(cleaned)

    st.markdown("---")
    st.markdown("#### 🧪 Tokenization & Stopwords Filtering Comparison")
    tokens_raw = tokenize_indic(raw_sample)
    cleaned_no_stopwords = clean_indic_text(raw_sample, remove_stopwords=True)
    tokens_filtered = cleaned_no_stopwords.split()

    tc1, tc2 = st.columns(2)
    with tc1:
        st.write("**All Tokens:**", tokens_raw)
    with tc2:
        st.write("**After Indic & English Stopwords Filter:**", tokens_filtered)


# ==========================================
# TAB 4: Model Benchmarks & Training
# ==========================================
def render_modeling_page(feature_mode: str, model_type: str):
    st.markdown("## 🤖 Model Benchmarks & Training")
    engine = st.session_state.engine
    df = st.session_state.df_indian

    if df is None:
        st.warning("Dataset not available.")
        return

    # Prepare features
    texts = df["comment_text"].apply(clean_indic_text).tolist()
    y_true = df[LABEL_COLS].values

    # Load or train models
    with st.spinner("Checking / Training models..."):
        if feature_mode == "tfidf":
            vec = engine.load_tfidf()
            if vec is None:
                vec, X_feat = engine.train_tfidf(texts)
            else:
                X_feat = vec.transform(texts)
        else:
            ft = engine.load_fasttext()
            if ft is None:
                sentences = [tokenize_indic(t) for t in texts]
                ft = engine.train_fasttext(sentences)
            X_feat = np.vstack([engine.embed_comment_fasttext(t) for t in texts])

        lr, rf = engine.load_models(feature_mode)
        if lr is None or rf is None:
            lr, rf = engine.train_models(X_feat, y_true, feature_mode=feature_mode)

    # Active model predictions
    selected_model = lr if model_type == "Logistic Regression" else rf
    y_pred = selected_model.predict(X_feat)

    # Probabilities
    y_proba = None
    try:
        raw_prob = selected_model.predict_proba(X_feat)
        if isinstance(raw_prob, list):
            y_proba = np.vstack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in raw_prob]).T
        else:
            y_proba = raw_prob
    except Exception:
        y_proba = None

    metrics = compute_multilabel_metrics(y_true, y_pred)
    rep, confs, rocs = compute_detailed_evaluations(y_true, y_pred, y_proba)

    # Top metrics display
    st.markdown(f"### 📈 Performance Metrics: **{model_type}** ({'TF-IDF' if feature_mode == 'tfidf' else 'FastText'})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Macro F1", f"{metrics['Macro F1']:.3f}")
    m2.metric("Micro F1", f"{metrics['Micro F1']:.3f}")
    m3.metric("Weighted F1", f"{metrics['Weighted F1']:.3f}")
    m4.metric("Hamming Loss", f"{metrics['Hamming Loss']:.4f}")

    tab_rep, tab_cm, tab_roc = st.tabs(["📋 Classification Report", "🎯 Confusion Matrices", "📉 ROC Curves"])

    with tab_rep:
        st.markdown("#### Per-Label Performance")
        rep_df = pd.DataFrame(rep).transpose()
        st.dataframe(rep_df.style.format("{:.3f}"), use_container_width=True)

    with tab_cm:
        st.markdown("#### 2x2 Confusion Matrices (Per Toxicity Category)")
        c_cols = st.columns(3)
        for idx, label in enumerate(LABEL_COLS):
            with c_cols[idx % 3]:
                cm = confs[label]
                fig, ax = plt.subplots(figsize=(3.5, 2.8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], ax=ax)
                ax.set_title(f"Label: {label}", fontsize=11, fontweight="bold")
                st.pyplot(fig, use_container_width=True)

    with tab_roc:
        st.markdown("#### ROC-AUC Curves")
        if rocs:
            roc_rows = []
            for lbl, data in rocs.items():
                fpr = data["fpr"]
                tpr = data["tpr"]
                auc_val = data["auc"]
                for f, t in zip(fpr, tpr):
                    roc_rows.append({"Label": lbl, "FPR": f, "TPR": t, "AUC": auc_val})
            roc_df = pd.DataFrame(roc_rows)

            chart_roc = alt.Chart(roc_df).mark_line().encode(
                x=alt.X("FPR:Q", title="False Positive Rate", scale=alt.Scale(domain=(0, 1))),
                y=alt.Y("TPR:Q", title="True Positive Rate", scale=alt.Scale(domain=(0, 1))),
                color="Label:N",
                tooltip=["Label", alt.Tooltip("AUC:Q", format=".3f")]
            ).properties(height=320, title=f"ROC Curves ({model_type})")
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
            st.altair_chart(chart_roc + diag, use_container_width=True)
        else:
            st.info("ROC curves require probability outputs.")


# ==========================================
# TAB 5: Live Multilingual Comment Detector
# ==========================================
def render_prediction_page(feature_mode: str, model_type: str):
    st.markdown("## 🔍 Live Multilingual Comment Detector")
    st.markdown("Test Indian social media comments across Hindi, Hinglish, Tamil, Telugu, Malayalam, Kannada, and Indian English in real-time.")

    engine = st.session_state.engine

    # Preset examples
    presets = {
        "Select a quick preset sample...": "",
        "🇮🇳 [Hinglish Threat] Tere ghar aake tujhe jaan se maar dalunga kutte": "Tere ghar aake tujhe jaan se maar dalunga kutte, police bhi nahi bacha payegi.",
        "🇮🇳 [Hindi Devanagari Obscene] चुप कर गधे, तुझे कुछ नहीं पता, बकवास बंद कर अपनी": "चुप कर गधे, तुझे कुछ नहीं पता, बकवास बंद कर अपनी।",
        "🇮🇳 [Hinglish Vulgar] Chup bsdk, apni aukat me reh warna teri maa chod dunga": "Chup bsdk, apni aukat me reh warna teri maa chod dunga.",
        "🇮🇳 [Telugu Obscene] Dengey ra lanjakodaka, ne amma dengi pampistha kukka": "Dengey ra lanjakodaka, ne amma dengi pampistha kukka.",
        "🇮🇳 [Tamil Insult] Loose madhiri pesadha, unakku onnum theriyaadhu muttal": "Loose madhiri pesadha, unakku onnum theriyaadhu muttal.",
        "🇮🇳 [Malayalam Threat] Ninne njan kollum da, ninte veettil kayari vettum": "Ninne njan kollum da, ninte veettil kayari vettum.",
        "🇮🇳 [Kannada Insult] Ninage thale kettideya, yenu gothilla sumne bidthiya": "Ninage thale kettideya, yenu gothilla sumne bidthiya.",
        "🇮🇳 [Indian English Hate] Dalits and lower caste people are dirty and don't deserve reservations": "Dalits and lower caste people are dirty and don't deserve reservations.",
        "🇮🇳 [Hindi Clean] नमस्ते सर, आपका यह वीडियो बहुत ज्ञानवर्धक और उपयोगी था। धन्यवाद!": "नमस्ते सर, आपका यह वीडियो बहुत ज्ञानवर्धक और उपयोगी था। धन्यवाद!",
        "🇮🇳 [Hinglish Clean] Bhai aapka explanation bohot badhiya tha, sab samajh aa gaya!": "Bhai aapka explanation bohot badhiya tha, sab samajh aa gaya!",
        "🇮🇳 [Telugu Clean] Super anna! Video chala bagundi, inka regular ga videos cheyandi": "Super anna! Video chala bagundi, inka regular ga videos cheyandi.",
        "🇮🇳 [Indian English Clean] Wonderful analysis! Keep up the great work and upload more content": "Wonderful analysis! Keep up the great work and upload more content."
    }

    choice = st.selectbox("⚡ Quick-Test Presets:", list(presets.keys()))
    default_val = presets[choice] if choice != "Select a quick preset sample..." else ""

    user_text = st.text_area(
        "Enter comment to moderate:",
        value=default_val,
        height=120,
        placeholder="Type or paste any Indian social media comment in Hindi, Hinglish, Tamil, Telugu, etc..."
    )

    if st.button("🛡️ Analyze Toxicity & Moderate", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter a comment to analyze.")
            return

        with st.spinner("Analyzing with IndiTox Hybrid NLP Pipeline..."):
            res = engine.predict_hybrid(user_text, model_type=model_type, feature_mode=feature_mode)
            highlighted_html, entities = highlight_toxic_spans(user_text)
            suggestions = suggest_polite_alternatives(user_text)

        # Header Summary
        st.markdown("### 📊 Moderation Result")
        h1, h2, h3 = st.columns([1.5, 1.5, 3])
        with h1:
            if res["is_toxic"]:
                st.markdown('<div class="badge-toxic">⚠️ TOXIC DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-clean">✅ CLEAN / SAFE</div>', unsafe_allow_html=True)

        with h2:
            st.markdown(f'<div class="badge-lang">🌐 {res["detected_language"]}</div>', unsafe_allow_html=True)

        with h3:
            st.progress(min(1.0, res["overall_score"]))
            st.caption(f"Overall Toxicity Severity: **{res['overall_score']*100:.1f}%**")

        st.markdown("---")

        # Category Breakdown
        st.markdown("#### 🏷️ Multi-Label Category Breakdown")
        pill_cols = st.columns(6)
        for idx, col in enumerate(LABEL_COLS):
            with pill_cols[idx]:
                score = res["probabilities"].get(col, 0.0)
                is_active = res["binary_predictions"].get(col, 0) == 1
                color = "#ef4444" if is_active else "#10b981"
                st.markdown(f"""
                <div style="border: 1px solid {color}; border-radius: 10px; padding: 8px; text-align: center; background: {'rgba(239, 68, 68, 0.12)' if is_active else 'rgba(16, 185, 129, 0.12)'};">
                    <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">{col.replace('_', ' ')}</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: {color};">{score*100:.0f}%</div>
                    <div style="font-size: 0.75rem; color: {'#ef4444' if is_active else '#10b981'}; font-weight: 600;">{'FLAGGED' if is_active else 'CLEAR'}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Explainability Span Highlighting
        st.markdown("#### 🔎 Explainability: Detected Toxic Spans")
        st.markdown(f"""
        <div style="background: rgba(125, 125, 125, 0.08); border: 1px solid rgba(125, 125, 125, 0.2); border-radius: 10px; padding: 16px; font-size: 1.05rem; line-height: 1.8;">
            {highlighted_html}
        </div>
        """, unsafe_allow_html=True)

        if entities:
            st.caption(f"Detected **{len(entities)}** toxic phrase trigger(s): " + ", ".join([f"`{e['term']}` ({e['category']})" for e in entities]))

        # Polite Rephrase Suggester
        if res["is_toxic"]:
            st.markdown("---")
            st.markdown("#### 💡 Polite & Civil Alternative Suggestions")
            st.markdown("Help foster healthier discourse on Indian platforms by suggesting respectful phrasing:")
            for s in suggestions:
                st.info(f"**Instead of aggressive tone:**\n\n👉 *\"{s['polite_suggestion']}\"*")


# ==========================================
# TAB 6: Batch CSV Moderation
# ==========================================
def render_batch_page(feature_mode: str, model_type: str):
    st.markdown("## 📁 Batch CSV Comment Moderation")
    st.markdown("Upload any CSV file containing a column of comments to run bulk moderation and download an annotated report.")

    engine = st.session_state.engine

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    else:
        st.info("No file uploaded. Showing demo using built-in benchmark samples.")
        batch_df = st.session_state.df_indian.sample(min(20, len(st.session_state.df_indian)), random_state=42).copy()

    # Column selection
    text_cols = [c for c in batch_df.columns if batch_df[c].dtype == "object"]
    target_col = st.selectbox("Select Comment Column:", text_cols, index=0)

    if st.button("🚀 Run Batch Moderation", type="primary"):
        with st.spinner("Processing comments in batch..."):
            results = []
            for text in batch_df[target_col].astype(str):
                r = engine.predict_hybrid(text, model_type=model_type, feature_mode=feature_mode)
                row_dict = {
                    "detected_language": r["detected_language"],
                    "is_toxic": int(r["is_toxic"]),
                    "severity_score": r["overall_score"],
                }
                for c in LABEL_COLS:
                    row_dict[f"pred_{c}"] = r["binary_predictions"].get(c, 0)
                    row_dict[f"prob_{c}"] = r["probabilities"].get(c, 0.0)
                results.append(row_dict)

            res_df = pd.DataFrame(results)
            out_df = pd.concat([batch_df.reset_index(drop=True), res_df], axis=1)

        st.success(f"Successfully processed {len(out_df)} comments!")

        # Batch Summary Metrics
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Rows", f"{len(out_df)}")
        toxic_num = out_df["is_toxic"].sum()
        b2.metric("Flagged Toxic", f"{toxic_num} ({toxic_num/len(out_df)*100:.1f}%)")
        b3.metric("Clean Rows", f"{len(out_df)-toxic_num}")
        b4.metric("Avg Severity", f"{out_df['severity_score'].mean()*100:.1f}%")

        st.dataframe(out_df.head(25), use_container_width=True, height=350)

        # Download CSV
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Moderation Report (CSV)",
            data=csv_bytes,
            file_name="inditox_moderation_report.csv",
            mime="text/csv"
        )


# ==========================================
# Main Router
# ==========================================
def main():
    choice, feature_mode, model_type = render_sidebar()

    if choice == "🏠 Home & Overview":
        render_home_page()
    elif choice == "📊 Indian Multilingual EDA":
        render_eda_page()
    elif choice == "🧼 Indic Preprocessing Studio":
        render_preprocessing_page()
    elif choice == "🤖 Model Benchmarks & Training":
        render_modeling_page(feature_mode, model_type)
    elif choice == "🔍 Live Comment Detector":
        render_prediction_page(feature_mode, model_type)
    elif choice == "📁 Batch CSV Moderation":
        render_batch_page(feature_mode, model_type)

    # Footer
    st.markdown("""
    <div class="inditox-footer">
        🇮🇳 <strong>IndiTox 2.0</strong> • Indian Multilingual Toxic Comment Detector • Developed by <strong>Sanket Kurve (CS23121)</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

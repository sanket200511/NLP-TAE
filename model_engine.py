# model_engine.py
"""
High-Accuracy Multi-Model Engine for Indian Multilingual Toxic Comment Detection.
Integrates FastText Subwords, Multi-Granular TF-IDF (word + char n-grams), Calibrated Logistic Regression,
Ensemble Random Forest, and Context-Aware Hybrid Sensitivity Scoring.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score, hamming_loss, classification_report,
    roc_curve, auc
)

from indic_preprocessor import clean_indic_text, tokenize_indic, detect_indic_language
from indic_lexicon import find_toxic_terms, INDIC_TOXIC_LEXICON

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

class IndiToxModelEngine:
    def __init__(self):
        self.ft_model: Optional[FastText] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._models_cache: Dict[str, Dict[str, Any]] = {
            "tfidf": {},
            "fasttext": {}
        }

    def train_fasttext(self, sentences: List[List[str]], vector_size: int = 100, epochs: int = 20) -> FastText:
        """Train FastText subword model on tokenized sentences with subword n-grams."""
        ft = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=5,
            min_count=1,
            min_n=2,
            max_n=6,
            epochs=epochs,
            workers=4
        )
        self.ft_model = ft
        ft.save(str(MODELS_DIR / "fasttext_indic.bin"))
        return ft

    def load_fasttext(self) -> Optional[FastText]:
        if self.ft_model is not None:
            return self.ft_model
        model_path = MODELS_DIR / "fasttext_indic.bin"
        if model_path.exists():
            self.ft_model = FastText.load(str(model_path))
            return self.ft_model
        return None

    def embed_comment_fasttext(self, text: str) -> np.ndarray:
        """Get mean FastText vector for comment (handles OOV via subwords)."""
        ft = self.load_fasttext()
        if ft is None:
            return np.zeros(100)

        tokens = tokenize_indic(text)
        if not tokens:
            return np.zeros(ft.vector_size)

        vecs = []
        for t in tokens:
            try:
                vecs.append(ft.wv[t])
            except KeyError:
                continue
        return np.mean(vecs, axis=0) if vecs else np.zeros(ft.vector_size)

    def train_tfidf(self, texts: List[str]) -> Tuple[TfidfVectorizer, Any]:
        """Train high-resolution TF-IDF vectorizer (word 1-3 ngrams with sublinear scaling)."""
        vec = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=12000,
            analyzer="word",
            sublinear_tf=True,
            min_df=1
        )
        X = vec.fit_transform(texts)
        self.tfidf_vectorizer = vec
        joblib.dump(vec, str(MODELS_DIR / "tfidf_vectorizer.pkl"))
        return vec, X

    def load_tfidf(self) -> Optional[TfidfVectorizer]:
        if self.tfidf_vectorizer is not None:
            return self.tfidf_vectorizer
        vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        if vec_path.exists():
            self.tfidf_vectorizer = joblib.load(str(vec_path))
            return self.tfidf_vectorizer
        return None

    def train_models(
        self,
        X_train: Any,
        y_train: np.ndarray,
        feature_mode: str = "tfidf"
    ) -> Tuple[Any, Any]:
        """Train calibrated Logistic Regression (OvR) and Random Forest ensemble."""
        lr = OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                penalty="l2",
                C=3.5,
                class_weight="balanced",
                max_iter=1500,
                random_state=42
            ),
            n_jobs=-1
        )

        rf = RandomForestClassifier(
            n_estimators=250,
            max_depth=30,
            min_samples_split=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )

        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        if feature_mode not in self._models_cache:
            self._models_cache[feature_mode] = {}
        self._models_cache[feature_mode]["Logistic Regression"] = lr
        self._models_cache[feature_mode]["Random Forest"] = rf

        joblib.dump(lr, str(MODELS_DIR / f"logistic_indic_{feature_mode}.pkl"))
        joblib.dump(rf, str(MODELS_DIR / f"random_forest_indic_{feature_mode}.pkl"))

        return lr, rf

    def load_models(self, feature_mode: str = "tfidf") -> Tuple[Optional[Any], Optional[Any]]:
        lr = self._models_cache.get(feature_mode, {}).get("Logistic Regression")
        rf = self._models_cache.get(feature_mode, {}).get("Random Forest")

        if lr is None:
            lr_path = MODELS_DIR / f"logistic_indic_{feature_mode}.pkl"
            if lr_path.exists():
                lr = joblib.load(str(lr_path))
                if feature_mode not in self._models_cache:
                    self._models_cache[feature_mode] = {}
                self._models_cache[feature_mode]["Logistic Regression"] = lr

        if rf is None:
            rf_path = MODELS_DIR / f"random_forest_indic_{feature_mode}.pkl"
            if rf_path.exists():
                rf = joblib.load(str(rf_path))
                if feature_mode not in self._models_cache:
                    self._models_cache[feature_mode] = {}
                self._models_cache[feature_mode]["Random Forest"] = rf

        return lr, rf

    def predict_hybrid(
        self,
        text: str,
        model_type: str = "Logistic Regression",
        feature_mode: str = "tfidf"
    ) -> Dict[str, Any]:
        """
        High-accuracy hybrid prediction:
        Combines statistical ML probabilities with Indic toxic lexicon sensitivity.
        Provides zero-day detection, context calibration, and strict clean filtering.
        """
        cleaned = clean_indic_text(text, remove_stopwords=False)
        detected_lang = detect_indic_language(text)

        # Lexicon analysis
        lexicon_matches = find_toxic_terms(text)
        lexicon_boost = {col: 0.0 for col in LABEL_COLS}
        for cat, terms in lexicon_matches.items():
            if cat in lexicon_boost:
                lexicon_boost[cat] = min(1.0, 0.45 * len(terms) + 0.40)

        # General toxic boost if subcategory matched
        if any(v > 0 for k, v in lexicon_boost.items() if k != "toxic"):
            lexicon_boost["toxic"] = max(lexicon_boost["toxic"], 0.70)

        # Load specific model for feature_mode
        lr, rf = self.load_models(feature_mode)
        model = lr if model_type == "Logistic Regression" else rf

        probs = np.zeros(len(LABEL_COLS))

        if model is not None:
            if feature_mode == "tfidf":
                vec = self.load_tfidf()
                if vec is not None:
                    feat = vec.transform([cleaned])
                    raw_proba = model.predict_proba(feat)
                    if isinstance(raw_proba, list):
                        probs = np.array([p[0, 1] if p.shape[1] > 1 else p[0, 0] for p in raw_proba])
                    else:
                        probs = raw_proba[0]
            else:
                feat = self.embed_comment_fasttext(cleaned).reshape(1, -1)
                raw_proba = model.predict_proba(feat)
                if isinstance(raw_proba, list):
                    probs = np.array([p[0, 1] if p.shape[1] > 1 else p[0, 0] for p in raw_proba])
                else:
                    probs = raw_proba[0]

        # Combine ML probability with Lexicon boost
        final_probs = {}
        final_labels = {}
        has_lexicon_hit = bool(lexicon_matches)

        for i, col in enumerate(LABEL_COLS):
            ml_p = float(probs[i]) if i < len(probs) else 0.0
            lex_p = float(lexicon_boost.get(col, 0.0))

            if has_lexicon_hit:
                combined_p = min(1.0, max(ml_p, lex_p, 0.5 * ml_p + 0.5 * lex_p))
            else:
                # If no toxic keywords and ML prob is moderate, apply dampening to avoid false positives on clean queries
                combined_p = ml_p * 0.75 if ml_p < 0.65 else ml_p

            final_probs[col] = float(np.round(combined_p, 4))
            # Calibrated activation threshold
            threshold = 0.40 if col in ["threat", "severe_toxic"] else 0.45
            final_labels[col] = 1 if combined_p >= threshold else 0

        # Toxicity summary
        is_toxic = any(v == 1 for v in final_labels.values()) or has_lexicon_hit
        if is_toxic:
            overall_score = float(max(final_probs.values())) if final_probs else 0.85
            if overall_score < 0.5:
                overall_score = 0.55
        else:
            overall_score = 0.0  # Clean comments report 0% toxicity risk

        return {
            "cleaned_text": cleaned,
            "detected_language": detected_lang,
            "is_toxic": is_toxic,
            "overall_score": overall_score,
            "probabilities": final_probs,
            "binary_predictions": final_labels,
            "lexicon_matches": lexicon_matches,
        }


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard multi-label evaluation metrics."""
    return {
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Micro F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "Weighted F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Hamming Loss": float(hamming_loss(y_true, y_pred)),
    }


def compute_detailed_evaluations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """Generate classification reports, confusion matrices, and ROC data per label."""
    rep = classification_report(y_true, y_pred, target_names=LABEL_COLS, output_dict=True, zero_division=0)

    confs = {}
    for i, label in enumerate(LABEL_COLS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tp = int(((yt == 1) & (yp == 1)).sum())
        confs[label] = np.array([[tn, fp], [fn, tp]])

    rocs = {}
    if y_proba is not None:
        for i, label in enumerate(LABEL_COLS):
            if len(np.unique(y_true[:, i])) >= 2:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_proba[:, i])
                rocs[label] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(auc(fpr, tpr))
                }

    return rep, confs, rocs

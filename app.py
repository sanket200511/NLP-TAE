# app.py
"""
Flask Web Server Backend for IndiTox 2.0.
Exposes real-time classification and batch file moderation endpoints.
"""

import os
import io
import csv
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

# Import engine modules
from model_engine import IndiToxModelEngine, LABEL_COLS
from indic_preprocessor import clean_indic_text
from indic_lexicon import highlight_toxic_spans, suggest_polite_alternatives

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static"
)

# Initialize engine globally
engine = IndiToxModelEngine()

# Ensure model parameters are loaded at startup
try:
    engine.load_tfidf()
    engine.load_fasttext()
    engine.load_models("tfidf")
    engine.load_models("fasttext")
    print("IndiTox NLP models loaded successfully.")
except Exception as e:
    print(f"Warning: Models could not be auto-loaded: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json() or {}
    comment = data.get("comment", "").strip()
    model_type = data.get("model_type", "Logistic Regression")
    feature_mode = data.get("feature_mode", "tfidf")

    if not comment:
        return jsonify({"error": "No comment provided"}), 400

    # Get predictions
    res = engine.predict_hybrid(comment, model_type=model_type, feature_mode=feature_mode)

    # Highlight toxic spans
    highlighted_html, entities = highlight_toxic_spans(comment)

    # Get polite alternatives
    suggestions = suggest_polite_alternatives(comment)

    return jsonify({
        "is_toxic": res["is_toxic"],
        "overall_score": res["overall_score"],
        "probabilities": res["probabilities"],
        "binary_predictions": res["binary_predictions"],
        "detected_language": res["detected_language"],
        "highlighted_html": highlighted_html,
        "suggestions": suggestions
    })


@app.route("/api/moderate_file", methods=["POST"])
def moderate_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    comment_col = request.form.get("comment_column", "").strip()
    model_type = request.form.get("model_type", "Logistic Regression")
    feature_mode = request.form.get("feature_mode", "tfidf")

    if not comment_col:
        return jsonify({"error": "No comment column selected"}), 400

    try:
        # Read uploaded CSV
        file_stream = io.StringIO(uploaded_file.stream.read().decode("utf-8-sig"), newline="")
        reader = csv.DictReader(file_stream)
        
        if reader.fieldnames is None:
            return jsonify({"error": "Failed to parse CSV headers"}), 400

        if comment_col not in reader.fieldnames:
            return jsonify({"error": f"Column '{comment_col}' not found in CSV headers"}), 400

        output_rows = []
        preview_rows = []

        fieldnames = list(reader.fieldnames) + [
            "detected_language", "is_toxic", "severity_score"
        ] + [f"pred_{c}" for c in LABEL_COLS] + [f"prob_{c}" for c in LABEL_COLS]

        # Write to in-memory output stream
        out_stream = io.StringIO()
        writer = csv.DictWriter(out_stream, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader):
            text = row.get(comment_col, "").strip()
            # Predict
            r = engine.predict_hybrid(text, model_type=model_type, feature_mode=feature_mode)
            
            row["detected_language"] = r["detected_language"]
            row["is_toxic"] = int(r["is_toxic"])
            row["severity_score"] = r["overall_score"]

            for c in LABEL_COLS:
                row[f"pred_{c}"] = r["binary_predictions"].get(c, 0)
                row[f"prob_{c}"] = r["probabilities"].get(c, 0.0)

            writer.writerow(row)

            # Store in list for preview (max 15 rows)
            preview_row = {
                "comment_text": text,
                "detected_language": r["detected_language"],
                "is_toxic": int(r["is_toxic"]),
                "severity_score": r["overall_score"]
            }
            for c in LABEL_COLS:
                preview_row[f"pred_{c}"] = r["binary_predictions"].get(c, 0)

            preview_rows.append(preview_row)

        csv_content = out_stream.getvalue()

        return jsonify({
            "rows": preview_rows,
            "csv_content": csv_content
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process CSV file: {str(e)}"}), 500


if __name__ == "__main__":
    # Launch local server
    app.run(host="localhost", port=8080, debug=True)

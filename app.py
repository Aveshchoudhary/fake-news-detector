"""Flask app exposing the fake-news classifier."""
import os
import re
import joblib
from flask import Flask, render_template, request, jsonify

MODEL_PATH = os.path.join(os.path.dirname(__file__),"model.joblib")

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model not found. Run `python train.py` first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def detect_language(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "hindi"
    return "english"


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or request.form
    text = (payload.get("text") or "").strip()
    if len(text) < 10:
        return jsonify({"error": "Please provide at least 10 characters."}), 400

    model = get_model()
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    idx = int(proba.argmax())
    label = classes[idx]
    confidence = round(float(proba[idx]) * 100, 2)

    return jsonify({
        "verdict": label,                 # "real" or "fake"
        "confidence": confidence,         # 0-100
        "language": detect_language(text),
        "probabilities": {c: round(float(p) * 100, 2) for c, p in zip(classes, proba)},
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

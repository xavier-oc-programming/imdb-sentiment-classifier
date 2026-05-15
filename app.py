import os
import re
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_DIR = Path('models')



# ── Text cleaning (mirrors train.py) ─────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Lazy model loading ────────────────────────────────────────────────────────
_models = {}

def load_models():
    global _models
    if _models:
        return _models

    best_model_path = MODEL_DIR / 'best_model.txt'
    if not best_model_path.exists():
        raise FileNotFoundError("models/best_model.txt not found — run train.py first")

    best = best_model_path.read_text().strip()
    _models['best'] = best

    lr_path  = MODEL_DIR / 'logistic_regression.pkl'
    vec_path = MODEL_DIR / 'tfidf_vectorizer.pkl'
    if lr_path.exists() and vec_path.exists():
        with open(vec_path, 'rb') as f:
            _models['tfidf_vectorizer'] = pickle.load(f)
        with open(lr_path, 'rb') as f:
            _models['logistic_regression'] = pickle.load(f)

    lstm_path  = MODEL_DIR / 'lstm_model.keras'
    vocab_path = MODEL_DIR / 'lstm_vocab.pkl'
    if lstm_path.exists() and vocab_path.exists():
        try:
            import tensorflow as tf
            from tensorflow.keras.layers import TextVectorization

            _models['lstm'] = tf.keras.models.load_model(str(lstm_path))

            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)

            vectorize_layer = TextVectorization(
                max_tokens=len(vocab),
                output_mode='int',
                output_sequence_length=256
            )
            vectorize_layer.set_vocabulary(vocab)
            _models['lstm_vectorizer'] = vectorize_layer
        except ImportError:
            pass  # TensorFlow not available in this environment (e.g. Lambda)

    return _models


# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_lr(text: str, models: dict) -> dict:
    cleaned    = clean_text(text)
    X          = models['tfidf_vectorizer'].transform([cleaned])
    prob       = models['logistic_regression'].predict_proba(X)[0]
    label      = int(models['logistic_regression'].predict(X)[0])
    confidence = float(prob[label])
    return {
        'sentiment':  'positive' if label == 1 else 'negative',
        'confidence': round(confidence, 4)
    }


def predict_lstm(text: str, models: dict) -> dict:
    cleaned    = clean_text(text)
    vec        = models['lstm_vectorizer']
    X          = vec(np.array([cleaned])).numpy()
    prob       = float(models['lstm'].predict(X, verbose=0)[0][0])
    label      = 1 if prob >= 0.5 else 0
    confidence = prob if label == 1 else 1 - prob
    return {
        'sentiment':  'positive' if label == 1 else 'negative',
        'confidence': round(confidence, 4)
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return {'status': 'ok'}, 200


@app.route('/')
def index():
    try:
        models = load_models()
        best   = models.get('best', 'unknown')
    except FileNotFoundError:
        best = 'not trained'
    return render_template('index.html', best_model=best)


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Request body must be JSON with a "text" field'}), 400

    text = data['text']
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': '"text" must be a non-empty string'}), 400
    if len(text) > 5000:
        return jsonify({'error': '"text" must be 5,000 characters or fewer'}), 400

    try:
        models = load_models()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503

    best = models['best']

    if best == 'logistic_regression':
        if 'logistic_regression' not in models:
            return jsonify({'error': 'Logistic regression model files not found'}), 503
        result      = predict_lr(text, models)
        model_label = 'TF-IDF + Logistic Regression'
    else:
        if 'lstm' not in models:
            return jsonify({'error': 'LSTM model files not found'}), 503
        result      = predict_lstm(text, models)
        model_label = 'Keras LSTM'

    return jsonify({
        'text':       text,
        'sentiment':  result['sentiment'],
        'confidence': result['confidence'],
        'model_used': model_label
    })


@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Request body must be JSON with a "text" field'}), 400

    text = data['text']
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': '"text" must be a non-empty string'}), 400
    if len(text) > 5000:
        return jsonify({'error': '"text" must be 5,000 characters or fewer'}), 400

    try:
        models = load_models()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503

    results = {}

    if 'logistic_regression' in models:
        results['logistic_regression'] = predict_lr(text, models)
    else:
        results['logistic_regression'] = {'error': 'Model not available'}

    if 'lstm' in models:
        results['lstm'] = predict_lstm(text, models)
    else:
        results['lstm'] = {'error': 'Model not available'}

    return jsonify({'text': text, 'results': results})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)

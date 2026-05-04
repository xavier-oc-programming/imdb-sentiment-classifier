import os
import re
import json
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_DIR = Path('models')

BEDROCK_MODEL_ID = 'anthropic.claude-haiku-4-5'
BEDROCK_REGION   = 'eu-west-1'


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


def predict_bedrock(text: str) -> dict:
    """Call Claude on Amazon Bedrock. Returns an error dict if credentials are not configured."""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError

        client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)

        prompt = (
            "Classify the sentiment of this movie review as exactly \"positive\" or \"negative\". "
            "Also provide a confidence score between 0.0 and 1.0 reflecting how certain you are. "
            "Respond with only valid JSON in this exact format, no other text: "
            "{\"sentiment\": \"positive\", \"confidence\": 0.95}\n\n"
            f"Review: {text}"
        )

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}]
        })

        response = client.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
        response_body = json.loads(response['body'].read())
        raw = response_body['content'][0]['text'].strip()
        result = json.loads(raw)

        return {
            'sentiment':  result['sentiment'],
            'confidence': round(float(result['confidence']), 4)
        }

    except ImportError:
        return {'error': 'boto3 not installed — pip install boto3'}
    except Exception as e:
        err = str(e)
        if 'NoCredentialsError' in err or 'credentials' in err.lower():
            return {'error': 'Bedrock not configured — set AWS credentials to enable'}
        if 'Could not connect' in err or 'EndpointResolutionError' in err:
            return {'error': 'Bedrock not configured — set AWS credentials to enable'}
        return {'error': f'Bedrock unavailable: {err}'}


# ── Routes ────────────────────────────────────────────────────────────────────
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


@app.route('/analyze-bedrock', methods=['POST'])
def analyze_bedrock():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Request body must be JSON with a "text" field'}), 400

    text = data['text']
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': '"text" must be a non-empty string'}), 400
    if len(text) > 5000:
        return jsonify({'error': '"text" must be 5,000 characters or fewer'}), 400

    result = predict_bedrock(text)

    if 'error' in result:
        return jsonify({'error': result['error']}), 503

    return jsonify({
        'text':       text,
        'sentiment':  result['sentiment'],
        'confidence': result['confidence'],
        'model_used': f'Claude Haiku (Amazon Bedrock)'
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

    results['bedrock'] = predict_bedrock(text)

    return jsonify({'text': text, 'results': results})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

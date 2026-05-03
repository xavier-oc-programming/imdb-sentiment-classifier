import os
import re
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from tensorflow.keras.layers import (
    TextVectorization, Embedding, LSTM, Dense
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── Constants ────────────────────────────────────────────────────────────────
MAX_FEATURES   = 10000
MAX_LEN        = 256
EMBEDDING_DIM  = 64
LSTM_UNITS     = 64
BATCH_SIZE     = 32
EPOCHS         = 5
MODEL_DIR      = Path('models')
PLOTS_DIR      = Path('plots')

MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ── Text cleaning ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)       # strip HTML tags
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # keep alphanumeric only
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data():
    print("Loading IMDb dataset via tensorflow_datasets …")
    (train_ds, test_ds), _ = tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )

    def decode_and_clean(ds):
        out_texts, out_labels = [], []
        for text, label in ds.as_numpy_iterator():
            out_texts.append(clean_text(text.decode('utf-8')))
            out_labels.append(int(label))
        return out_texts, out_labels

    print("Processing training set …")
    train_texts, train_labels = decode_and_clean(train_ds)

    print("Processing test set …")
    test_texts, test_labels = decode_and_clean(test_ds)

    print(f"  Train: {len(train_texts):,} reviews | Test: {len(test_texts):,} reviews")
    return train_texts, train_labels, test_texts, test_labels


# ── Model 1 — TF-IDF + Logistic Regression ───────────────────────────────────
def train_logistic_regression(train_texts, train_labels, test_texts, test_labels):
    print("\n" + "="*60)
    print("MODEL 1 — TF-IDF + Logistic Regression")
    print("="*60)

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))

    print("Fitting TF-IDF vectorizer …")
    t0 = time.time()
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    print("Training logistic regression …")
    clf.fit(X_train, train_labels)
    training_time = time.time() - t0

    y_pred = clf.predict(X_test)

    metrics = {
        'accuracy':      accuracy_score(test_labels, y_pred),
        'precision':     precision_score(test_labels, y_pred),
        'recall':        recall_score(test_labels, y_pred),
        'f1':            f1_score(test_labels, y_pred),
        'training_time': training_time
    }

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Time:      {training_time:.1f}s")

    # Save artefacts
    with open(MODEL_DIR / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_DIR / 'logistic_regression.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("  Saved → models/tfidf_vectorizer.pkl")
    print("  Saved → models/logistic_regression.pkl")

    return clf, vectorizer, metrics, y_pred


# ── Model 2 — Keras LSTM ─────────────────────────────────────────────────────
def train_lstm(train_texts, train_labels, test_texts, test_labels):
    print("\n" + "="*60)
    print("MODEL 2 — Keras LSTM")
    print("="*60)

    # Build TextVectorization layer and adapt on training data
    vectorize_layer = TextVectorization(
        max_tokens=MAX_FEATURES,
        output_mode='int',
        output_sequence_length=MAX_LEN
    )
    vectorize_layer.adapt(train_texts)

    # Vectorise texts to integer sequences
    X_train = vectorize_layer(np.array(train_texts)).numpy()
    X_test  = vectorize_layer(np.array(test_texts)).numpy()
    y_train = np.array(train_labels)
    y_test  = np.array(test_labels)

    model = Sequential([
        Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(LSTM_UNITS, dropout=0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    early_stop = EarlyStopping(patience=2, restore_best_weights=True)

    print("Training LSTM …")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    training_time = time.time() - t0

    # Evaluate
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'accuracy':      accuracy_score(y_test, y_pred),
        'precision':     precision_score(y_test, y_pred),
        'recall':        recall_score(y_test, y_pred),
        'f1':            f1_score(y_test, y_pred),
        'training_time': training_time
    }

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Time:      {training_time:.1f}s")

    model.save(MODEL_DIR / 'lstm_model.keras')
    print("  Saved → models/lstm_model.keras")

    # Also persist the vectorize_layer vocabulary so app.py can rebuild it
    vocab = vectorize_layer.get_vocabulary()
    with open(MODEL_DIR / 'lstm_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("  Saved → models/lstm_vocab.pkl")

    return model, vectorize_layer, metrics, y_pred, history


# ── Charts ───────────────────────────────────────────────────────────────────
def plot_model_comparison(lr_metrics, lstm_metrics):
    """Grouped bar chart — both models across all four metrics."""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    lr_vals   = [lr_metrics['accuracy'],   lr_metrics['precision'],
                 lr_metrics['recall'],     lr_metrics['f1']]
    lstm_vals = [lstm_metrics['accuracy'], lstm_metrics['precision'],
                 lstm_metrics['recall'],   lstm_metrics['f1']]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, lr_vals,   width, label='TF-IDF + LR',  color='#2E75B6')
    bars2 = ax.bar(x + width/2, lstm_vals, width, label='Keras LSTM',   color='#ED7D31')

    ax.set_ylim(0.80, 1.00)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — TF-IDF + LR vs Keras LSTM')
    ax.legend()
    ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=8)
    ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / '01_model_comparison.png', dpi=150)
    plt.close(fig)
    print("  Saved → plots/01_model_comparison.png")


def plot_lstm_training_curves(history):
    """Loss and accuracy per epoch for the LSTM."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.history['loss']) + 1)

    ax1.plot(epochs, history.history['loss'],          label='Train loss',    color='#2E75B6')
    ax1.plot(epochs, history.history['val_loss'],      label='Val loss',      color='#ED7D31', linestyle='--')
    ax1.set_title('LSTM — Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(linestyle='--', alpha=0.4)

    ax2.plot(epochs, history.history['accuracy'],      label='Train accuracy', color='#2E75B6')
    ax2.plot(epochs, history.history['val_accuracy'],  label='Val accuracy',   color='#ED7D31', linestyle='--')
    ax2.set_title('LSTM — Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / '02_lstm_training_curves.png', dpi=150)
    plt.close(fig)
    print("  Saved → plots/02_lstm_training_curves.png")


def plot_confusion_matrices(test_labels, lr_pred, lstm_pred):
    """Side-by-side confusion matrices for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, preds, title in [
        (ax1, lr_pred,   'TF-IDF + Logistic Regression'),
        (ax2, lstm_pred, 'Keras LSTM')
    ]:
        cm = confusion_matrix(test_labels, preds)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        ax.set_title(f'Confusion Matrix — {title}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / '03_confusion_matrices.png', dpi=150)
    plt.close(fig)
    print("  Saved → plots/03_confusion_matrices.png")


def plot_top_words(clf, vectorizer):
    """Top 20 most predictive words for each sentiment class (LR coefficients)."""
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]

    top_pos_idx = np.argsort(coefs)[-20:][::-1]
    top_neg_idx = np.argsort(coefs)[:20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Positive words
    ax1.barh(
        [feature_names[i] for i in top_pos_idx[::-1]],
        [coefs[i] for i in top_pos_idx[::-1]],
        color='#70AD47'
    )
    ax1.set_title('Top 20 Positive Sentiment Words')
    ax1.set_xlabel('LR Coefficient')
    ax1.grid(axis='x', linestyle='--', alpha=0.4)

    # Negative words
    ax2.barh(
        [feature_names[i] for i in top_neg_idx],
        [coefs[i] for i in top_neg_idx],
        color='#FF0000'
    )
    ax2.set_title('Top 20 Negative Sentiment Words')
    ax2.set_xlabel('LR Coefficient')
    ax2.grid(axis='x', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / '04_top_words.png', dpi=150)
    plt.close(fig)
    print("  Saved → plots/04_top_words.png")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    train_texts, train_labels, test_texts, test_labels = load_data()

    clf, vectorizer, lr_metrics, lr_pred = train_logistic_regression(
        train_texts, train_labels, test_texts, test_labels
    )

    model, vectorize_layer, lstm_metrics, lstm_pred, history = train_lstm(
        train_texts, train_labels, test_texts, test_labels
    )

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    header = f"{'Model':<35} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Time':>8}"
    print(header)
    print("-" * 70)

    def fmt_row(name, m):
        return (
            f"{name:<35} "
            f"{m['accuracy']:>9.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{m['training_time']:>6.1f}s"
        )

    print(fmt_row('TF-IDF + Logistic Regression', lr_metrics))
    print(fmt_row('Keras LSTM',                   lstm_metrics))
    print("="*70)

    # ── Select winner ─────────────────────────────────────────────────────────
    if lr_metrics['f1'] >= lstm_metrics['f1']:
        winner = 'logistic_regression'
        winner_label = 'TF-IDF + Logistic Regression'
        winner_f1    = lr_metrics['f1']
    else:
        winner = 'lstm'
        winner_label = 'Keras LSTM'
        winner_f1    = lstm_metrics['f1']

    (MODEL_DIR / 'best_model.txt').write_text(winner)
    print(f"\nWinner: {winner_label}  (F1 = {winner_f1:.4f})")
    print(f"Saved  → models/best_model.txt")

    # ── Generate charts ───────────────────────────────────────────────────────
    print("\nGenerating charts …")
    plot_model_comparison(lr_metrics, lstm_metrics)
    plot_lstm_training_curves(history)
    plot_confusion_matrices(test_labels, lr_pred, lstm_pred)
    plot_top_words(clf, vectorizer)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"TF-IDF + LR  →  Accuracy {lr_metrics['accuracy']:.4f} | F1 {lr_metrics['f1']:.4f} | {lr_metrics['training_time']:.1f}s")
    print(f"Keras LSTM   →  Accuracy {lstm_metrics['accuracy']:.4f} | F1 {lstm_metrics['f1']:.4f} | {lstm_metrics['training_time']:.1f}s")
    print(f"\nBest model : {winner_label}")
    print(f"Reason     : Highest F1 score ({winner_f1:.4f})")
    print(f"\nModel files:")
    print(f"  models/tfidf_vectorizer.pkl")
    print(f"  models/logistic_regression.pkl")
    print(f"  models/lstm_model.keras")
    print(f"  models/lstm_vocab.pkl")
    print(f"  models/best_model.txt → {winner}")
    print("="*70)


if __name__ == '__main__':
    main()

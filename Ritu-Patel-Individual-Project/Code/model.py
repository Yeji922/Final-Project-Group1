# ------------------------------------------------------------
# model.py : Model utilities for Suicide & Depression Detection
# ------------------------------------------------------------
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------
# TF-IDF Vectorization
# ------------------------------------------------------------
def vectorize_text(train_texts, val_texts=None, test_texts=None, max_features=20000):
    """Fit TF-IDF on training text and transform all sets."""
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts) if val_texts is not None else None
    X_test = tfidf.transform(test_texts) if test_texts is not None else None
    return tfidf, X_train, X_val, X_test

# ------------------------------------------------------------
# Train Logistic Regression Model
# ------------------------------------------------------------
def train_lr(X_train, y_train):
    """Train Logistic Regression baseline model."""
    model = LogisticRegression(max_iter=300, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# ------------------------------------------------------------
# Evaluate Model
# ------------------------------------------------------------
def evaluate_model(model, X, y, title="Evaluation"):
    """Generate evaluation metrics and confusion matrix."""
    y_pred = model.predict(X)
    print(f"\n--- {title} ---")
    print(classification_report(y, y_pred))
    print("Accuracy:", accuracy_score(y, y_pred))

    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    sns.heatmap(pd.DataFrame(cm, index=model.classes_, columns=model.classes_),
                annot=True, fmt='d', cmap='Purples')
    plt.title(f"Confusion Matrix - {title}")
    plt.show()

# ------------------------------------------------------------
# Feature Inspection
# ------------------------------------------------------------
def show_top_words(model, tfidf, top_n=10):
    """Display top features for binary or multi-class LogisticRegression."""
    feature_names = np.array(tfidf.get_feature_names_out())

    if len(model.classes_) == 2:
        # Binary case: only one coefficient vector
        coef = model.coef_[0]
        top_pos = feature_names[np.argsort(coef)[-top_n:]]
        top_neg = feature_names[np.argsort(coef)[:top_n]]
        print(f"Top words for {model.classes_[1]} (positive): {', '.join(top_pos)}")
        print(f"Top words for {model.classes_[0]} (negative): {', '.join(top_neg)}")
    else:
        for i, cls in enumerate(model.classes_):
            coef = model.coef_[i]
            top_words = feature_names[np.argsort(coef)[-top_n:]]
            print(f"Top words for {cls}: {', '.join(top_words)}")


# ------------------------------------------------------------
# Save Model & Vectorizer
# ------------------------------------------------------------
def save_model(model, tfidf, path_prefix="baseline"):
    joblib.dump(model, f"{path_prefix}_lr_model.pkl")
    joblib.dump(tfidf, f"{path_prefix}_tfidf.pkl")
    print(f" Saved: {path_prefix}_lr_model.pkl and TF-IDF vectorizer.")

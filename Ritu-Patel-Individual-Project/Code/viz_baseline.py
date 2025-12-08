
# viz_baseline.py : Plot Insights for Improved Baseline Model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)


#  Load Dataset and Model

DATA_PATH = "Suicide_Detection.csv"
MODEL_PATH = "baseline_lr_model.pkl"
TFIDF_PATH = "baseline_tfidf.pkl"

df = pd.read_csv(DATA_PATH)
# Standardize column names to match training pipeline
if "post_text" in df.columns:
    df["text"] = df["title"].fillna("") + " " + df["post_text"].fillna("")

if "class" in df.columns:
    df["label"] = df["class"]

df = df[['text', 'label']].dropna().reset_index(drop=True)


model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

print("Loaded dataset and trained baseline model successfully.")


#  Text Cleaning for length plots only

df["clean_text"] = df["text"].astype(str).str.lower()
df["text_length"] = df["clean_text"].apply(lambda x: len(x.split()))



# ⃣ Visualizations


def plot_label_distribution():
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="label", palette="coolwarm")
    plt.title("Label Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_length_distribution():
    plt.figure(figsize=(7,5))
    sns.histplot(df["text_length"], bins=60, kde=True)
    plt.title("Distribution of Word Count in Posts")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_length_by_class():
    plt.figure(figsize=(7,5))
    sns.boxplot(data=df, x="label", y="text_length", palette="coolwarm")
    plt.title("Text Length Comparison by Class")
    plt.xlabel("Class")
    plt.ylabel("Word Count")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(X_test, y_test):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(X_test, y_test):
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label="suicide")
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Suicide Ideation Detection")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_top_predictive_words():
    feature_names = np.array(tfidf.get_feature_names_out())
    coef = model.coef_[0]

    top_pos_idx = np.argsort(coef)[-15:]
    top_neg_idx = np.argsort(coef)[:15]

    # Suicide Word Indicators
    plt.figure(figsize=(7,5))
    plt.barh(feature_names[top_pos_idx], coef[top_pos_idx], color="red")
    plt.title("Top Predictive Words for Suicide Class")
    plt.xlabel("Coefficient Weight")
    plt.tight_layout()
    plt.show()

    # Non-Suicide Indicators
    plt.figure(figsize=(7,5))
    plt.barh(feature_names[top_neg_idx], coef[top_neg_idx], color="green")
    plt.title("Top Predictive Words for Non-Suicide Class")
    plt.xlabel("Coefficient Weight")
    plt.tight_layout()
    plt.show()



# Generate TF-IDF test set to evaluate plots

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.20, stratify=df["label"], random_state=42
)

X_test_tfidf = tfidf.transform(X_test)



# Execute All Plots

def run_all_visualizations():
    print("\n Generating insights...")
    plot_label_distribution()
    plot_length_distribution()
    plot_length_by_class()
    plot_confusion_matrix(X_test_tfidf, y_test)
    plot_roc_curve(X_test_tfidf, y_test)
    plot_top_predictive_words()
    print("\n All visualizations generated successfully!")


if __name__ == "__main__":
    run_all_visualizations()

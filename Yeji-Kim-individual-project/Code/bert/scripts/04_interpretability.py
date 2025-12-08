"""
Interpretability Analysis for Suicide Detection Model (Standalone)

- Loads a fine-tuned HuggingFace model from `model_dir`
- Reads a CSV (train or test) directly
- Generates predictions inside this script (no external predictions CSV)
- Performs:
    * Error analysis (confusion matrix, length/word stats)
    * Keyword extraction using attention (CLEANED)
    * Attention visualizations on example texts
    * Summary JSON report
"""


import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import string

warnings.filterwarnings("ignore")

# ==========================
# Keyword cleaning config
# ==========================
STOPWORDS = {
    "i", "me", "my", "mine", "you", "your", "yours", "u",
    "he", "she", "it", "we", "they", "them", "their", "ours",
    "the", "a", "an", "and", "or", "but", "if", "so",
    "to", "of", "in", "on", "for", "with", "at", "by", "from",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those",
    "as", "about", "just", "really", "very", "like",
    "im", "i'm", "dont", "don't", "cant", "can't"
}

PUNCT = set(string.punctuation) | {"[unk]", "’", "“", "”"}
SPECIAL_TOKENS = {"[cls]", "[sep]", "[pad]", "[bos]", "[eos]"}


# ---------------------------
# Model wrapper
# ---------------------------

class ModelInterpreter:
    def __init__(self, model_dir, max_length=320, threshold=0.5):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.threshold = float(threshold)

        print(f"Loading model from: {model_dir}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            output_attentions=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"Using device: {self.device}")
        print(f"Using threshold: {self.threshold:.3f}")

    def predict_batch(self, texts, return_attentions=False):
        """Run the model on a batch of texts and return probs & preds."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=return_attentions)

        logits = outputs.logits  # (batch, num_labels)
        probs = torch.softmax(logits, dim=-1)[:, 1]  # p(class 1 = suicide)
        preds = (probs >= self.threshold).long()

        result = {
            "probs": probs.cpu().numpy(),
            "preds": preds.cpu().numpy(),
            "logits": logits.cpu().numpy(),
        }

        if return_attentions:
            result["attentions"] = outputs.attentions

        return result

    def get_attention_weights(self, text):
        """
        Get token list + attention values for a single text.
        Uses CLS->token attention averaged over layers & heads.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # attentions: list[num_layers] of (batch, num_heads, seq_len, seq_len)
        attentions = outputs.attentions
        # average over layers and heads -> (batch, seq_len, seq_len)
        avg_attn = torch.stack(attentions).mean(dim=(0, 2))
        cls_attention = avg_attn[0, 0, :].cpu().numpy()  # CLS → all tokens

        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0].cpu()
        )

        return tokens, cls_attention, outputs.logits.cpu()

    def visualize_attention(self, text, save_path=None):
        """Make a heat-like visualization of attention over tokens."""
        tokens, attention, logits = self.get_attention_weights(text)

        probs = torch.softmax(logits, dim=-1)[0, 1].item()
        pred_label = "SUICIDE" if probs >= self.threshold else "NON-SUICIDE"

        fig, ax = plt.subplots(figsize=(14, 3))

        max_attn = attention.max() if attention.max() > 0 else 1.0
        colors = plt.cm.Reds(attention / max_attn)

        y_pos = 0
        x_i = 0
        for tok, attn, color in zip(tokens, attention, colors):
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            ax.text(
                x_i, y_pos, tok.replace("##", ""),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=10
            )
            x_i += 1

        ax.set_xlim(-1, x_i + 1)
        ax.set_ylim(-1, 1)
        ax.axis("off")
        ax.set_title(
            f"Attention | Prediction: {pred_label} (p={probs:.3f})",
            fontsize=12,
            fontweight="bold"
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return tokens, attention, probs

    def extract_important_tokens(self, text, top_k=10):
        """
        Return top_k tokens + attention weights for a single text.
        (Used for per-example explanation, not the global keyword stats.)
        """
        tokens, attention, _ = self.get_attention_weights(text)
        token_importance = []

        for tok, attn in zip(tokens, attention):
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            clean = tok.replace("##", "").strip()
            if not clean:
                continue
            token_importance.append((clean, float(attn)))

        token_importance.sort(key=lambda x: x[1], reverse=True)
        return token_importance[:top_k]


# ---------------------------
# Data loading & predictions
# ---------------------------

def choose_text_column(df, user_text_col=None):
    """Pick which text column to use."""
    if user_text_col:
        if user_text_col not in df.columns:
            raise ValueError(f"Requested text column '{user_text_col}' not in CSV.")
        print(f"Using user-specified text column: {user_text_col}")
        return user_text_col

    cols = df.columns.tolist()
    print(f"Available columns: {cols}")

    # Prefer 'clean_text' if it looks non-empty/substantial
    if "clean_text" in df.columns:
        non_empty = df["clean_text"].astype(str).str.strip().str.len() > 0
        if non_empty.sum() > 0 and df["clean_text"].astype(str).str.strip().str.len().mean() > 5:
            print("Using 'clean_text' as text column.")
            return "clean_text"
        else:
            print("'clean_text' exists but seems mostly empty or very short. Falling back to 'text' if available.")

    if "text" in df.columns:
        print("Using 'text' as text column.")
        return "text"

    raise ValueError("Could not find a suitable text column (expected 'text' or 'clean_text').")


def load_dataset(csv_path, text_column=None):
    """Read CSV robustly and return (df, chosen_text_col)."""
    csv_path = Path(csv_path)
    print(f"\nReading CSV from: {csv_path}")

    df = pd.read_csv(
        csv_path,
        engine="python",
        on_bad_lines="skip"
    )
    print(f"Loaded {len(df)} rows (after skipping bad lines).")

    if "y" not in df.columns:
        raise ValueError("CSV must contain a 'y' column with labels (0/1).")

    chosen_col = choose_text_column(df, user_text_col=text_column)

    df[chosen_col] = df[chosen_col].fillna("").astype(str)
    df["y"] = df["y"].astype(int)

    return df, chosen_col


def generate_predictions(interpreter, df, text_col, batch_size=64):
    """Run model on all rows and return probs, preds, labels, texts."""
    texts = df[text_col].tolist()
    labels = df["y"].values

    print(f"\n=== Generating Predictions ===")
    print(f"Total samples: {len(texts)}")

    all_probs = []
    all_preds = []

    for i in range(0, len(texts), batch_size):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(texts)}...")
        batch_texts = texts[i: i + batch_size]
        res = interpreter.predict_batch(batch_texts)
        all_probs.extend(res["probs"])
        all_preds.extend(res["preds"])

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    acc = (all_preds == labels).mean()
    print(f"✓ Predictions complete! Accuracy: {acc:.4f}")

    return all_probs, all_preds, labels, texts


# ---------------------------
# Error analysis
# ---------------------------

def analyze_errors(probs, preds, labels, texts, output_dir):
    """Compute confusion, distributions, and save plots/CSVs."""
    print("\n=== Error Analysis ===")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "pred_label": preds,
        "prob_suicide": probs
    })

    df = df.dropna(subset=["text", "true_label", "pred_label", "prob_suicide"])

    df["text_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().str.len()

    correct = df[df["true_label"] == df["pred_label"]]
    errors = df[df["true_label"] != df["pred_label"]]

    fp = errors[errors["pred_label"] == 1]
    fn = errors[errors["pred_label"] == 0]

    total = len(df)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {len(correct)} ({len(correct)/total*100:.2f}%)")
    print(f"Errors: {len(errors)} ({len(errors)/total*100:.2f}%)")
    if len(errors) > 0:
        print(f"  - False Positives: {len(fp)} ({len(fp)/len(errors)*100:.2f}% of errors)")
        print(f"  - False Negatives: {len(fn)} ({len(fn)/len(errors)*100:.2f}% of errors)")
    else:
        print("No errors at all (perfect accuracy).")

    def stats_subset(sub):
        if len(sub) == 0:
            return {"mean_length": 0.0, "mean_words": 0.0, "count": 0}
        return {
            "mean_length": float(sub["text_length"].mean()),
            "mean_words": float(sub["word_count"].mean()),
            "count": int(len(sub))
        }

    stats = {
        "correct": stats_subset(correct),
        "false_positives": stats_subset(fp),
        "false_negatives": stats_subset(fn),
    }

    # Save detailed CSVs
    df.to_csv(output_dir / "all_predictions.csv", index=False)
    if len(fp) > 0:
        fp.head(50).to_csv(output_dir / "false_positives_examples.csv", index=False)
    if len(fn) > 0:
        fn.head(50).to_csv(output_dir / "false_negatives_examples.csv", index=False)

    # 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Confusion matrix
    cm = confusion_matrix(df["true_label"], df["pred_label"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
        xticklabels=["Non-Suicide", "Suicide"],
        yticklabels=["Non-Suicide", "Suicide"]
    )
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted Label")
    axes[0, 0].set_ylabel("True Label")

    # 2) Probability distributions
    pos_true = df[df["true_label"] == 1]["prob_suicide"]
    neg_true = df[df["true_label"] == 0]["prob_suicide"]
    axes[0, 1].hist(pos_true, bins=30, alpha=0.5, label="True Suicide", color="red")
    axes[0, 1].hist(neg_true, bins=30, alpha=0.5, label="True Non-Suicide", color="blue")
    axes[0, 1].set_title("Predicted Suicide Probability by True Label")
    axes[0, 1].set_xlabel("Predicted P(suicide)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # 3) Text length distribution
    data_len = [correct["text_length"].values]
    labels_len = ["Correct"]
    if len(fp) > 0:
        data_len.append(fp["text_length"].values)
        labels_len.append("False Pos")
    if len(fn) > 0:
        data_len.append(fn["text_length"].values)
        labels_len.append("False Neg")

    axes[1, 0].boxplot(data_len, labels=labels_len)
    axes[1, 0].set_title("Text Length by Prediction Type")
    axes[1, 0].set_ylabel("Characters")

    # 4) Word count distribution
    data_wc = [correct["word_count"].values]
    labels_wc = ["Correct"]
    if len(fp) > 0:
        data_wc.append(fp["word_count"].values)
        labels_wc.append("False Pos")
    if len(fn) > 0:
        data_wc.append(fn["word_count"].values)
        labels_wc.append("False Neg")

    axes[1, 1].boxplot(data_wc, labels=labels_wc)
    axes[1, 1].set_title("Word Count by Prediction Type")
    axes[1, 1].set_ylabel("Words")

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Classification report
    report = classification_report(
        df["true_label"], df["pred_label"],
        target_names=["Non-Suicide", "Suicide"]
    )
    (output_dir / "classification_report.txt").write_text(report)

    return stats, fp, fn


# ---------------------------
# Keyword extraction (CLEANED)
# ---------------------------

def extract_predictive_keywords(interpreter, probs, preds, labels, texts,
                                output_dir, n_samples=500, top_k_per_example=5,
                                min_count=10, top_k_tokens=20):
    """
    Use attention-based per-example important tokens, then:
      - filter stopwords/punctuation/special tokens
      - aggregate per-token attention weights separately for each class
      - compute class-specific mean attention and differences
    """
    print("\n=== Extracting Predictive Keywords (CLEANED) ===")
    output_dir = Path(output_dir)

    labels = np.asarray(labels)
    preds = np.asarray(preds)

    correct_mask = preds == labels
    suicide_mask = (labels == 1) & correct_mask
    non_suicide_mask = (labels == 0) & correct_mask

    suicide_indices = np.where(suicide_mask)[0][:n_samples]
    non_suicide_indices = np.where(non_suicide_mask)[0][:n_samples]

    print(f"Analyzing {len(suicide_indices)} correctly-predicted suicide examples...")
    print(f"Analyzing {len(non_suicide_indices)} correctly-predicted non-suicide examples...")

    # Aggregate raw token stats (sum of weights + counts) per class
    totals_su = defaultdict(float)
    counts_su = defaultdict(int)

    for idx in suicide_indices:
        try:
            important = interpreter.extract_important_tokens(
                texts[idx], top_k=top_k_per_example
            )
            for tok, w in important:
                tok_l = tok.lower().strip()
                if len(tok_l) == 1 and tok_l not in {"i"}:
                    continue
                if not any(ch.isalpha() for ch in tok_l):
                    continue
                if not tok_l:
                    continue
                if tok_l in SPECIAL_TOKENS:
                    continue
                if tok_l in PUNCT:
                    continue
                if tok_l in STOPWORDS:
                    continue
                totals_su[tok_l] += float(w)
                counts_su[tok_l] += 1
        except Exception:
            continue

    totals_ns = defaultdict(float)
    counts_ns = defaultdict(int)

    for idx in non_suicide_indices:
        try:
            important = interpreter.extract_important_tokens(
                texts[idx], top_k=top_k_per_example
            )
            for tok, w in important:
                tok_l = tok.lower().strip()
                if len(tok_l) == 1 and tok_l not in {"i"}:
                    continue
                if not any(ch.isalpha() for ch in tok_l):
                    continue
                if not tok_l:
                    continue
                if tok_l in SPECIAL_TOKENS:
                    continue
                if tok_l in PUNCT:
                    continue
                if tok_l in STOPWORDS:
                    continue
                totals_ns[tok_l] += float(w)
                counts_ns[tok_l] += 1
        except Exception:
            continue

    # Build merged stats: mean_attention per class + differences
    tokens = set(list(totals_su.keys()) + list(totals_ns.keys()))
    rows = []
    for tok in tokens:
        c_su = counts_su.get(tok, 0)
        c_ns = counts_ns.get(tok, 0)
        if c_su + c_ns < min_count:
            continue

        mean_su = totals_su.get(tok, 0.0) / c_su if c_su > 0 else 0.0
        mean_ns = totals_ns.get(tok, 0.0) / c_ns if c_ns > 0 else 0.0
        diff_su = mean_su - mean_ns
        diff_ns = mean_ns - mean_su

        rows.append({
            "token": tok,
            "mean_suicide": mean_su,
            "mean_non_suicide": mean_ns,
            "diff_suicide": diff_su,
            "diff_non_suicide": diff_ns,
            "count_total": c_su + c_ns
        })

    if not rows:
        print("No tokens passed the frequency threshold for keyword extraction.")
        keywords_data = {
            "suicide_keywords": [],
            "non_suicide_keywords": [],
        }
        (output_dir / "predictive_keywords_clean.json").write_text(
            json.dumps(keywords_data, indent=2)
        )
        return keywords_data

    kw_df = pd.DataFrame(rows)

    # Top tokens that are more attended in suicide vs non-suicide, and vice versa
    su_df = kw_df.sort_values("diff_suicide", ascending=False)
    ns_df = kw_df.sort_values("diff_non_suicide", ascending=False)

    top_su_tokens = su_df.head(top_k_tokens)
    top_ns_tokens = ns_df.head(top_k_tokens)

    # JSON structure similar to original, but now with "weight" = diff_suicide / diff_non_suicide
    suicide_keywords = [
        {
            "word": row["token"],
            "weight": float(row["diff_suicide"]),
            "mean_suicide": float(row["mean_suicide"]),
            "mean_non_suicide": float(row["mean_non_suicide"]),
            "count_total": int(row["count_total"])
        }
        for _, row in top_su_tokens.iterrows()
        if row["diff_suicide"] > 0
    ]

    non_suicide_keywords = [
        {
            "word": row["token"],
            "weight": float(row["diff_non_suicide"]),
            "mean_suicide": float(row["mean_suicide"]),
            "mean_non_suicide": float(row["mean_non_suicide"]),
            "count_total": int(row["count_total"])
        }
        for _, row in top_ns_tokens.iterrows()
        if row["diff_non_suicide"] > 0
    ]

    keywords_data = {
        "suicide_keywords": suicide_keywords,
        "non_suicide_keywords": non_suicide_keywords,
    }

    (output_dir / "predictive_keywords_clean.json").write_text(
        json.dumps(keywords_data, indent=2)
    )

    if suicide_keywords:
        print("Top suicide indicators (cleaned):",
              ", ".join([k["word"] for k in suicide_keywords[:10]]))
    if non_suicide_keywords:
        print("Top non-suicide indicators (cleaned):",
              ", ".join([k["word"] for k in non_suicide_keywords[:10]]))

    # Plot clean bar charts
    plt.figure(figsize=(14, 6))

    # Suicide side
    ax1 = plt.subplot(1, 2, 1)
    if suicide_keywords:
        words_su = [k["word"] for k in suicide_keywords[:top_k_tokens]]
        weights_su = [k["weight"] for k in suicide_keywords[:top_k_tokens]]
        ax1.barh(words_su, weights_su, color="red", alpha=0.7)
        ax1.invert_yaxis()
        ax1.set_title("Top Suicide-Specific Keywords (Cleaned)")
        ax1.set_xlabel("Attention Difference (Suicide - Non-Suicide)")
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center")
        ax1.set_title("Top Suicide-Specific Keywords (Cleaned)")

    # Non-suicide side
    ax2 = plt.subplot(1, 2, 2)
    if non_suicide_keywords:
        words_ns = [k["word"] for k in non_suicide_keywords[:top_k_tokens]]
        weights_ns = [k["weight"] for k in non_suicide_keywords[:top_k_tokens]]
        ax2.barh(words_ns, weights_ns, color="blue", alpha=0.7)
        ax2.invert_yaxis()
        ax2.set_title("Top Non-Suicide-Specific Keywords (Cleaned)")
        ax2.set_xlabel("Attention Difference (Non-Suicide - Suicide)")
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center")
        ax2.set_title("Top Non-Suicide-Specific Keywords (Cleaned)")

    plt.tight_layout()
    plt.savefig(output_dir / "predictive_keywords_clean.png", dpi=300, bbox_inches="tight")
    plt.close()

    return keywords_data


# ---------------------------
# Attention visualizations
# ---------------------------

def visualize_attention_examples(interpreter, probs, preds, labels, texts,
                                 output_dir, n_examples=5, seed=42):
    print("\n=== Generating Attention Visualizations ===")
    output_dir = Path(output_dir)
    attn_dir = output_dir / "attention_visualizations"
    attn_dir.mkdir(exist_ok=True)

    labels = np.asarray(labels)
    preds = np.asarray(preds)
    correct_mask = preds == labels

    suicide_indices = np.where((labels == 1) & correct_mask)[0]
    non_suicide_indices = np.where((labels == 0) & correct_mask)[0]

    np.random.seed(seed)
    suicide_sample = np.random.choice(
        suicide_indices,
        size=min(n_examples, len(suicide_indices)),
        replace=False
    ) if len(suicide_indices) > 0 else []

    non_suicide_sample = np.random.choice(
        non_suicide_indices,
        size=min(n_examples, len(non_suicide_indices)),
        replace=False
    ) if len(non_suicide_indices) > 0 else []

    print(f"Creating visualizations for {len(suicide_sample)} suicide examples...")
    for i, idx in enumerate(suicide_sample):
        save_path = attn_dir / f"suicide_example_{i}.png"
        interpreter.visualize_attention(texts[idx], save_path=save_path)

    print(f"Creating visualizations for {len(non_suicide_sample)} non-suicide examples...")
    for i, idx in enumerate(non_suicide_sample):
        save_path = attn_dir / f"non_suicide_example_{i}.png"
        interpreter.visualize_attention(texts[idx], save_path=save_path)

    print(f"Saved attention visualizations to: {attn_dir}")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Interpretability analysis (standalone)")
    ap.add_argument("--model_dir", default="../models/bert_base_uncased_single",
                    help="Path to fine-tuned model directory")
    ap.add_argument("--test_csv", default="../data/processed/test.csv",
                    help="CSV with columns: y, text/clean_text")
    ap.add_argument("--output_dir", default="../outputs/interpretability",
                    help="Directory to save interpretability results")
    ap.add_argument("--max_length", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for class 1 (suicide)")
    ap.add_argument("--n_attention_examples", type=int, default=5)
    ap.add_argument("--n_keyword_samples", type=int, default=500)
    ap.add_argument("--text_column", type=str, default=None,
                    help="Force a particular text column name instead of auto-detect")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INTERPRETABILITY ANALYSIS (STANDALONE)")
    print("=" * 60)

    # 1) Load data
    df, text_col = load_dataset(args.test_csv, text_column=args.text_column)

    # 2) Init model
    interpreter = ModelInterpreter(
        model_dir=args.model_dir,
        max_length=args.max_length,
        threshold=args.threshold
    )

    # 3) Predictions
    probs, preds, labels, texts = generate_predictions(
        interpreter, df, text_col, batch_size=args.batch_size
    )

    # 4) Error analysis
    error_stats, fp, fn = analyze_errors(
        probs, preds, labels, texts, output_dir
    )
    (output_dir / "error_statistics.json").write_text(
        json.dumps(error_stats, indent=2)
    )

    # 5) Keyword extraction (CLEANED)
    keywords = extract_predictive_keywords(
        interpreter, probs, preds, labels, texts, output_dir,
        n_samples=args.n_keyword_samples
    )

    # 6) Attention visualizations
    visualize_attention_examples(
        interpreter, probs, preds, labels, texts, output_dir,
        n_examples=args.n_attention_examples
    )

    # 7) Summary
    print("\n=== Generating Summary Report ===")
    accuracy = float((preds == labels).mean())

    # pick only top 10 words for the summary
    top_su_words = [k["word"] for k in keywords["suicide_keywords"][:10]]
    top_ns_words = [k["word"] for k in keywords["non_suicide_keywords"][:10]]

    summary = {
        "model_dir": str(args.model_dir),
        "test_csv": str(args.test_csv),
        "text_column_used": text_col,
        "test_samples": int(len(labels)),
        "accuracy": accuracy,
        "error_analysis": error_stats,
        "top_suicide_keywords": top_su_words,
        "top_non_suicide_keywords": top_ns_words,
        "attention_visualizations_generated": int(args.n_attention_examples * 2),
        "threshold": args.threshold,
    }

    (output_dir / "interpretability_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n" + "=" * 60)
    print("INTERPRETABILITY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("  - Error analysis: error_analysis.png")
    print("  - Classification report: classification_report.txt")
    print("  - Predictive keywords (clean): predictive_keywords_clean.png / .json")
    print("  - Attention visualizations: attention_visualizations/")
    print("  - All predictions: all_predictions.csv")
    print("  - Summary: interpretability_summary.json")


if __name__ == "__main__":
    main()

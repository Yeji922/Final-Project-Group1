import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ---- Config ----
PREDICTIONS_CSV = "outputs/interpretability_ckpt34800_clean/all_predictions.csv"
OUT_DIR = "outputs/interpretability_ckpt34800_clean"

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions
    df = pd.read_csv(PREDICTIONS_CSV)

    # y_true: ground-truth labels, y_score: model probability for suicide
    y_true = df["true_label"].values
    y_score = df["prob_suicide"].values

    # 2. ROC curve + AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Suicide Detection")
    plt.legend(loc="lower right")
    roc_path = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve to: {roc_path} (AUC = {roc_auc:.4f})")

    # 3. Precision-Recall curve + Average Precision (AP)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve - Suicide Detection")
    plt.legend(loc="lower left")
    pr_path = out_dir / "precision_recall_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"Saved Precision–Recall curve to: {pr_path} (AP = {ap:.4f})")

    # 4. Save the scalar metrics as JSON for your report
    metrics = {
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
    }
    (out_dir / "roc_pr_metrics.json").write_text(
        pd.Series(metrics).to_json(indent=2)
    )
    print("Saved ROC/PR metrics to:", out_dir / "roc_pr_metrics.json")


if __name__ == "__main__":
    main()

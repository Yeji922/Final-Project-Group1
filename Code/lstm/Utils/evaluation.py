import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

def evaluate_model(model, data_loader, label_encoder, device="cpu", save_path="saved_model"):
    """
    Runs full evaluation: classification report, confusion matrix,
    accuracy, precision, recall, F1 for each class.
    Saves confusion matrix as PNG.
    """

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            outputs, _ = model(X)
            preds = outputs.argmax(1).cpu().numpy()
            labels = y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    #metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )

    print("\n====== Evaluation Results ======")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")

    #classification report
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:\n")
    print(report)

    #confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    save_fp = f"{save_path}/confusion_matrix.png"
    plt.savefig(save_fp, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nConfusion matrix saved to: {save_fp}")

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "report": report,
        "confusion_matrix": cm
    }

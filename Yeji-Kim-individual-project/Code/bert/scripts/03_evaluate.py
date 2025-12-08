import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def load_split(path):
    df = pd.read_csv(path)
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    return Dataset.from_pandas(df[[text_col, "y"]].rename(columns={text_col: "text"}), preserve_index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--model_dir", default="models/bert_base_uncased")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--max_length", type=int, default=320)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ds = load_split(data_dir / "test.csv")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    threshold = joblib.load(Path(args.model_dir) / "threshold.joblib")

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_length)
    test_tok = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

    # Inference
    import torch
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(test_tok), 64):
            batch = test_tok[i:i+64]
            inputs = {k: torch.tensor(batch[k]) for k in ["input_ids", "attention_mask"] if k in batch.features}
            out = model(**inputs)
            logits.append(out.logits.cpu().numpy())
    logits = np.vstack(logits)
    probs = 1 / (1 + np.exp(-logits[:, 1] + logits[:, 0]))
    preds = (probs >= threshold).astype(int)
    y_true = np.array(test_ds["y"])

    # Metrics
    report = classification_report(y_true, preds, digits=4)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")

    # Save
    (out_dir / "classification_report.txt").write_text(report + f"\nROC_AUC: {auc:.4f}\n")
    pd.DataFrame({"prob_suicide": probs, "pred": preds, "label": y_true}).to_csv(out_dir / "test_predictions.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    (out_dir / "confusion_matrix.json").write_text(json.dumps(cm.tolist(), indent=2))
    print("Saved evaluation artifacts to", out_dir)

if __name__ == "__main__":
    main()

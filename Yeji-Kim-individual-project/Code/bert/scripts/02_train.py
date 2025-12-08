import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import evaluate
import joblib
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback)
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def load_split(path):
    df = pd.read_csv(path)
    # Prefer clean_text
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    return Dataset.from_pandas(df[[text_col, "y"]].rename(columns={text_col: "text"}), preserve_index=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits[:, 1] + logits[:, 0]))  # convert to pseudo-prob for class 1
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    acc = (preds == labels).mean()
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="models/bert_base_uncased")
    ap.add_argument("--max_length", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_split(data_dir / "train.csv")
    val_ds = load_split(data_dir / "val.csv")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_length)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    trainer.train()

    # Save best model + tokenizer
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Choose threshold on val for best F1 (positive class)
    preds = trainer.predict(val_ds)
    logits = preds.predictions
    probs = 1 / (1 + np.exp(-logits[:, 1] + logits[:, 0]))
    best_f1, best_t = -1, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        pred_bin = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(val_ds["y"], pred_bin, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    joblib.dump(best_t, out_dir / "threshold.joblib")

    # Save simple metadata
    meta = {"model_name": args.model_name, "max_length": args.max_length, "best_val_f1": float(best_f1), "threshold": float(best_t)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {out_dir}. Best threshold={best_t:.3f}, F1={best_f1:.4f}")

if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict_texts(texts, model_dir, max_length=320):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    threshold = joblib.load(Path(model_dir) / "threshold.joblib")
    model.eval()

    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = 1 / (1 + torch.exp(-(logits[:,1] - logits[:,0])))
    preds = (probs >= threshold).long().cpu().numpy()
    return probs.cpu().numpy(), preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/bert_base_uncased")
    ap.add_argument("--text", default=None, help="Single text input")
    ap.add_argument("--csv", default=None, help="Path to CSV with a 'text' column")
    ap.add_argument("--out", default="outputs/infer_predictions.csv")
    ap.add_argument("--max_length", type=int, default=320)
    args = ap.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    if args.text:
        probs, preds = predict_texts([args.text], args.model_dir, args.max_length)
        print(f"prob_suicide={float(probs[0]):.4f}, pred={int(preds[0])}")
    elif args.csv:
        df = pd.read_csv(args.csv)
        col = "clean_text" if "clean_text" in df.columns else "text"
        probs, preds = predict_texts(df[col].astype(str).tolist(), args.model_dir, args.max_length)
        out_df = pd.DataFrame({ "prob_suicide": probs, "pred": preds })
        out_df.to_csv(args.out, index=False)
        print(f"Saved predictions to {args.out}")
    else:
        raise SystemExit("Provide either --text or --csv.")

if __name__ == "__main__":
    main()

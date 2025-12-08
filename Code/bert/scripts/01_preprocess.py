import argparse, re, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MD_RE = re.compile(r"\[(.*?)\]\((.*?)\)")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = MD_RE.sub(r"\1", s)          # keep anchor text, drop URL
    s = URL_RE.sub(" ", s)            # drop raw URLs
    s = re.sub(r"\s+", " ", s).strip()
    return s

def map_label(v):
    s = str(v).strip().lower()
    if s in {"1", "suicide", "suicidal", "suicidal_ideation"}:
        return 1
    if s in {"0", "non-suicide", "nonsuicide", "normal"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        raise ValueError(f"Unrecognized label: {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", default="data/raw/Suicide_Detection.csv")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.5)  # of temp split
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="class")
    ap.add_argument("--max_length", type=int, default=320, help="Stored in stats for downstream tokenizer choice")
    args = ap.parse_args()

    raw = Path(args.raw_csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw, encoding="utf-8", engine="python")
    df = df[[args.text_col, args.label_col]].rename(columns={args.text_col: "text", args.label_col: "label"})
    df["clean_text"] = df["text"].map(clean_text)
    df["y"] = df["label"].map(map_label)

    # drop empties & exact duplicates
    before = len(df)
    df = df.dropna(subset=["clean_text", "y"])
    df = df.drop_duplicates(subset=["clean_text", "y"])
    after = len(df)

    # stratified split 80/10/10
    train_df, temp_df = train_test_split(df, test_size=args.test_size, random_state=42, stratify=df["y"])
    val_df, test_df = train_test_split(temp_df, test_size=args.val_size, random_state=42, stratify=temp_df["y"])

    train_path = out / "train.csv"
    val_path = out / "val.csv"
    test_path = out / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    stats = {
        "rows_before": int(before),
        "rows_after": int(after),
        "dropped": int(before - after),
        "class_balance": train_df["y"].value_counts(normalize=True).to_dict(),
        "max_length_suggested": args.max_length
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    joblib.dump({0: "non-suicide", 1: "suicide"}, out / "label_map.joblib")

    print(f"Saved splits to {out}. Stats: {stats}")

if __name__ == "__main__":
    main()

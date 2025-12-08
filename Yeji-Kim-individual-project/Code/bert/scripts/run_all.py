import subprocess, sys, os
from pathlib import Path

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    # 0) Fetch data
    src = os.environ.get("SRC_CSV", "data/raw/Suicide_Detection.csv")
    if not Path(src).exists():
        print("NOTE: Set SRC_CSV to your CSV path or copy it to data/raw/Suicide_Detection.csv before running.")
    else:
        run([sys.executable, "scripts/00_fetch_data.py", "--src", src])

    # 1) Preprocess
    run([sys.executable, "scripts/01_preprocess.py", "--max_length", "320"])

    # 2) Train
    run([sys.executable, "scripts/02_train.py", "--model_name", "bert-base-uncased", "--batch_size", "16", "--epochs", "3", "--lr", "2e-5", "--max_length", "320"])

    # 3) Evaluate
    run([sys.executable, "scripts/03_evaluate.py", "--max_length", "320"])

if __name__ == "__main__":
    main()

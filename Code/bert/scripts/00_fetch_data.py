import argparse
from pathlib import Path
import shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to Suicide_Detection.csv")
    ap.add_argument("--dest", default="data/raw/Suicide_Detection.csv", help="Destination path under project")
    args = ap.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    print(f"Copied {src} -> {dest}")

if __name__ == "__main__":
    main()

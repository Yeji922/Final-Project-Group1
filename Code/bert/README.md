# BERT Suicide-Detection Pipeline

A clean, modular pipeline to fine-tune BERT on the Reddit suicide-detection dataset.

## Structure
```
bert_pipeline/
├─ data/
│  ├─ raw/                # raw source CSV goes here
│  └─ processed/          # cleaned & stratified splits (train/val/test)
├─ models/
│  └─ bert_base_uncased/  # best checkpoint + tokenizer + joblib artifacts
├─ outputs/               # metrics, predictions, reports
├─ scripts/
│  ├─ 00_fetch_data.py    # copy/import the dataset into data/raw
│  ├─ 01_preprocess.py    # minimal clean, dedup, stratified split (80/10/10)
│  ├─ 02_train.py         # fine-tune BERT using HF Trainer (early stopping)
│  ├─ 03_evaluate.py      # evaluate on test set, save predictions & reports
│  ├─ 04_interpretability.py      # interpretability
│  ├─ 05_curves.py      # visualiztions
│  ├─ 99_infer.py         # inference for single text or a CSV
│  └─ run_all.py          # orchestrate steps 00-03
├─ requirements.txt
└─ README.md
```

## Quickstart
1. **Install deps** (ideally in a virtualenv):
   ```bash
   pip install -r requirements.txt
   ```

2. **Fetch data** (copies your CSV):
   ```bash
   python scripts/00_fetch_data.py --src "/PATH/TO/Suicide_Detection.csv"
   ```

3. **Preprocess**:
   ```bash
   python scripts/01_preprocess.py --max_length 320
   ```

4. **Train**:
   ```bash
   python scripts/02_train.py --model_name bert-base-uncased --batch_size 16 --epochs 3 --lr 2e-5 --max_length 320
   ```

5. **Evaluate**:
   ```bash
   python scripts/03_evaluate.py
   ```

6. **Infer** (single text):
   ```bash
   python scripts/99_infer.py --text "I feel hopeless and can't go on."
   ```

7. **Infer** (CSV with a `text` column):
   ```bash
   python scripts/99_infer.py --csv "/path/to/new_posts.csv" --out "/path/to/preds.csv"
   ```

### Notes
- Labels: binary (0 = non-suicide, 1 = suicide).
- We save several artifacts in `models/bert_base_uncased/`: best model, tokenizer, a tuned threshold (`threshold.joblib`), and label mapping (`label_map.joblib`). Threshold is chosen to maximize F1 for the positive class on the validation set.
- To capture longer posts, try `--max_length 480` in both preprocess & train.

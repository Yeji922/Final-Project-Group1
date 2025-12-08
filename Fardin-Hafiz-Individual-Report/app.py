import os
import streamlit as st
import joblib
import torch
import numpy as np
from pathlib import Path
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.set_num_threads(1)

IMPROVED_TFIDF = "improved_baseline_tfidf.pkl"
IMPROVED_MODEL = "improved_baseline_lr.pkl"

BERT_MODEL_DIR = "models/bert_base_uncased_single"

LSTM_MODEL_PATH = "saved_model/lstm_attention_model.pt"
LSTM_VOCAB_PATH = "saved_model/vocab.pkl"
LSTM_LABEL_PATH = "saved_model/label_encoder.pkl"


@st.cache_resource
def load_bert_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    threshold = joblib.load(Path(BERT_MODEL_DIR) / "threshold.joblib")

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, threshold, device


def predict_with_bert(text: str):
    model, tokenizer, threshold, device = load_bert_model()

    encoded = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=320,
        return_tensors="pt"
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prob_suicide = probs[:, 1].cpu().numpy()[0]

    pred = 1 if prob_suicide >= threshold else 0
    pred_label = "suicide" if pred == 1 else "non-suicide"

    return pred_label, prob_suicide


@st.cache_resource
def load_improved():
    tfidf = joblib.load(IMPROVED_TFIDF)
    model = joblib.load(IMPROVED_MODEL)
    return tfidf, model


def predict_improved(text: str):
    tfidf, model = load_improved()
    X = tfidf.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    classes = model.classes_
    suicide_idx = None
    for i, c in enumerate(classes):
        if str(c).lower() == "suicide":
            suicide_idx = i
            break
    if suicide_idx is None:
        suicide_idx = 1 if len(classes) > 1 else 0

    prob_suicide = proba[suicide_idx]

    return str(pred), float(prob_suicide)


@st.cache_resource
def load_lstm_components():
    from Utils.model import LSTMAttentionClassifier
    import pickle

    with open(LSTM_VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(LSTM_LABEL_PATH, "rb") as f:
        encoder = pickle.load(f)

    model = LSTMAttentionClassifier(
        vocab_size=len(vocab),
        embedding_dim=300,
        hidden_dim=128,
        output_dim=len(encoder.classes_),
        bidirectional=True
    )
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, vocab, encoder


def predict_lstm(text: str):
    model, vocab, encoder = load_lstm_components()

    tokens = text.lower().split()
    if not tokens:
        return "N/A", None

    indices = torch.tensor([[vocab.get(tok, 1) for tok in tokens]])

    with torch.no_grad():
        outputs, attn = model(indices)
        pred = outputs.argmax(1).item()
        probs = torch.softmax(outputs, dim=-1)[0].cpu().numpy()

    label = encoder.inverse_transform([pred])[0]

    classes = list(encoder.classes_)
    if "suicide" in classes:
        suicide_idx = int(np.where(np.array(classes) == "suicide")[0][0])
    else:
        suicide_idx = 1 if len(classes) > 1 else 0

    prob_suicide = float(probs[suicide_idx])

    return str(label), prob_suicide


st.set_page_config(page_title="NLP Suicide Detection", layout="wide")

st.title("Suicidal Text Detection — DATS 6312")

st.write("Enter text below and click the button to see predictions from different models.")


user_text = st.text_area("Input text:", height=150, placeholder="Type the message you want to classify...")

if st.button("Run models", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running models..."):
            results = []

            try:
                pred_imp, proba_imp = predict_improved(user_text)
                results.append({
                    "Model": "Improved Baseline LR",
                    "Prediction": pred_imp,
                    "Confidence": f"{proba_imp:.4f}",
                })
            except Exception as e:
                results.append({"Model": "Improved Baseline LR", "Prediction": "Error", "Confidence": str(e)})
                st.error(f"**Improved Baseline LR Error:** {str(e)}")

            try:
                pred_bert, proba_bert = predict_with_bert(user_text)
                results.append({
                    "Model": "BERT",
                    "Prediction": pred_bert,
                    "Confidence": f"{proba_bert:.4f}",
                })
            except Exception as e:
                results.append({"Model": "BERT", "Prediction": "Error", "Confidence": str(e)})
                st.error(f"**BERT Error:** {str(e)}")

            try:
                pred_lstm, proba_lstm = predict_lstm(user_text)
                if proba_lstm is not None:
                    results.append({
                        "Model": "LSTM + Attention",
                        "Prediction": pred_lstm,
                        "Confidence": f"{proba_lstm:.4f}",
                    })
                else:
                    results.append({
                        "Model": "LSTM + Attention",
                        "Prediction": pred_lstm,
                        "Confidence": "—",
                    })
            except Exception as e:
                err_msg = str(e)
                results.append({"Model": "LSTM + Attention", "Prediction": "Error", "Confidence": err_msg})
                st.error(f"**LSTM + Attention Error:** {err_msg}")

            st.subheader("Results")
            st.table(results)

            gc.collect()

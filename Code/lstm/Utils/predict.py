import torch
import pickle
from Utils.model import LSTMAttentionClassifier
from pathlib import Path
from Utils.explainability import (
    plot_attention,
    highlight_text,
    plot_attention_distribution,
    get_attention_mapping
)

def predict_text(model_path, vocab_path, label_encoder_path, device="cpu", text=""):
    
    script_dir = Path(__file__).resolve().parent
    model_path = (script_dir.parent / model_path).resolve()
    vocab_path = (script_dir.parent / vocab_path).resolve()
    label_encoder_path = (script_dir.parent / label_encoder_path).resolve()

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    #model
    model = LSTMAttentionClassifier(
        vocab_size=len(vocab),
        embedding_dim=300,
        hidden_dim=128,
        output_dim=len(label_encoder.classes_),
        bidirectional=True
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

 
    if text:
        tokens = text.lower().split()
        indices = torch.tensor([[vocab.get(tok, 1) for tok in tokens]]).to(device)

        with torch.no_grad():
            outputs, attn = model(indices)
            pred = outputs.argmax(1).item()
            label = label_encoder.inverse_transform([pred])[0]

        attn_np = attn.squeeze().cpu().numpy()

        #explanability code
        plot_attention(text, attn_np, save_path="saved_model/attention_plot.png")
        plot_attention_distribution(attn_np, save_path="saved_model/attention_distribution.png")
        highlight_text(text, attn_np)

        # Optional mapping for report
        mapping = get_attention_mapping(text, attn_np)

        return label, attn_np, mapping
    else:
        while True:
            text = input("Enter text (type 'end' to quit):\n")
            if text.lower() == "end":
                break

            tokens = text.lower().split()
            indices = torch.tensor([[vocab.get(tok, 1) for tok in tokens]]).to(device)

            with torch.no_grad():
                outputs, attn = model(indices)
                pred = outputs.argmax(1).item()
                label = label_encoder.inverse_transform([pred])[0]

            attn_np = attn.squeeze().cpu().numpy()

            print(f"\nPredicted class: {label}")

            #explainability plots
            plot_attention(text, attn_np, save_path="saved_model/attention_plot.png")
            plot_attention_distribution(attn_np, save_path="saved_model/attention_distribution.png")
            highlight_text(text, attn_np)
            print(get_attention_mapping(text, attn_np))

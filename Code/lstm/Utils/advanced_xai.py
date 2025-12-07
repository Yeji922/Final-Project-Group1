import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

#captum for IG + LRP
try:
    from captum.attr import LayerIntegratedGradients, LayerLRP
except ImportError:
    LayerIntegratedGradients = None
    LayerLRP = None

# SHAP for model-agnostic explanations
try:
    import shap
except ImportError:
    shap = None



def _forward_logits(model, input_ids):
    """
    Simple wrapper: given token ids, return logits.
    """
    logits, _ = model(input_ids)
    return logits


def _forward_probs(model, input_ids):
    """
    Wrapper returning probabilities (for SHAP).
    """
    logits, _ = model(input_ids)
    return torch.softmax(logits, dim=-1)


#Integrated gradients (token-level attributions)
def integrated_gradients_for_text(model, text, vocab, label_encoder,
                                  device="cpu", target_class=None,
                                  n_steps=50, show_plot=True,
                                  save_path="saved_model/ig_importance.png"):
    """
    Computes Integrated Gradients on the embedding layer for a single text.
    Returns (tokens, importances). Optionally plots and saves a bar chart.
    """

    if LayerIntegratedGradients is None:
        raise ImportError("captum is not installed. Run: pip install captum")

    model.eval()
    model.to(device)

    tokens = text.lower().split()
    input_ids = torch.tensor([[vocab.get(tok, 1) for tok in tokens]], device=device)

    # If target_class=None, use model's predicted class
    if target_class is None:
        with torch.no_grad():
            logits = _forward_logits(model, input_ids)
            target_class = logits.argmax(dim=-1).item()

    # Baseline: all PAD tokens (index 0)
    baseline_ids = torch.zeros_like(input_ids)

    lig = LayerIntegratedGradients(lambda x: _forward_logits(model, x), model.embedding)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True
    )

    # Sum over embedding dimension → one score per token
    token_importances = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # Normalize for plotting
    norm_imp = (token_importances - token_importances.min()) / (
        token_importances.max() - token_importances.min() + 1e-9
    )

    if show_plot:
        plt.figure(figsize=(12, 2))
        plt.bar(range(len(tokens)), norm_imp, color="darkgreen")
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
        plt.title(f"Integrated Gradients Token Importances (class: {label_encoder.inverse_transform([target_class])[0]})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Saved] Integrated Gradients plot → {save_path}")

    return tokens, token_importances


#SHAP VALUES FOR LSTM (KernelExplainer over token positions)
def shap_for_texts(model, texts, vocab, device="cpu",
                   max_len=40, num_background=20, num_samples=100):
    """
    Uses SHAP KernelExplainer at the token-index level.
    Each feature = token position (0..max_len-1).
    This is approximate but good enough for a project.

    Returns shap_values (list per class), token_ids array, and tokenized_forms.
    """

    if shap is None:
        raise ImportError("shap is not installed. Run: pip install shap")

    model.eval()
    model.to(device)

    # Tokenize and pad/truncate
    def encode(text):
        tokens = text.lower().split()
        ids = [vocab.get(tok, 1) for tok in tokens][:max_len]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))  # PAD=0
        return ids, tokens

    encoded = [encode(t) for t in texts]
    token_ids = np.array([e[0] for e in encoded])
    token_lists = [e[1] for e in encoded]

    # Background set for KernelExplainer
    background = token_ids[:num_background]

    def f(batch_np):
        batch = torch.tensor(batch_np, dtype=torch.long, device=device)
        with torch.no_grad():
            probs = _forward_probs(model, batch)
        return probs.cpu().numpy()

    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(token_ids, nsamples=num_samples)

    return shap_values, token_ids, token_lists


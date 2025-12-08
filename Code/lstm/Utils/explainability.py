import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_attention(text, attn_weights, save_path="saved_model/attention_plot.png"):
    """
    Bar plot of attention weights for each token.
    """
    tokens = text.lower().split()
    attn = attn_weights[:len(tokens)]

    plt.figure(figsize=(12, 2))
    plt.bar(range(len(tokens)), attn, color="purple")
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.title("Token-level Attention Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Saved] Token-level attention heatmap → {save_path}")


def highlight_text(text, attn_weights):
    """
    Print text with background color intensity based on attention.
    Darker red = higher attention.
    """
    tokens = text.split()
    attn = attn_weights[:len(tokens)]

    # Normalize 0 → 1
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-9)

    highlighted_tokens = []
    for tok, score in zip(tokens, attn_norm):
        intensity = int(score * 255)
        highlighted_tokens.append(
            f"\033[48;2;255;{255-intensity};{255-intensity}m {tok} \033[0m"
        )

    print("\n========== Highlighted Text (Token → Attention) ==========")
    print(" ".join(highlighted_tokens))
    print("\n")



def plot_attention_distribution(attn_weights, save_path="saved_model/attention_distribution.png"):
    """
    Plot histogram + KDE distribution of attention weights.
    Helps visualize how peaky, sparse, or uniform attention is.
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(attn_weights, bins=20, kde=True, color="teal")
    plt.title("Attention Weight Distribution")
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Saved] Attention distribution plot - {save_path}")



def get_attention_mapping(text, attn_weights):
    """
    Returns dictionary {token: attention_weight}
    """
    tokens = text.split()
    attn = attn_weights[:len(tokens)]
    return {tok: float(w) for tok, w in zip(tokens, attn)}



def compare_attention_across_classes(model, dataset, label_encoder, device="cpu", 
                                     samples_per_class=5, save_path="saved_model/class_attention_comparison.png"):
    """
    Computes average attention patterns for each class.
    Steps:
    1. Pick N samples from each class.
    2. Compute model attention for each.
    3. Average attentions per class.
    4. Plot class-level attention curves.
    """

    print("\n[Running] Class-level attention comparison...")

    class_indices = {label: [] for label in np.unique(dataset.labels)}

    # Collect samples per class
    for idx, (_, label) in enumerate(zip(dataset.texts, dataset.labels)):
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)

    # Ensure classes have enough samples
    for label in class_indices:
        if len(class_indices[label]) == 0:
            print(f"[Warning] Not enough samples for class {label}")
            continue

    avg_attn_per_class = {}

    # Compute attention for each class
    for label, indices in class_indices.items():
        attns = []

        for idx in indices:
            text = dataset.texts[idx]
            seq = torch.tensor([dataset.text_to_sequence(text)]).to(device)

            with torch.no_grad():
                _, attn = model(seq)
                attn = attn.squeeze().cpu().numpy()
                attns.append(attn[:50])  # normalize sequence length for plotting

        # Pad shorter sequences with zeros for averaging
        max_len = max(len(a) for a in attns)
        attns_padded = [np.pad(a, (0, max_len - len(a))) for a in attns]

        avg_attn = np.mean(attns_padded, axis=0)
        class_name = label_encoder.inverse_transform([label])[0]
        avg_attn_per_class[class_name] = avg_attn

    # Plot averages
    plt.figure(figsize=(10, 5))
    for class_name, avg_attn in avg_attn_per_class.items():
        plt.plot(avg_attn, label=class_name)

    plt.title("Class-Level Average Attention Patterns")
    plt.xlabel("Token Position (0–50)")
    plt.ylabel("Attention Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Saved] Class-level attention comparison - {save_path}")
    return avg_attn_per_class

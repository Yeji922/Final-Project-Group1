import os
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from Utils.dataloader import TextDataset, collate_fn
from Utils.model import LSTMAttentionClassifier
import pickle

os.makedirs("saved_model", exist_ok=True)
log_path = "saved_model/log_file.txt"
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=5):
    model.to(device)
    with open(log_path , "w") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Acc\n")
    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == y).sum().item()
        print(f"Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {total_acc/len(train_loader.dataset):.4f}")

        model.eval()
        val_acc = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs, _ = model(X)
                val_acc += (outputs.argmax(1) == y).sum().item()
        print(f"Val Acc: {val_acc/len(val_loader.dataset):.4f}")

        with open(log_path, "a") as f:
            f.write(f"{epoch+1}, {total_loss/len(train_loader):.4f},{total_acc/len(train_loader.dataset):.4f},{val_acc/len(val_loader.dataset):.4f}\n")

    torch.save(model.state_dict(), "saved_model/lstm_attention_model.pt")

def train_pipeline(dataset, device = "cpu"):
    # Load CSV
    df = pd.read_csv(dataset)
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Build datasets
    train_data = TextDataset(train_df)
    val_data = TextDataset(val_df, vocab=train_data.vocab)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate_fn)

    # Model, optimizer, loss
    model = LSTMAttentionClassifier(
        vocab_size=len(train_data.vocab),
        embedding_dim=300,
        hidden_dim=128,
        output_dim=len(set(train_data.labels)),
        bidirectional=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=20)

    with open("saved_model/vocab.pkl", "wb") as f:
        pickle.dump(train_data.vocab, f)

    with open("saved_model/label_encoder.pkl", "wb") as f:
        pickle.dump(train_data.label_encoder, f)
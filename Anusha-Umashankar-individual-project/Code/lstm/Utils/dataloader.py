import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, df, vocab=None, max_vocab_size=10000):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["class"].tolist()
        self.tokenizer = lambda x: x.lower().split()

        if vocab is None:
            all_tokens = [token for text in self.texts for token in self.tokenizer(text)]
            freq = pd.Series(all_tokens).value_counts()
            vocab_tokens = freq.index[:max_vocab_size-2]
            self.vocab = {"<PAD>": 0, "<UNK>": 1}
            self.vocab.update({tok: i+2 for i, tok in enumerate(vocab_tokens)})
        else:
            self.vocab = vocab

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def text_to_sequence(self, text):
        return [self.vocab.get(tok, 1) for tok in self.tokenizer(text)]

    def __getitem__(self, idx):
        return torch.tensor(self.text_to_sequence(self.texts[idx])), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.texts)


def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return padded, labels

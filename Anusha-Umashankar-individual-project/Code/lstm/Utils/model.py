import torch
from torch import nn


class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        scores = self.Va(torch.tanh(self.Wa(decoder_hidden_expanded) + self.Ua(encoder_outputs)))
        attn_weights = torch.softmax(scores, dim=1)
        context_vector = torch.bmm(attn_weights.permute(0, 2, 1), encoder_outputs)
        return context_vector.squeeze(1), attn_weights.squeeze(-1)

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, bidirectional=True, dropout=0.3):
        super(LSTMAttentionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.attention = SimpleAttention(hidden_dim * 2 if bidirectional else hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        context, attn_weights = self.attention(hidden, lstm_out)
        output = self.fc(self.dropout(context))
        return output, attn_weights

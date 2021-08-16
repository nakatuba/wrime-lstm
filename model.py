import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vectors, output_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors)
        self.lstm = nn.LSTM(200, 64, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(64 * 2, output_dim)

    def forward(self, text):
        output = self.embedding(text)
        output, (hn, cn) = self.lstm(output)
        output = self.linear(output[-1])
        return output


class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_tasks):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, output_dim)] * num_tasks)

    def forward(self, text):
        output = self.embedding(text)
        output, (hn, cn) = self.lstm(output)
        output = [linear(output[-1]) for linear in self.linears]
        return output

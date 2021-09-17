import torch
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
        output = torch.cat([hn[-2], hn[-1]], dim=1)
        output = self.linear(output)
        return output


class MultiTaskLSTM(nn.Module):
    def __init__(self, vectors, n_classes, n_tasks):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors)
        self.lstm = nn.LSTM(200, 64, num_layers=2, bidirectional=True)
        self.linears = nn.ModuleList(
            [nn.Linear(64 * 2, n_classes) for _ in range(n_tasks)]
        )

    def forward(self, text):
        output = self.embedding(text)
        output, (hn, cn) = self.lstm(output)
        output = torch.cat([hn[-2], hn[-1]], dim=1)
        output = [linear(output) for linear in self.linears]
        return output

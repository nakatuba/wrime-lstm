import MeCab
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from model import LSTM
from utils.dataset import DifferenceDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DifferenceDataset("./data/train.tsv")
    valid_dataset = DifferenceDataset("./data/val.tsv")
    test_dataset = DifferenceDataset("./data/test.tsv")

    vocab = build_vocab_from_iterator(
        [tokenizer(text) for text in train_dataset.texts], specials=["<unk>", "<pad>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    def collate_batch(batch):
        text_list = [torch.tensor(vocab(tokenizer(text))) for text, _ in batch]
        label_list = [label for _, label in batch]
        text_list = pad_sequence(text_list, padding_value=vocab["<pad>"])
        label_list = torch.tensor(label_list)
        return text_list.to(device), label_list.to(device)

    batch_size = 32

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

    model = LSTM(len(vocab), 32, 32, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer)
        val_loss = evaluate(model, valid_dataloader, criterion)
        print(f"Epoch {epoch + 1:>2}/{num_epochs}", end=" ")
        print(f"train_loss {train_loss:.4f}", end=" ")
        print(f"val_loss {val_loss:.4f}")


def tokenizer(text):
    tagger = MeCab.Tagger("-Owakati")
    text = tagger.parse(text)
    return text.split()


def train(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0

    for text, label in dataloader:
        output = model(text)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for text, label in dataloader:
            output = model(text)

            loss = criterion(output, label)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


if __name__ == "__main__":
    main()

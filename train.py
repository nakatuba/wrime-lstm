import MeCab
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchtext.vocab import build_vocab_from_iterator

import wandb
from model import LSTM
from utils.dataset import TabularDataset
from utils.sampler import BalancedBatchSampler


def main():
    wandb.init(project="sentiment-analysis")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TabularDataset("./data/train.tsv")
    # valid_dataset = TabularDataset("./data/valid.tsv")
    test_dataset = TabularDataset("./data/test.tsv")

    vocab = build_vocab_from_iterator(
        [tokenizer(text) for text in train_dataset.texts],
        min_freq=10,
        specials=["<unk>", "<pad>"],
    )
    vocab.set_default_index(vocab["<unk>"])

    def collate_batch(batch):
        text_list = [torch.tensor(vocab(tokenizer(text))) for text, _ in batch]
        label_list = [label for _, label in batch]
        text_list = pad_sequence(text_list, padding_value=vocab["<pad>"])
        label_list = torch.tensor(label_list)
        return text_list.to(device), label_list.to(device)

    batch_size = 32

    # weights = [1 / (train_dataset.labels == label).sum() for label in [0, 1, 2, 3]]
    # weights = [weights[label] for label in train_dataset.labels]

    # sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
    batch_sampler = BalancedBatchSampler(train_dataset, batch_size=batch_size)

    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_batch
    # )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, collate_fn=collate_batch
    )
    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    # )
    # valid_dataloader = DataLoader(
    #     valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    # )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    model = LSTM(len(vocab), 32, 32, 4).to(device)
    # counts = [(train_dataset.labels == label).sum() for label in [0, 1, 2, 3]]
    # weight = 1 / torch.tensor(counts).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, _ = train(model, train_dataloader, criterion, optimizer)
        # valid_loss, valid_acc, _ = evaluate(model, valid_dataloader, criterion)
        print(f"Epoch {epoch}/{num_epochs}", end=" ")
        print(f"| train | Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}")
        # print("F1 score:", " ".join([f"{score:.4f}" for score in train_f1]), end=" ")
        # print(f"| valid | Loss: {valid_loss:.4f} Accuracy: {valid_acc:.4f}")
        # print("F1 score:", " ".join([f"{score:.4f}" for score in valid_f1]))
        # wandb.log(
        #     {
        #         "epoch": epoch,
        #         "train_loss": train_loss,
        #         "train_acc": train_acc,
        #         "valid_loss": valid_loss,
        #         "valid_acc": valid_acc,
        #     }
        # )

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for text, label in test_dataloader:
            output = model(text)
            y_true += label.tolist()
            y_pred += output.argmax(dim=1).tolist()

    target_names = ["class 0", "class 1", "class 2", "class 3"]
    print(classification_report(y_true, y_pred, target_names=target_names))

    df = pd.DataFrame(
        {"Sentence": test_dataset.texts, "Label": y_true, "Predicted": y_pred}
    )
    df.to_csv("./data/result.tsv", sep="\t", index=False)


def tokenizer(text):
    tagger = MeCab.Tagger("-Owakati")
    text = tagger.parse(text)
    return text.split()


def train(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    y_true = []
    y_pred = []

    for text, label in dataloader:
        output = model(text)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        y_true += label.tolist()
        y_pred += output.argmax(dim=1).tolist()

    epoch_loss /= len(dataloader)
    epoch_acc = accuracy_score(y_true, y_pred)
    epoch_f1 = f1_score(y_true, y_pred, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for text, label in dataloader:
            output = model(text)

            loss = criterion(output, label)

            epoch_loss += loss.item()

            y_true += label.tolist()
            y_pred += output.argmax(dim=1).tolist()

    epoch_loss /= len(dataloader)
    epoch_acc = accuracy_score(y_true, y_pred)
    epoch_f1 = f1_score(y_true, y_pred, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


if __name__ == "__main__":
    main()

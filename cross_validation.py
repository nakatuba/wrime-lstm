import random

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchtext.vocab import Vectors, build_vocab_from_iterator

from model import LSTM
from utils.dataset import TabularDataset
from utils.tokenizer import MeCabTokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TabularDataset("./data/joy.tsv")

    tokenizer = MeCabTokenizer()

    skf = StratifiedKFold(n_splits=5)
    precisions = []
    recalls = []
    fscores = []

    for fold, (train_index, test_index) in enumerate(
        skf.split(dataset.texts, dataset.labels)
    ):
        print("-" * 20, "Fold", fold + 1, "-" * 20)

        test_index_sampled = [
            index for index in test_index if dataset.labels[index] == 3
        ]
        for label in [0, 1, 2]:
            test_index_sampled += random.sample(
                [index for index in test_index if dataset.labels[index] == label],
                500,
            )
        test_index = test_index_sampled

        vocab = build_vocab_from_iterator(
            [tokenizer.tokenize(text) for text in dataset.texts[train_index]],
            min_freq=10,
            specials=["<unk>", "<pad>"],
        )
        vocab.set_default_index(vocab["<unk>"])

        def collate_batch(batch):
            text_list = [
                torch.tensor(vocab(tokenizer.tokenize(text))) for text, _ in batch
            ]
            label_list = [label for _, label in batch]
            text_list = pad_sequence(text_list, padding_value=vocab["<pad>"])
            label_list = torch.tensor(label_list)
            return text_list.to(device), label_list.to(device)

        batch_size = 64

        weights = [
            1 / (dataset.labels[train_index] == label).sum()
            for label in dataset.labels[train_index]
        ]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))

        train_dataloader = DataLoader(
            Subset(dataset, train_index),
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_batch,
        )
        test_dataloader = DataLoader(
            Subset(dataset, test_index),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

        japanese_word2vec_vectors = Vectors(name="./data/japanese_word2vec_vectors.vec")
        vectors = japanese_word2vec_vectors.get_vecs_by_tokens(vocab.get_itos())

        model = LSTM(vectors, 4).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=2e-3)

        num_epochs = 15
        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
            print(f"Epoch {epoch + 1}/{num_epochs}", end=" ")
            print(f"| train | Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}")

        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for text, label in test_dataloader:
                output = model(text)
                y_true += label.tolist()
                y_pred += output.argmax(dim=1).tolist()

        print(classification_report(y_true, y_pred))
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

    print("-" * 50)
    print("precision", sum(precisions) / len(precisions))
    print("recall", sum(recalls) / len(recalls))
    print("fscore", sum(fscores) / len(fscores))


def train(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for text, label in dataloader:
        output = model(text)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        acc = (pred == label).sum() / len(pred)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


if __name__ == "__main__":
    main()

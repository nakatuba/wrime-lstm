import MeCab
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from torchtext.legacy import data

from model import MultiTaskLSTM


def tokenizer(text):
    tagger = MeCab.Tagger("-Owakati")
    text = tagger.parse(text)
    return text.split()


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        output = model(batch.text)
        labels = [
            batch.joy,
            batch.sadness,
            batch.anticipation,
            batch.surprise,
            batch.anger,
            batch.fear,
            batch.disgust,
            batch.trust,
        ]

        loss = 0
        for i, label in enumerate(labels):
            loss += criterion(output[i], label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            output = model(batch.text)
            labels = [
                batch.joy,
                batch.sadness,
                batch.anticipation,
                batch.surprise,
                batch.anger,
                batch.fear,
                batch.disgust,
                batch.trust,
            ]

            loss = 0
            for i, label in enumerate(labels):
                loss += criterion(output[i], label)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


sentiments = [
    "joy",
    "sadness",
    "anticipation",
    "surprise",
    "anger",
    "fear",
    "disgust",
    "trust",
]

TEXT = data.Field(sequential=True, use_vocab=True, tokenize=tokenizer)
LABEL = data.Field(sequential=False, use_vocab=False)

train_dataset, val_dataset, test_dataset = data.TabularDataset.splits(
    path="./data",
    train="train.tsv",
    validation="val.tsv",
    test="test.tsv",
    format="tsv",
    fields=[("text", TEXT)] + [(sentiment, LABEL) for sentiment in sentiments],
    skip_header=True,
)

TEXT.build_vocab(train_dataset, min_freq=1)

batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter = data.Iterator(
    train_dataset, batch_size=batch_size, device=device, train=True
)

val_iter = data.Iterator(
    val_dataset, batch_size=batch_size, device=device, train=False, sort=False
)

test_iter = data.Iterator(
    test_dataset, batch_size=batch_size, device=device, train=False, sort=False
)

model = MultiTaskLSTM(len(TEXT.vocab), 32, 32, 4, len(sentiments)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_iter, optimizer, criterion)
    val_loss = evaluate(model, val_iter, criterion)
    print(f"Epoch {epoch + 1:>2}/{num_epochs}", end=" ")
    print(f"train_loss {train_loss:.4f}", end=" ")
    print(f"val_loss {val_loss:.4f}")

model.eval()
intensity_true = {sentiment: [] for sentiment in sentiments}
intensity_pred = {sentiment: [] for sentiment in sentiments}

with torch.no_grad():
    for batch in test_iter:
        output = model(batch.text)
        labels = [
            batch.joy,
            batch.sadness,
            batch.anticipation,
            batch.surprise,
            batch.anger,
            batch.fear,
            batch.disgust,
            batch.trust,
        ]

        for i, sentiment in enumerate(sentiments):
            intensity_true[sentiment] += labels[i].cpu()
            intensity_pred[sentiment] += torch.argmax(output[i], dim=1).cpu()

print("-" * 10, "MAE", "-" * 10)
for sentiment in sentiments:
    mae = mean_absolute_error(intensity_true[sentiment], intensity_pred[sentiment])
    print(sentiment, mae)

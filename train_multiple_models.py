import MeCab
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from torchtext.legacy import data

from model import LSTM


def tokenizer(text):
    tagger = MeCab.Tagger("-Owakati")
    text = tagger.parse(text)
    return text.split()


def train(model, iterator, optimizer, criterion, sentiment):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        output = model(batch.text)
        label_dict = {
            "joy": batch.joy,
            "sadness": batch.sadness,
            "anticipation": batch.anticipation,
            "surprise": batch.surprise,
            "anger": batch.anger,
            "fear": batch.fear,
            "disgust": batch.disgust,
            "trust": batch.trust,
        }

        loss = criterion(output, label_dict[sentiment])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, sentiment):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            output = model(batch.text)
            label_dict = {
                "joy": batch.joy,
                "sadness": batch.sadness,
                "anticipation": batch.anticipation,
                "surprise": batch.surprise,
                "anger": batch.anger,
                "fear": batch.fear,
                "disgust": batch.disgust,
                "trust": batch.trust,
            }

            loss = criterion(output, label_dict[sentiment])

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

model_dict = {}
criterion = nn.CrossEntropyLoss()
optimizer_dict = {}
for sentiment in sentiments:
    model = LSTM(len(TEXT.vocab), 32, 32, 4).to(device)
    model_dict[sentiment] = model
    optimizer_dict[sentiment] = optim.Adam(model.parameters(), lr=2e-5)

num_epochs = 10
for sentiment in sentiments:
    for epoch in range(num_epochs):
        model = model_dict[sentiment]
        optimizer = optimizer_dict[sentiment]
        train_loss = train(model, train_iter, optimizer, criterion, sentiment)
        val_loss = evaluate(model, val_iter, criterion, sentiment)
        print(f"Epoch {epoch + 1:>2}/{num_epochs}", end=" ")
        print(f"train_loss {train_loss:.4f}", end=" ")
        print(f"val_loss {val_loss:.4f}")

for sentiment in sentiments:
    model = model_dict[sentiment]
    model.eval()
    label_true = []
    label_pred = []

    with torch.no_grad():
        for batch in test_iter:
            output = model(batch.text)
            label_dict = {
                "joy": batch.joy,
                "sadness": batch.sadness,
                "anticipation": batch.anticipation,
                "surprise": batch.surprise,
                "anger": batch.anger,
                "fear": batch.fear,
                "disgust": batch.disgust,
                "trust": batch.trust,
            }

            label_true += label_dict[sentiment].cpu()
            label_pred += torch.argmax(output, dim=1).cpu()

    print(sentiment, mean_absolute_error(label_true, label_pred))

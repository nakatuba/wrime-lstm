import pandas as pd
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, sep="\t")
        self.texts = df["Sentence"].values
        self.labels = df["Label"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

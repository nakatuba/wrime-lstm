import pandas as pd
from torch.utils.data import Dataset


class DifferenceDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, sep="\t")
        df = df[["Sentence", "Writer_Joy", "Reader1_Joy", "Reader2_Joy", "Reader3_Joy"]]

        mean = df[["Reader1_Joy", "Reader2_Joy", "Reader3_Joy"]].mean(axis=1)
        std = df[["Reader1_Joy", "Reader2_Joy", "Reader3_Joy"]].std(axis=1)
        diff = df["Writer_Joy"] - mean

        df["Difference"] = 0
        df.loc[std < 1, "Difference"] = 1
        df.loc[(std < 1) & (diff > 1), "Difference"] = 2
        df.loc[(std < 1) & (diff < -1), "Difference"] = 3

        self.texts = df["Sentence"].values
        self.labels = df["Difference"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

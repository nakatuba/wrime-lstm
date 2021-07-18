import numpy as np
from torch.utils.data import BatchSampler


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.indices_list = [
            np.where(dataset.labels == label)[0] for label in [0, 1, 2, 3]
        ]
        # self.major_indices = np.where(dataset.labels != 3)[0]
        # self.minor_indices = np.where(dataset.labels == 3)[0]
        # self.major_dataset = dataset[(dataset.labels == 0) | (dataset.labels == 1)]
        self.batch_size = batch_size

    def __iter__(self):
        indices = []
        # for idx in self.major_indices:
        np.random.shuffle(self.indices_list[1])
        for idx in self.indices_list[1]:
            indices.append(idx)
            if len(indices) == self.batch_size / 4:
                for label in [0, 2, 3]:
                    indices += np.random.choice(
                        self.indices_list[label],
                        int(self.batch_size / 4),
                        replace=False,
                    ).tolist()
                yield indices
                indices = []
        if len(indices) > 0:
            yield indices

    def __len__(self):
        return len(self.indices_list[1]) // int(self.batch_size / 4)

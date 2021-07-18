import numpy as np
from torch.utils.data import BatchSampler


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.indices_list = [
            np.where(dataset.labels == label)[0] for label in [0, 1, 2, 3]
        ]
        self.sample_size = batch_size // 4

    def __iter__(self):
        indices = []
        np.random.shuffle(self.indices_list[1])
        for idx in self.indices_list[1]:
            indices.append(idx)
            if len(indices) == self.sample_size:
                for label in [0, 2, 3]:
                    indices += np.random.choice(
                        self.indices_list[label],
                        self.sample_size,
                        replace=False,
                    ).tolist()
                yield indices
                indices = []
        if len(indices) > 0:
            yield indices

    def __len__(self):
        return len(self.indices_list[1]) // self.sample_size

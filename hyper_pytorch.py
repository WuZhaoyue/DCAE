import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class HyperData(Dataset):
    def __init__(self, dataset):
        self.data1 = dataset.astype(np.float32)
        # self.data2 = dataset[1].astype(np.float32)
        # self.labels = dataset[2].astype(np.float32)

    def __getitem__(self, index):
        img1 = torch.from_numpy(np.asarray(self.data1[index,:,:,:]))
        # img2 = torch.from_numpy(np.asarray(self.data2[index,:,:,:]))
        # label = torch.from_numpy(np.asarray(self.labels[index,:]))
        return img1

    def __len__(self):
        return len(self.data1)

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from util.util import oneHot, standardize, normalized


class Dataset(Dataset):
    def __init__(self, path, n_class, random_state=3407):
        Data = pd.read_csv(path + "/data.csv")
        Label = pd.read_csv(path + "/labels.csv")
        Data = np.array(Data)
        Label = np.array(Label).reshape(-1)
        # 划分数据集
        self.data, self.testData, self.label, self.testLabel = \
            train_test_split(Data, Label, test_size=0.8, random_state=1, shuffle=True)

        # one-hot编码
        self.label_not_onehot = self.label
        self.testLabel_not_onehot = self.testLabel
        self.label = oneHot(self.label, n_class)
        self.testLabel = oneHot(self.testLabel, n_class)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Data/processed_data
if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), "../Data/processed_data"))
    dataset = Dataset(path, 15)
    print(dataset.data.shape)
    print(dataset.label.shape)
    print(dataset.testData.shape)
    print(dataset.testLabel.shape)
    pass

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat


class Trainset(Dataset):
    def __init__(self, subject_name):
        if subject_name == 'A':
            raw_data = loadmat('dataset/sub_a.mat')
        else:
            raw_data = loadmat('dataset/sub_b.mat')

        signals = raw_data['responses']
        label = raw_data['is_stimulate']
        data = []
        target = []
        for i in range(12):
            for j in range(85):
                if label[i, j] == 1:
                    data.append(signals[i, :, :, j].reshape(-1, 64))
                    target.append(label[i, j])
                    data.append(signals[i, :, :, j].reshape(-1, 64))
                    target.append(label[i, j])
                    data.append(signals[i, :, :, j].reshape(-1, 64))
                    target.append(label[i, j])
                    data.append(signals[i, :, :, j].reshape(-1, 64))
                    target.append(label[i, j])
                data.append(signals[i, :, :, j].reshape(-1, 64))
                target.append(label[i, j])

        self.data = np.array(data)
        self.target = np.array(target)

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        return self.data[index, :, :].astype(np.float32).T, self.target[index].astype(np.float32)


class Testset(Dataset):
    def __init__(self, subject_name):
        if subject_name == 'A':
            raw_data = loadmat('dataset/sub_a_test.mat')
        else:
            raw_data = loadmat('dataset/sub_b_test.mat')
        self.signals = raw_data['responses']

    def __len__(self):
        return self.signals.shape[-1]

    def __getitem__(self, index):
        col = self.signals[:6, :, :, index].astype(np.float32).transpose([0, 2, 1])
        row = self.signals[6:, :, :, index].astype(np.float32).transpose([0, 2, 1])
        return col, row


"""
-*- coding = utf-8 -*-
@File : dataset.py
@Software : vscode
"""
import torch
from torch.utils.data import Dataset


class SSLECGTextDataset(Dataset):
    """
    Description:
        A dataset class that returns ECG and text data for training.

    Args:
        num_samples: The number of samples in the dataset.
        ecg_length: The length of the ECG signal.
    """
    def __init__(self, num_samples, ecg_length):
        super(SSLECGTextDataset, self).__init__()
        self.num_samples = num_samples
        self.ecg_length = ecg_length
        self.ecg_data = torch.randn(num_samples, 16, ecg_length)
        self.text_data = ["This is a generated clinical report for sample " + str(i) for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.ecg_data[idx], self.text_data[idx]


class ZeroShotTestECGTextDataset(Dataset):
    """
    Description:
        A dataset class that returns ECG and text data for zero-shot testing.

    Args:
        num_samples: The number of samples in the dataset.
        ecg_length: The length of the ECG signal.
    """
    def __init__(self, num_samples, ecg_length):
        super(ZeroShotTestECGTextDataset, self).__init__()
        self.num_samples = num_samples
        self.ecg_length = ecg_length
        self.ecg_data = torch.randn(num_samples, 16, ecg_length)
        self.categories = ["Normal ECG", "Myocardial Infarction", "ST/T change", "Hypertrophy", "Conducion Disturbance"]
        self.label = torch.zeros((num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.ecg_data[idx], self.label[idx]

    def load_categories(self):
        pass

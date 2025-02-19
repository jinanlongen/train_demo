
import struct
from tkinter import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class MNISTDataset():
    """
    MNIST 数据集
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def train_data(self):
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        return train_data
    
    def test_data(self):
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        return test_data
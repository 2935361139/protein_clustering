#dataset.py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms

class ProteinDataset(Dataset):
    def __init__(self, directory_path, target_size=(256, 256)):
        self.files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.npy')]
        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
        self.target_size = target_size
        self.data = [self.normalize_and_resize(np.load(f)) for f in self.files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        matrix = self.data[idx]
        matrix = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # 添加一个通道维度
        return matrix, 0  # 返回数据及其标签（这里简化处理，用0代表所有标签）

    def normalize_and_resize(self, data):
        """将数据归一化到0-1范围，并调整到目标尺寸"""
        data_min = data.min()
        data_max = data.max()
        normalized_data = (data - data_min) / (data_max - data_min)
        resized_data = self.resize_matrix(normalized_data, self.target_size)
        return resized_data

    def resize_matrix(self, matrix, target_size):
        """调整矩阵大小到目标尺寸"""
        target_matrix = np.zeros(target_size)
        min_rows = min(matrix.shape[0], target_size[0])
        min_cols = min(matrix.shape[1], target_size[1])
        target_matrix[:min_rows, :min_cols] = matrix[:min_rows, :min_cols]
        return target_matrix

    def get_filenames(self):
        """返回文件名列表"""
        return self.files

def load_datasets(directory_path, target_size=(256, 256), train_ratio=0.8):
    full_dataset = ProteinDataset(directory_path, target_size=target_size)
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset, full_dataset

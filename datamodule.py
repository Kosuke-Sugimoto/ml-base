import torch
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return None
    
    def __getitem__(self, idx):
        return None


class BaseDataModule(object):
    def __init__(self):
        self.train_paths = []
        self.val_paths = []
        self.test_paths = []

        self.train_dataset = BaseDataset()
        self.val_dataset = BaseDataset()
        self.test_dataset = BaseDataset()
    
    def get_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=2)
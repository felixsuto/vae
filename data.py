import os
from typing import Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torchvision as tv
import pytorch_lightning as pl
import random

from torchvision import transforms as tr

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, file_list, transform):
        self.file_list = file_list
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img = tv.io.read_image(os.path.join(self.root, self.file_list[index]))
        if self.transform is not None:
            img = self.transform(img)
        return img
    
class DataModule(pl.LightningDataModule):
    
    def __init__(self, root, patch_size, batch_size, num_workers, pin_memory):
        super().__init__()
        self.root = root
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self):
        self.train_loader, self.val_loader, self.test_loader = prepare_dataloaders(root=self.root,
                                                                         patch_size=self.patch_size,
                                                                         batch_size=self.batch_size,
                                                                         num_workers=self.num_workers,
                                                                         pin_memory=self.pin_memory)
    
    def train_dataloader(self):
        return self.train_loaer
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader
         
def prepare_dataloaders(root, patch_size, batch_size, num_workers, pin_memory):

    train_transforms = tr.Compose([tr.RandomHorizontalFlip(),
                                   tr.CenterCrop(148),
                                   tr.Resize(patch_size),
                                   tr.ToTensor()])
    
    val_transforms = tr.Compose([tr.CenterCrop(148),
                                 tr.Resize(patch_size),
                                 tr.ToTensor()])
    
    test_transforms = tr.Compose([tr.CenterCrop(148),
                                 tr.Resize(patch_size),
                                 tr.ToTensor()])
    
    file_list = os.listdir(root)
    random.shuffle(file_list)

    train_size = int(0.7 * len(file_list))
    val_size = int(0.15 * len(file_list))
    test_size = len(file_list) - train_size - val_size
    
    train_files, val_files, test_files = torch.utils.data.random_split(file_list, [train_size, val_size, test_size])

    train_dataset =  UnlabeledDataset(root, train_files, train_transforms)
    val_dataset =  UnlabeledDataset(root, val_files, val_transforms)
    test_dataset =  UnlabeledDataset(root, test_files, test_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory,
                                               drop_last=True,
                                               shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory,
                                               drop_last=False,
                                               shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory,
                                               drop_last=False,
                                               shuffle=False)
    return train_loader, val_loader, test_loader

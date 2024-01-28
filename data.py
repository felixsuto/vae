import os
from typing import Optional, Sequence, Union, List
import torch
import torchvision as tv
import pytorch_lightning as pl

from torchvision import transforms as tr
from torch.utils.data import DataLoader

class UnlabeledDataset(tv.datasets.CelebA):
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
         
def prepare_dataloaders(root, patch_size, batch_size, num_workers, pin_memory):

    train_transforms = tr.Compose([tr.RandomHorizontalFlip(),
                                   tr.CenterCrop(148),
                                   tr.Resize(patch_size)])
    
    val_transforms = tr.Compose([tr.CenterCrop(148),
                                 tr.Resize(patch_size)])
    
    test_transforms = tr.Compose([tr.CenterCrop(148),
                                 tr.Resize(patch_size)])
    
    file_list = os.listdir(root)
    dataset_bounds = [162771, 182638]

    train_dataset =  UnlabeledDataset(root, file_list[:dataset_bounds[0]], train_transforms)
    val_dataset =  UnlabeledDataset(root, file_list[dataset_bounds[0]: dataset_bounds[1]], val_transforms)
    test_dataset =  UnlabeledDataset(root, file_list[dataset_bounds[1]:], test_transforms)
    
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

class CelebAWrapper(tv.datasets.CelebA):
    def _check_integrity(self) -> bool:
        return True
    
class CelebAModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = tr.Compose([tr.RandomHorizontalFlip(),
                                       tr.CenterCrop(148),
                                       tr.Resize(self.patch_size),
                                       tr.ToTensor(),
                                       tr.Normalize(mean=[0.5063, 0.4258, 0.3832], std=[0.2661, 0.2452, 0.2414]),
                                       ])
            
        val_transforms = tr.Compose([tr.CenterCrop(148),
                                     tr.Resize(self.patch_size),
                                     tr.ToTensor(),
                                     tr.Normalize(mean=[0.5063, 0.4258, 0.3832], std=[0.2661, 0.2452, 0.2414]),
                                     ])
            
        test_transforms = tr.Compose([tr.CenterCrop(148),
                                      tr.Resize(self.patch_size),
                                      tr.ToTensor(),
                                      tr.Normalize(mean=[0.5063, 0.4258, 0.3832], std=[0.2661, 0.2452, 0.2414]),
                                      ])
            
        self.train_dataset = CelebAWrapper(self.data_dir,
                                           split='train',
                                           transform=train_transforms,
                                           download=False)
        
        self.val_dataset = CelebAWrapper(self.data_dir,
                                         split='valid',
                                         transform=val_transforms,
                                         download=False)
        
        self.test_dataset = CelebAWrapper(self.data_dir,
                                         split='test',
                                         transform=test_transforms,
                                         download=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
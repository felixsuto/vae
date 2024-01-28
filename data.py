import os
import torch
import torchvision as tv
import torchvision.transforms as tr

class UnlabeledDataset(tv.datasets.CelebA):
    def _check_integrity(self) -> bool:
        return True

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

    train_dataset =  UnlabeledDataset(root, split='train', transform=train_transforms, download=False)
    val_dataset = UnlabeledDataset(root, split='valid', transform=train_transforms, download=False)
    test_dataset = UnlabeledDataset(root, split='test', transform=train_transforms, download=False)
    
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
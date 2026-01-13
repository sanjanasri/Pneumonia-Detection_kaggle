# src/dataset.py
import os
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tf, eval_tf


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Expects:
    data_dir/
      train/class_x/...
      val/class_x/...
      test/class_x/...
    """
    train_tf, eval_tf = get_transforms(img_size)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)

    class_to_idx = train_ds.class_to_idx

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_to_idx


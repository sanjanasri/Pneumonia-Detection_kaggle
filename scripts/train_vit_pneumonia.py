import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_dataloaders(data_dir, batch_size=32):
    """
    data_dir should contain train/, val/, test/ subfolders.
    """

    # ImageNet stats (because we use ImageNet-pretrained ViT)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # X-rays are grayscale
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    test_dir  = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_test_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    class_names = train_dataset.classes  # ['NORMAL', 'PNEUMONIA'] usually

    return train_loader, val_loader, test_loader, class_names


def build_vit_model(num_classes=2):
    """
    Build a Vision Transformer using torchvision's ViT_B_16 pretrained on ImageNet.
    """
    # vit_b_16 available in newer torchvision versions; fall back to timm if needed.
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)

    # Replace classification head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)      # shape (B, 2)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def train_vit_pneumonia(
    data_dir,
    num_epochs=15,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-4,
    save_path="vit_pneumonia_best.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir, batch_size=batch_size
    )
    print("Classes:", class_names)

    model = build_vit_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"  --> New best model saved to {save_path}")

    print("\nTraining complete.")
    print(f"Best val acc: {best_val_acc:.4f}")

    # Load best weights and evaluate on test set
    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    return model


if __name__ == "__main__":
    # In Kaggle, something like:
    # data_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray"
    data_dir = "/home/sanjana/pytorch-xray-pneumonia/data/chest_xray"
    train_vit_pneumonia(data_dir)


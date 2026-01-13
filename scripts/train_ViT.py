#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 16:57:01 2025

@author: sanjana
"""

import os
import time
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----------------------------
# Optional: Optuna
# ----------------------------
try:
    import optuna
except ImportError as e:
    optuna = None


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (slower). If you prefer speed, you can disable these.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    data_dir should contain train/, val/, test/ subfolders.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
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

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names


def build_vit_model(num_classes=2, head_dropout=0.0):
    """
    Build a Vision Transformer using torchvision's ViT_B_16 pretrained on ImageNet.
    Adds optional dropout before the final head.
    """
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)

    in_features = model.heads.head.in_features
    if head_dropout and head_dropout > 0:
        model.heads.head = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.heads.head = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone_except_head(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze head
    for p in model.heads.parameters():
        p.requires_grad = True


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_corrects / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_corrects / max(total, 1)
    return epoch_loss, epoch_acc


def train_vit_pneumonia(
    data_dir,
    num_epochs=15,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-4,
    head_dropout=0.0,
    freeze_backbone=False,
    save_path="vit_pneumonia_best.pt",
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )
    print("Classes:", class_names)

    model = build_vit_model(num_classes=len(class_names), head_dropout=head_dropout)

    if freeze_backbone:
        freeze_backbone_except_head(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"  --> New best model saved to {save_path}")

    print("\nTraining complete.")
    print(f"Best val acc: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    return model, best_val_acc, test_acc


def tune_with_optuna(
    data_dir: str,
    n_trials: int = 20,
    tune_epochs: int = 5,          # fewer epochs per trial for speed
    num_workers: int = 4,
    seed: int = 42,
):
    if optuna is None:
        raise ImportError("Optuna is not installed. Install with: pip install optuna")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Optuna tuning on device:", device)

    # Pruner helps stop bad trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    def objective(trial: "optuna.Trial"):
        # --- Search space ---
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        head_dropout = trial.suggest_float("head_dropout", 0.0, 0.5)
        freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])

        # Data
        train_loader, val_loader, _, class_names = get_dataloaders(
            data_dir, batch_size=batch_size, num_workers=num_workers
        )

        # Model
        model = build_vit_model(num_classes=len(class_names), head_dropout=head_dropout)
        if freeze_backbone:
            freeze_backbone_except_head(model)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tune_epochs)

        best_val_acc = 0.0

        for epoch in range(tune_epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            _, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            best_val_acc = max(best_val_acc, val_acc)

            # Report intermediate objective value for pruning
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                # Free GPU memory before pruning exception
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned()

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_val_acc

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n==== Optuna Best Trial ====")
    print("Best val acc:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study


if __name__ == "__main__":
    # data_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray"
    data_dir = "/home/sanjana/pytorch-xray-pneumonia/data/chest_xray"

    # 1) Tune hyperparameters
    study = tune_with_optuna(
        data_dir=data_dir,
        n_trials=20,
        tune_epochs=5,
        num_workers=4,
        seed=42,
    )

    # 2) Train final model with the best hyperparameters for longer
    best = study.best_params
    final_save_path = "vit_pneumonia_best_optuna.pt"

    model, best_val_acc, test_acc = train_vit_pneumonia(
        data_dir=data_dir,
        num_epochs=15,  # full training
        batch_size=best["batch_size"],
        lr=best["lr"],
        weight_decay=best["weight_decay"],
        head_dropout=best["head_dropout"],
        freeze_backbone=best["freeze_backbone"],
        save_path=final_save_path,
        num_workers=4,
    )

    print("\n==== Final Results ====")
    print("Final best val acc:", best_val_acc)
    print("Final test acc:", test_acc)
    print("Saved model:", final_save_path)

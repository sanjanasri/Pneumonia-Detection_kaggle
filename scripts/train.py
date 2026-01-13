# src/train.py
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score

from dataset import get_dataloaders
from models import XRayResNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # (B, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1).detach().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    softmax = nn.Softmax(dim=1)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        probs = softmax(outputs)[:, 1].detach().cpu()  # prob of class 1
        preds = outputs.argmax(dim=1).detach().cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # e.g. if only one class present

    return avg_loss, acc, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, class_to_idx = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_to_idx)

    model = XRayResNet(num_classes=num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_auc = -1.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, AUC: {val_auc:.4f}"
        )

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_to_idx": class_to_idx,
                },
                best_ckpt_path,
            )
            print(f"  → Saved new best model to {best_ckpt_path} (AUC={val_auc:.4f})")


if __name__ == "__main__":
    main()


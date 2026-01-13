import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# -------------------- model --------------------
def build_vit_model(num_classes=2):
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# -------------------- data --------------------
def get_test_loader(data_dir, batch_size=32, num_workers=4):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    test_dir = os.path.join(data_dir, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader, test_dataset.classes


# -------------------- prediction collection --------------------
@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred


# -------------------- metrics --------------------
def class_metrics_from_preds(y_true, y_pred, c):
    # one-vs-rest counts for class c
    tp = int(np.sum((y_true == c) & (y_pred == c)))
    fp = int(np.sum((y_true != c) & (y_pred == c)))
    fn = int(np.sum((y_true == c) & (y_pred != c)))
    tn = int(len(y_true) - tp - fp - fn)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "tn": tn, "tp": tp, "fn": fn, "fp": fp,
        "precision": prec, "recall": rec, "f1": f1,
        "support": int(np.sum(y_true == c)),
    }


def bootstrap_se(y_true, y_pred, num_classes, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # store bootstrap estimates per class
    boot = {
        c: {k: [] for k in ["tn","tp","fn","fp","precision","recall","f1","support"]}
        for c in range(num_classes)
    }
    acc_list = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        yt = y_true[idx]
        yp = y_pred[idx]

        acc_list.append(float(np.mean(yt == yp)))

        for c in range(num_classes):
            m = class_metrics_from_preds(yt, yp, c)
            for k, v in m.items():
                boot[c][k].append(float(v))

    # std of bootstrap distribution = bootstrap SE
    se = {}
    for c in range(num_classes):
        se[c] = {f"{k}_se": float(np.std(v, ddof=1)) for k, v in boot[c].items()}

    acc_se = float(np.std(acc_list, ddof=1))
    acc_mean = float(np.mean(y_true == y_pred))
    return se, acc_mean, acc_se


def evaluate_saved_model_with_bootstrap(
    data_dir,
    weights_path="vit_pneumonia_best.pt",
    batch_size=32,
    n_boot=1000,
    seed=42,
    out_csv="test_metrics_bootstrap.csv",
    out_csv_overall="test_overall_accuracy_bootstrap.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader, class_names = get_test_loader(data_dir, batch_size=batch_size)
    print("Classes:", class_names)

    # build + load
    model = build_vit_model(num_classes=len(class_names)).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from: {weights_path}")

    # predictions
    y_true, y_pred = collect_predictions(model, test_loader, device)

    # point estimates
    rows = []
    for c, name in enumerate(class_names):
        m = class_metrics_from_preds(y_true, y_pred, c)
        m["class"] = name
        rows.append(m)

    df = pd.DataFrame(rows)[
        ["class","support","precision","recall","f1","tn","tp","fn","fp"]
    ]

    # bootstrap SEs
    se_dict, acc, acc_se = bootstrap_se(
        y_true, y_pred, num_classes=len(class_names), n_boot=n_boot, seed=seed
    )

    # attach SE columns
    for c, name in enumerate(class_names):
        for k, v in se_dict[c].items():
            base = k.replace("_se", "")
            # ensure matching column order: metric + metric_se
            df.loc[df["class"] == name, k] = v

    # save classwise metrics
    df.to_csv(out_csv, index=False)
    print(f"Saved classwise metrics + bootstrap SE to: {out_csv}")

    # save overall accuracy + SE
    df_overall = pd.DataFrame([{
        "metric": "accuracy",
        "value": float(acc),
        "se": float(acc_se),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_test": int(len(y_true)),
    }])
    df_overall.to_csv(out_csv_overall, index=False)
    print(f"Saved overall accuracy + bootstrap SE to: {out_csv_overall}")

    # quick print
    print("\nClasswise results (point estimate + SE):")
    print(df)

    print(f"\nOverall accuracy: {acc:.4f} (SE={acc_se:.4f})")


if __name__ == "__main__":
    data_dir = "/home/sanjana/pneumonia detection/data/chest_xray"
    evaluate_saved_model_with_bootstrap(
        data_dir=data_dir,
        weights_path="vit_pneumonia_best.pt",
        batch_size=32,
        n_boot=1000,         # increase to 5000 for more stable SE (slower)
        seed=42,
        out_csv="test_metrics_bootstrap.csv",
        out_csv_overall="test_overall_accuracy_bootstrap.csv",
    )

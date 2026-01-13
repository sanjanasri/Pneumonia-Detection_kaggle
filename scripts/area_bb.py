#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 23:42:54 2025

@author: sanjana
"""

# ------------------ Batch: ViT pneumonia "affected area" for ALL test images ------------------
# Saves per-image:
#   overlay heatmap:  OUT_DIR/<true_class>/<name>__overlay.png
#   binary mask:      OUT_DIR/<true_class>/<name>__mask.png
#   bbox image:       OUT_DIR/<true_class>/<name>__bbox.png
# Also saves a CSV summary: OUT_DIR/test_localization_summary.csv
#
# Requirements: torch, torchvision, pillow, opencv-python, pandas, numpy
# ---------------------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets, models

# ------------------ CONFIG ------------------
DATA_DIR     = "/home/sanjana/pneumonia detection/data/chest_xray"   # contains train/val/test
SPLIT        = "test"                                                  # "test" / "val" / etc.
WEIGHTS_PATH = "vit_pneumonia_best.pt"
OUT_DIR      = "vit_localization_outputs"
THRESH       = 0.55                 # heatmap threshold (0..1). Try 0.45-0.65
ONLY_PRED_PNEUMONIA = False         # True => save outputs only when model predicts PNEUMONIA
MAX_IMAGES   = None                 # e.g., 50 for a quick run, or None for all
BATCH_LOG_EVERY = 50
# -------------------------------------------


def build_vit_model(num_classes=2):
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def get_class_names_from_train(data_dir):
    train_dir = os.path.join(data_dir, "train")
    if os.path.isdir(train_dir):
        return datasets.ImageFolder(train_dir).classes
    # fallback
    split_dir = os.path.join(data_dir, SPLIT)
    return datasets.ImageFolder(split_dir).classes


def get_transform():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


@torch.no_grad()
def predict_probs(model, x):
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = int(torch.argmax(probs).item())
    return probs.detach().cpu().numpy(), pred


def vit_patch_cam(model, x, target_class):
    """
    Patch-token CAM for ViT-B/16 (14x14 patches for 224x224).
    Returns heatmap (224,224) in [0,1].
    """
    model.eval()
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["tokens"] = out  # (B, tokens, dim)

    def bwd_hook(module, grad_in, grad_out):
        gradients["tokens"] = grad_out[0]  # (B, tokens, dim)

    h1 = model.encoder.layers[-1].register_forward_hook(fwd_hook)
    h2 = model.encoder.layers[-1].register_full_backward_hook(bwd_hook)

    logits = model(x)
    score = logits[:, target_class].sum()
    model.zero_grad(set_to_none=True)
    score.backward()

    h1.remove()
    h2.remove()

    A = activations["tokens"]     # (1, tokens, dim)
    dA = gradients["tokens"]      # (1, tokens, dim)

    # Remove CLS token at index 0
    A_p  = A[:, 1:, :]            # (1, 196, dim)
    dA_p = dA[:, 1:, :]           # (1, 196, dim)

    # Channel weights: mean gradient over patches
    w = dA_p.mean(dim=1)          # (1, dim)

    # Weighted sum over channels => CAM per patch
    cam = (A_p * w.unsqueeze(1)).sum(dim=2)  # (1, 196)
    cam = F.relu(cam)

    # reshape patches and upsample to 224x224
    cam_2d = cam.reshape(1, 1, 14, 14)
    cam_up = F.interpolate(cam_2d, size=(224, 224), mode="bilinear", align_corners=False)
    heatmap = cam_up.squeeze().detach().cpu().numpy()

    # normalize [0,1]
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


def heatmap_to_mask_and_bbox(heatmap, thresh=0.55):
    """
    heatmap: float (224,224) in [0,1]
    returns: mask uint8 (224,224) values 0/255, bbox (x,y,w,h) or None
    """
    hm = (heatmap * 255).astype(np.uint8)
    _, mask = cv2.threshold(hm, int(thresh * 255), 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask, None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return mask, (x, y, w, h)


def save_outputs(image_path, heatmap, mask, bbox, out_dir, true_class, base_name):
    os.makedirs(os.path.join(out_dir, true_class), exist_ok=True)

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_bgr = cv2.resize(img_bgr, (224, 224))

    hm_uint8 = (heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)

    bbox_img = img_bgr.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    overlay_path = os.path.join(out_dir, true_class, f"{base_name}__overlay.png")
    mask_path    = os.path.join(out_dir, true_class, f"{base_name}__mask.png")
    bbox_path    = os.path.join(out_dir, true_class, f"{base_name}__bbox.png")

    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(bbox_path, bbox_img)

    return overlay_path, mask_path, bbox_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    class_names = get_class_names_from_train(DATA_DIR)
    print("Class names (from train if present):", class_names)

    if "PNEUMONIA" in class_names:
        pneumonia_idx = class_names.index("PNEUMONIA")
    else:
        pneumonia_idx = 1 if len(class_names) > 1 else 0
        print("WARNING: 'PNEUMONIA' not found in class_names; using index:", pneumonia_idx)

    # Load model
    model = build_vit_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print("Loaded weights:", WEIGHTS_PATH)

    # Dataset for paths/labels (no transform here; we handle transform ourselves)
    split_dir = os.path.join(DATA_DIR, SPLIT)
    ds = datasets.ImageFolder(split_dir)  # transform=None
    print(f"{SPLIT} images:", len(ds.samples))

    tfm = get_transform()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    n = len(ds.samples) if MAX_IMAGES is None else min(MAX_IMAGES, len(ds.samples))

    for i in range(n):
        img_path, true_label = ds.samples[i]
        true_class = ds.classes[true_label]

        # base name: original filename without extension (plus index to avoid collisions)
        fname = os.path.basename(img_path)
        base = os.path.splitext(fname)[0]
        base_name = f"{i:06d}__{base}"

        # preprocess
        pil = Image.open(img_path).convert("RGB")
        x = tfm(pil).unsqueeze(0).to(device)

        # predict
        probs, pred_idx = predict_probs(model, x)
        pred_class = class_names[pred_idx]
        pneu_prob = float(probs[pneumonia_idx])

        if ONLY_PRED_PNEUMONIA and pred_idx != pneumonia_idx:
            # still record row, but skip saving images
            rows.append({
                "image_path": img_path,
                "true_class": true_class,
                "pred_class": pred_class,
                "pneumonia_prob": pneu_prob,
                "bbox_x": None, "bbox_y": None, "bbox_w": None, "bbox_h": None,
                "mask_area_frac": None,
                "saved_overlay": None, "saved_mask": None, "saved_bbox": None
            })
            continue

        # CAM heatmap for pneumonia class (localization)
        heatmap = vit_patch_cam(model, x, target_class=pneumonia_idx)

        # mask + bbox
        mask, bbox = heatmap_to_mask_and_bbox(heatmap, thresh=THRESH)
        mask_area_frac = float((mask > 0).sum() / (224 * 224))

        # save images
        overlay_path, mask_path, bbox_path = save_outputs(
            img_path, heatmap, mask, bbox, OUT_DIR, true_class, base_name
        )

        bx = by = bw = bh = None
        if bbox is not None:
            bx, by, bw, bh = bbox

        rows.append({
            "image_path": img_path,
            "true_class": true_class,
            "pred_class": pred_class,
            "pneumonia_prob": pneu_prob,
            "bbox_x": bx, "bbox_y": by, "bbox_w": bw, "bbox_h": bh,
            "mask_area_frac": mask_area_frac,
            "saved_overlay": overlay_path,
            "saved_mask": mask_path,
            "saved_bbox": bbox_path
        })

        if (i + 1) % BATCH_LOG_EVERY == 0 or (i + 1) == n:
            print(f"Processed {i+1}/{n}")

        # optional: free GPU memory a bit
        if device.type == "cuda" and (i + 1) % 200 == 0:
            torch.cuda.empty_cache()

    # Save CSV summary
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "test_localization_summary.csv")
    df.to_csv(csv_path, index=False)
    print("\nSaved summary CSV:", csv_path)
    print("Outputs saved under:", OUT_DIR)


if __name__ == "__main__":
    main()

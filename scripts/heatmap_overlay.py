#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 23:37:43 2025

@author: sanjana
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models

# --------- model (same as training) ----------
def build_vit_model(num_classes=2):
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

# --------- preprocessing (same as test) ----------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# --------- ViT patch Grad-CAM-ish heatmap ----------
def vit_patch_cam(model, x, target_class=None):
    """
    Returns a (224,224) heatmap in [0,1] for a single image tensor x of shape (1,3,224,224).
    Uses last encoder layer output as token activations and gradients.
    """
    model.eval()

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["tokens"] = out  # (B, tokens, dim)

    def bwd_hook(module, grad_in, grad_out):
        gradients["tokens"] = grad_out[0]  # (B, tokens, dim)

    # hook the LAST encoder layer
    h1 = model.encoder.layers[-1].register_forward_hook(fwd_hook)
    h2 = model.encoder.layers[-1].register_full_backward_hook(bwd_hook)

    logits = model(x)  # (1, num_classes)
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    score = logits[:, target_class].sum()
    model.zero_grad(set_to_none=True)
    score.backward()

    h1.remove()
    h2.remove()

    A = activations["tokens"]       # (1, tokens, dim)
    dA = gradients["tokens"]        # (1, tokens, dim)

    # drop CLS token at index 0 => use only patch tokens
    A_p = A[:, 1:, :]               # (1, num_patches, dim)
    dA_p = dA[:, 1:, :]             # (1, num_patches, dim)

    # weights per channel = mean grad across patches
    w = dA_p.mean(dim=1)            # (1, dim)

    # cam per patch = sum_c w_c * A_patch_c
    cam_patches = (A_p * w.unsqueeze(1)).sum(dim=2)  # (1, num_patches)
    cam_patches = F.relu(cam_patches)

    # ViT-B/16 on 224 => 14x14 patches
    cam_2d = cam_patches.reshape(1, 1, 14, 14)

    # upsample to 224x224
    cam_up = F.interpolate(cam_2d, size=(224, 224), mode="bilinear", align_corners=False)
    cam_up = cam_up.squeeze().detach().cpu().numpy()

    # normalize to [0,1]
    cam_up = cam_up - cam_up.min()
    cam_up = cam_up / (cam_up.max() + 1e-8)

    return cam_up, target_class

def show_overlay(img_path, heatmap, alpha=0.45):
    img = Image.open(img_path).convert("RGB").resize((224,224))
    img_np = np.array(img)

    plt.figure(figsize=(6,6))
    plt.imshow(img_np)
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.show()

# --------- example run ----------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # IMPORTANT: class order must match training ImageFolder order
    class_names = ["NORMAL", "PNEUMONIA"]  # adjust if your folder order differs

    model = build_vit_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("vit_pneumonia_best.pt", map_location=device))

    img_path = "/home/sanjana/pneumonia detection/data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"  # <- change this
    x = test_transform(Image.open(img_path)).unsqueeze(0).to(device)

    heatmap, cls = vit_patch_cam(model, x, target_class=class_names.index("PNEUMONIA"))
    print("Target class:", class_names[cls])

    show_overlay(img_path, heatmap)






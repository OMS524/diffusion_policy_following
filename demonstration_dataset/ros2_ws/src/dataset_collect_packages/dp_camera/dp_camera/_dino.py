#!/usr/bin/env python3
#
# Copyright 2025 AIRO LABS., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DINOv2 (timm) feature extractor for cosine-similarity keypoint matching."""

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torchvision import transforms


class DinoFeatureExtractor:
    """Wraps a DINOv2 ViT and produces L2-normalised CLS embeddings for image crops."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, device: str = "cuda", patch_size: int = 56,
                 model_name: str = "vit_large_patch14_dinov2.lvd142m"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patch_size = int(patch_size)

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        ).eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def _crop(self, image_bgr: np.ndarray, cx: int, cy: int) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        half = self.patch_size // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, x0 + self.patch_size)
        y1 = min(h, y0 + self.patch_size)
        x0 = max(0, x1 - self.patch_size)
        y0 = max(0, y1 - self.patch_size)
        crop = image_bgr[y0:y1, x0:x1]
        if crop.shape[0] != self.patch_size or crop.shape[1] != self.patch_size:
            crop = cv2.resize(crop, (self.patch_size, self.patch_size),
                              interpolation=cv2.INTER_AREA)
        return crop

    @torch.no_grad()
    def embed(self, image_bgr: np.ndarray, cx: int, cy: int) -> torch.Tensor:
        crop_bgr = self._crop(image_bgr, int(cx), int(cy))
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        feat = self.model.forward_features(tensor)
        # forward_features returns (B, N+1, C) for ViT — index 0 is CLS.
        if feat.dim() == 3:
            cls = feat[:, 0]
        else:
            cls = feat
        return F.normalize(cls, dim=-1).squeeze(0)

    @torch.no_grad()
    def search_best(self, image_bgr: np.ndarray, ref_feat: torch.Tensor,
                    cx: int, cy: int, roi: int = 100, stride: int = 8) -> tuple:
        """Return (best_x, best_y, best_sim) within an ROI window around (cx, cy)."""
        h, w = image_bgr.shape[:2]
        half = roi // 2
        xs = list(range(max(0, cx - half), min(w, cx + half) + 1, stride))
        ys = list(range(max(0, cy - half), min(h, cy + half) + 1, stride))
        if not xs or not ys:
            return cx, cy, -1.0

        best = (-1.0, cx, cy)
        for y in ys:
            for x in xs:
                feat = self.embed(image_bgr, x, y)
                sim = float(torch.dot(feat, ref_feat))
                if sim > best[0]:
                    best = (sim, x, y)
        return best[1], best[2], best[0]

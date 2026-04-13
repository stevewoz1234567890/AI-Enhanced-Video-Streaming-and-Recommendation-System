"""CNN visual encoder (ResNet-18 backbone) + ImageNet label pooling across frames."""

from __future__ import annotations

import threading

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

_lock = threading.Lock()
_backbone: nn.Module | None = None
_preprocess = None
_categories: list[str] | None = None
_fc_weight: torch.Tensor | None = None
_fc_bias: torch.Tensor | None = None


def _load(device: torch.device) -> None:
    global _backbone, _preprocess, _categories, _fc_weight, _fc_bias
    with _lock:
        if _backbone is not None:
            return
        weights = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=weights)
        _categories = list(weights.meta["categories"])
        _preprocess = weights.transforms()
        _fc_weight = net.fc.weight.detach().to(device)
        _fc_bias = net.fc.bias.detach().to(device)
        net.fc = nn.Identity()
        net.eval()
        net.to(device)
        _backbone = net


@torch.inference_mode()
def encode_frame_batch(image_paths: list[str], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits Bx1000, embeddings Bx512) as float32 numpy on CPU."""
    _load(device)
    assert _backbone is not None and _preprocess is not None and _fc_weight is not None and _fc_bias is not None
    tensors = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        tensors.append(_preprocess(img))
    batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
    emb = _backbone(batch)
    logits = emb @ _fc_weight.T + _fc_bias
    return logits.detach().float().cpu().numpy(), emb.detach().float().cpu().numpy()


def pooled_imagenet_labels(
    all_logits: list[np.ndarray],
    categories: list[str],
    top_k: int = 8,
) -> list[tuple[str, float]]:
    if not all_logits:
        return []
    stacked = np.concatenate(all_logits, axis=0)
    probs = torch.softmax(torch.from_numpy(stacked), dim=-1).numpy()
    mean_prob = probs.mean(axis=0)
    idx = np.argsort(-mean_prob)[:top_k]
    return [(categories[i], float(mean_prob[i])) for i in idx]


def imagenet_categories() -> list[str]:
    _load(torch.device("cpu"))
    assert _categories is not None
    return _categories

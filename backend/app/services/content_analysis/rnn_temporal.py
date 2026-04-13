"""GRU over CNN frame embeddings for sequential / transition structure (extend with supervised heads as needed)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TransitionGRU(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _h = self.gru(x)
        return out


@torch.inference_mode()
def transition_scores_from_sequence(embeddings: np.ndarray, device: torch.device) -> np.ndarray:
    """
    embeddings: (T, D) float32 — CNN frame features in time order.
    Returns length T-1 array of combined transition strengths in [0, 1].
    """
    if embeddings.shape[0] < 2:
        return np.array([], dtype=np.float64)
    torch.manual_seed(0)
    x = torch.from_numpy(embeddings).float().unsqueeze(0).to(device)
    model = TransitionGRU(input_dim=embeddings.shape[1], hidden_dim=128, num_layers=1).to(device)
    model.eval()
    h_seq = model(x)[0]
    feat_delta = torch.linalg.norm(x[0, 1:] - x[0, :-1], dim=-1)
    hidden_delta = torch.linalg.norm(h_seq[1:] - h_seq[:-1], dim=-1)
    combined = 0.65 * feat_delta + 0.35 * hidden_delta
    c = combined.detach().float().cpu().numpy()
    cmax = float(c.max()) if c.size else 1.0
    if cmax <= 1e-8:
        return np.zeros_like(c)
    return (c / cmax).clip(0.0, 1.0)


def peaks_to_timestamps(
    scores: np.ndarray,
    frame_times_sec: np.ndarray,
    min_gap_frames: int = 3,
    relative_threshold: float = 0.45,
) -> list[float]:
    if scores.size == 0 or frame_times_sec.size < 2:
        return []
    thr = float(np.quantile(scores, 0.85)) * relative_threshold + float(np.median(scores)) * (1.0 - relative_threshold)
    thr = max(thr, 1e-6)
    picks: list[int] = []
    last = -10_000
    for i in range(len(scores)):
        if scores[i] < thr:
            continue
        left = scores[i - 1] if i > 0 else 0.0
        right = scores[i + 1] if i + 1 < len(scores) else 0.0
        if scores[i] >= left and scores[i] >= right and (i - last) >= min_gap_frames:
            picks.append(i)
            last = i
    return [float(frame_times_sec[i + 1]) for i in picks if i + 1 < len(frame_times_sec)]

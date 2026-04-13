"""Orchestrates CNN visual encoding, GRU transition analysis, and NLP theme scoring."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

from app.content_schemas import ContentAnalysisResponse, ScoredLabel
from app.services.content_analysis import cnn_visual, nlp_themes, rnn_temporal, video_frames


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analyze_media(
    video_path: str,
    transcript: str | None,
    sample_fps: float = 0.5,
    max_frames: int = 120,
    batch_size: int = 8,
) -> ContentAnalysisResponse:
    device = _device()
    tmpdir: str | None = None
    try:
        tmpdir, paths = video_frames.extract_frame_paths(video_path, sample_fps=sample_fps, max_frames=max_frames)
        frame_times = np.arange(len(paths), dtype=np.float64) / float(sample_fps)

        all_logits: list[np.ndarray] = []
        all_embs: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            logits, emb = cnn_visual.encode_frame_batch(chunk, device)
            all_logits.append(logits)
            all_embs.append(emb)

        categories = cnn_visual.imagenet_categories()
        top = cnn_visual.pooled_imagenet_labels(all_logits, categories, top_k=8)
        embeddings = np.concatenate(all_embs, axis=0)

        t_scores = rnn_temporal.transition_scores_from_sequence(embeddings, device)
        cuts = rnn_temporal.peaks_to_timestamps(t_scores, frame_times)

        themes: list[ScoredLabel] = []
        if transcript and transcript.strip():
            for name, sc in nlp_themes.score_themes(transcript):
                themes.append(ScoredLabel(label=name, score=round(float(sc), 4)))

        return ContentAnalysisResponse(
            sample_fps=sample_fps,
            frames_analyzed=len(paths),
            visual_top_labels=[ScoredLabel(label=a, score=round(b, 4)) for a, b in top],
            scene_transition_timestamps_sec=cuts,
            dialogue_themes=themes,
            models={
                "visual_cnn": "torchvision ResNet-18 (ImageNet1K_V1)",
                "temporal_rnn": "PyTorch GRU over frame embeddings + feature deltas",
                "dialogue_nlp": "scikit-learn TF–IDF + theme seed phrases (transformer-ready)",
            },
            notes="TensorFlow equivalents: same API with tf.keras Applications + RNN layers + text pipeline.",
        )
    finally:
        if tmpdir and Path(tmpdir).exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


def analysis_available() -> bool:
    try:
        import sklearn  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return shutil.which("ffmpeg") is not None

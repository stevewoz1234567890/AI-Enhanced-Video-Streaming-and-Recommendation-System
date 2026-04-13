"""Time-series style bottleneck signals; swap LSTM/ARIMA (TensorFlow/PyTorch/Spark) for production."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastResult:
    horizon_steps: int
    predicted_bandwidth_mbps: list[float]
    predicted_latency_ms: list[float]
    bottleneck_risk: float  # 0–1 coarse score


def _linear_trend_next(values: np.ndarray, horizon: int) -> np.ndarray:
    n = len(values)
    if n < 2:
        return np.full(horizon, float(values[-1]) if n else 0.0)
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, values.astype(np.float64), 1)
    out = []
    for h in range(1, horizon + 1):
        out.append(float(intercept + slope * (n - 1 + h)))
    return np.array(out, dtype=np.float64)


def forecast_network(
    bandwidth_history_mbps: list[float],
    latency_history_ms: list[float],
    horizon: int = 6,
) -> ForecastResult:
    bw = np.array(bandwidth_history_mbps[-128:], dtype=np.float64)
    lat = np.array(latency_history_ms[-128:], dtype=np.float64)
    if bw.size == 0:
        bw = np.array([5.0])
    if lat.size == 0:
        lat = np.array([40.0])

    pred_bw = _linear_trend_next(bw, horizon).tolist()
    pred_lat = _linear_trend_next(lat, horizon).tolist()

    bw_drop = (float(np.mean(bw)) - pred_bw[-1]) / max(float(np.mean(bw)), 1e-6)
    lat_rise = (pred_lat[-1] - float(np.mean(lat))) / max(float(np.mean(lat)), 1e-6)
    bottleneck_risk = float(np.clip(0.5 * max(bw_drop, 0) + 0.5 * max(lat_rise, 0), 0.0, 1.0))

    return ForecastResult(
        horizon_steps=horizon,
        predicted_bandwidth_mbps=pred_bw,
        predicted_latency_ms=pred_lat,
        bottleneck_risk=bottleneck_risk,
    )

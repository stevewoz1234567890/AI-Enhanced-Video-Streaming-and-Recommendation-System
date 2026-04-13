"""
Tabular Q-learning style policy over a discretized state (production: replace with TF/PyTorch + SageMaker).
Reward is shaped from stall seconds and quality height when feedback is posted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

LADDER = (
    {"rungs": "2160p", "height": 2160, "bitrate_mbps": 25.0},
    {"rungs": "1440p", "height": 1440, "bitrate_mbps": 14.0},
    {"rungs": "1080p", "height": 1080, "bitrate_mbps": 8.0},
    {"rungs": "720p", "height": 720, "bitrate_mbps": 5.0},
    {"rungs": "480p", "height": 480, "bitrate_mbps": 2.5},
    {"rungs": "360p", "height": 360, "bitrate_mbps": 1.2},
)


def _bucket(x: float, edges: tuple[float, ...]) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def state_key(
    bandwidth_mbps: float,
    latency_ms: float,
    congestion: float,
    user_max_height: int,
    buffering_tolerance_sec: float,
) -> tuple[int, int, int, int, int]:
    return (
        _bucket(bandwidth_mbps, (3, 8, 15, 30)),
        _bucket(latency_ms, (40, 80, 150, 300)),
        _bucket(congestion, (0.15, 0.35, 0.55, 0.75)),
        _bucket(float(user_max_height), (480, 720, 1080, 1440)),
        _bucket(buffering_tolerance_sec, (2, 4, 8, 15)),
    )


@dataclass
class QualityChoice:
    rungs: str
    height: int
    target_bitrate_mbps: float
    policy: str  # "heuristic" | "epsilon_greedy"


class QualityAgent:
    def __init__(self, epsilon: float = 0.12, learning_rate: float = 0.35) -> None:
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.q: dict[tuple, list[float]] = {}
        self._last: dict[str, tuple[tuple, int]] = {}

    def _ensure(self, key: tuple) -> list[float]:
        if key not in self.q:
            self.q[key] = [0.0] * len(LADDER)
        return self.q[key]

    def recommend(
        self,
        user_id: str,
        bandwidth_mbps: float,
        latency_ms: float,
        congestion: float,
        user_max_height: int,
        buffering_tolerance_sec: float,
        forecast_bottleneck_risk: float | None = None,
        heuristic_only: bool = False,
    ) -> QualityChoice:
        effective_bw = max(bandwidth_mbps * (1.0 - min(congestion, 0.95)) * 0.92, 0.5)
        if forecast_bottleneck_risk is not None:
            effective_bw *= max(0.55, 1.0 - 0.45 * forecast_bottleneck_risk)

        latency_pressure = min(1.0, latency_ms / max(buffering_tolerance_sec * 200.0, 1.0))
        effective_bw *= 1.0 - 0.35 * latency_pressure

        key = state_key(bandwidth_mbps, latency_ms, congestion, user_max_height, buffering_tolerance_sec)
        qrow = self._ensure(key)

        best_i = len(LADDER) - 1
        for i, rung in enumerate(LADDER):
            if rung["height"] <= user_max_height and rung["bitrate_mbps"] <= effective_bw:
                best_i = i
                break
        heuristic_idx = best_i

        if heuristic_only:
            rung = LADDER[heuristic_idx]
            self._last[user_id] = (key, heuristic_idx)
            return QualityChoice(
                rungs=str(rung["rungs"]),
                height=int(rung["height"]),
                target_bitrate_mbps=float(rung["bitrate_mbps"]),
                policy="heuristic_fast",
            )

        if random.random() < self.epsilon:
            valid = [i for i, r in enumerate(LADDER) if r["height"] <= user_max_height]
            action_i = random.choice(valid) if valid else heuristic_idx
            policy = "epsilon_greedy"
        else:
            masked = [-1e9] * len(LADDER)
            for i, rung in enumerate(LADDER):
                if rung["height"] <= user_max_height:
                    masked[i] = qrow[i]
            action_i = max(range(len(LADDER)), key=lambda i: masked[i])
            if max(masked) <= -1e8:
                action_i = heuristic_idx
            policy = "epsilon_greedy"

        if max(qrow) - min(qrow) < 0.25:
            action_i = heuristic_idx
            policy = "heuristic"

        rung = LADDER[action_i]
        self._last[user_id] = (key, action_i)
        return QualityChoice(
            rungs=str(rung["rungs"]),
            height=int(rung["height"]),
            target_bitrate_mbps=float(rung["bitrate_mbps"]),
            policy=policy,
        )

    def learn(
        self,
        user_id: str,
        stall_seconds: float,
        played_height: int,
        ideal_height: int,
    ) -> None:
        last = self._last.pop(user_id, None)
        if last is None:
            return
        key, action_i = last
        qrow = self._ensure(key)
        quality_term = min(1.0, played_height / max(ideal_height, 1))
        reward = quality_term * 1.0 - 2.5 * stall_seconds
        td_target = reward
        qrow[action_i] += self.learning_rate * (td_target - qrow[action_i])


agent = QualityAgent()

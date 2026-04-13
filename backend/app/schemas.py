from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PreferenceIn(BaseModel):
    preferred_max_height: int = Field(ge=144, le=4320, default=1080)
    buffering_tolerance_sec: float = Field(gt=0, le=120, default=5.0)


class PreferenceOut(BaseModel):
    user_id: str
    preferred_max_height: int
    buffering_tolerance_sec: float


class NetworkProbeIn(BaseModel):
    bandwidth_mbps: float = Field(gt=0)
    latency_ms: float = Field(ge=0)
    congestion: float = Field(ge=0, le=1, default=0.0)
    source: str = Field(default="probe", max_length=32)


class QualityRequest(BaseModel):
    user_id: str
    bandwidth_mbps: float = Field(gt=0)
    latency_ms: float = Field(ge=0)
    congestion: float = Field(ge=0, le=1, default=0.0)
    use_forecast: bool = True
    fast_path: bool = Field(
        default=False,
        description="Skip forecast DB read and RL exploration for lower latency (edge / real-time paths).",
    )


class QualityResponse(BaseModel):
    rungs: str
    height: int
    target_bitrate_mbps: float
    policy: str
    delivery: str = "DASH or HLS ladder index; map to manifest rung"


class PlaybackFeedback(BaseModel):
    user_id: str
    stall_seconds: float = Field(ge=0)
    played_height: int = Field(ge=144, le=4320)
    ideal_height: int = Field(ge=144, le=4320)


class ForecastQuery(BaseModel):
    horizon: int = Field(default=6, ge=1, le=48)


class ViewingEventIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    content_id: str = Field(min_length=1, max_length=128)
    watch_seconds: float = Field(ge=0, default=0.0)
    rating: float | None = Field(default=None, ge=1.0, le=5.0)


class RecommendedItem(BaseModel):
    content_id: str
    score: float
    source: str  # hybrid_mlp | popularity | matrix_factorization


class RecommendationsResponse(BaseModel):
    user_id: str
    items: list[RecommendedItem]
    model_notes: dict[str, str]


class IngestEventIn(BaseModel):
    source: Literal["video_server", "user_device", "edge", "infra"]
    origin_id: str = Field(max_length=128)
    event_type: str = Field(max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)
    client_ts: datetime | None = None


class IngestBatchIn(BaseModel):
    events: list[IngestEventIn] = Field(max_length=500)


class EdgeRegisterIn(BaseModel):
    edge_id: str = Field(max_length=64)
    region: str = Field(default="default", max_length=64)
    base_url: str = Field(max_length=512)
    capacity_units: float = Field(gt=0, default=1.0)


class DeliveryOptimizeRequest(BaseModel):
    user_id: str
    bandwidth_mbps: float = Field(gt=0)
    latency_ms: float = Field(ge=0)
    congestion: float = Field(ge=0, le=1, default=0.0)
    client_region: str | None = Field(default=None, max_length=64)
    use_forecast: bool = True
    fast_path: bool = Field(
        default=False,
        description="Low-latency ABR: no predictive horizon, heuristic ladder only.",
    )


class DeliveryOptimizeResponse(BaseModel):
    quality: QualityResponse
    edge: dict[str, str | None]
    forecast_bottleneck_risk: float | None
    delivery_hints: dict[str, str]


class UserInteractionIn(BaseModel):
    user_id: str = Field(max_length=64)
    interaction_type: str = Field(max_length=64)
    content_id: str | None = Field(default=None, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)


class ModelMetricIn(BaseModel):
    model_name: str = Field(max_length=64)
    version: str = Field(max_length=32)
    metric_name: str = Field(max_length=64)
    metric_value: float
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelMetricOut(BaseModel):
    id: int
    model_name: str
    version: str
    metric_name: str
    metric_value: float
    extra: dict[str, Any]
    created_at: datetime


class FeedbackSummaryItem(BaseModel):
    interaction_type: str
    count: int


class FeedbackSummaryResponse(BaseModel):
    interactions_by_type: list[FeedbackSummaryItem]
    viewing_events_total: int
    user_interactions_total: int
    network_samples_total: int
    ingestion_events_total: int

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

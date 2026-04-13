from pydantic import BaseModel, Field


class ScoredLabel(BaseModel):
    label: str
    score: float = Field(ge=0.0)


class ContentAnalysisResponse(BaseModel):
    sample_fps: float
    frames_analyzed: int
    visual_top_labels: list[ScoredLabel]
    scene_transition_timestamps_sec: list[float]
    dialogue_themes: list[ScoredLabel]
    models: dict[str, str]
    notes: str | None = None

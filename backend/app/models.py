from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UserPreference(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    preferred_max_height: Mapped[int] = mapped_column(Integer, default=1080)
    buffering_tolerance_sec: Mapped[float] = mapped_column(Float, default=5.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class UserPrivacySettings(Base):
    """
    Opt-out style controls (default allow). Production may require explicit opt-in per GDPR/CCPA programs.
    """

    __tablename__ = "user_privacy_settings"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    consent_personalization: Mapped[bool] = mapped_column(Boolean, default=True)
    consent_analytics: Mapped[bool] = mapped_column(Boolean, default=True)
    consent_model_training: Mapped[bool] = mapped_column(Boolean, default=True)
    policy_version_ack: Mapped[str | None] = mapped_column(String(32), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ViewingEvent(Base):
    """Implicit/explicit feedback for collaborative filtering and hybrid recommenders."""

    __tablename__ = "viewing_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    content_id: Mapped[str] = mapped_column(String(128), index=True)
    watch_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IngestionEvent(Base):
    """Central aggregator: video servers, user devices, edges, infra (cache/LB/P2P telemetry)."""

    __tablename__ = "ingestion_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    origin_id: Mapped[str] = mapped_column(String(128), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    client_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EdgeNode(Base):
    """Edge / packaging nodes for ABR delivery and predictive routing."""

    __tablename__ = "edge_nodes"

    edge_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    region: Mapped[str] = mapped_column(String(64), default="default")
    base_url: Mapped[str] = mapped_column(String(512))
    capacity_units: Mapped[float] = mapped_column(Float, default=1.0)
    current_load: Mapped[float] = mapped_column(Float, default=0.0)
    healthy: Mapped[bool] = mapped_column(Boolean, default=True)


class TrainingQualityMetric(Base):
    """Offline / batch job reports (Spark, SageMaker, etc.) for monitoring model accuracy over time."""

    __tablename__ = "training_quality_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32), index=True)
    metric_name: Mapped[str] = mapped_column(String(64))
    metric_value: Mapped[float] = mapped_column(Float)
    extra: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FairnessAuditRecord(Base):
    """Bias / fairness slices from offline audits (fairness-aware ML, representative data checks)."""

    __tablename__ = "fairness_audit_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    slice_name: Mapped[str] = mapped_column(String(128), index=True)
    metric_name: Mapped[str] = mapped_column(String(64))
    metric_value: Mapped[float] = mapped_column(Float)
    extra: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserInteraction(Base):
    """Feedback loop: arbitrary interactions to refine recommendations and delivery models."""

    __tablename__ = "user_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    interaction_type: Mapped[str] = mapped_column(String(64), index=True)
    content_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class NetworkSample(Base):
    """Historical probes: bandwidth (Mbps), latency (ms), congestion score0–1."""

    __tablename__ = "network_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    bandwidth_mbps: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    congestion: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32), default="probe")

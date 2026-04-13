import asyncio
import json
import math
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import engine, get_session
from app.content_schemas import ContentAnalysisResponse
from app.metrics_prom import (
    ABR_FAST_PATH_DECISIONS,
    CONTENT_ANALYSIS_REQUESTS,
    CONTENT_ANALYSIS_SECONDS,
    DECISION_LATENCY,
    FAIRNESS_AUDIT_INGEST,
    FORECAST_REQUESTS,
    INGESTION_EVENTS,
    LAST_BANDWIDTH_MBPS,
    LAST_LATENCY_MS,
    MODEL_METRIC_INGEST,
    NETWORK_PROBE_INGEST,
    PRIVACY_ERASURE_REQUESTS,
    PRIVACY_EXPORT_REQUESTS,
    QUALITY_DECISIONS,
    RECOMMENDATION_FIT_SECONDS,
    RECOMMENDATION_REQUESTS,
    TRAINING_EXPORT_REQUESTS,
    USER_INTERACTIONS,
    metrics_response,
)
from app.models import (
    Base,
    EdgeNode,
    FairnessAuditRecord,
    IngestionEvent,
    NetworkSample,
    TrainingQualityMetric,
    UserInteraction,
    UserPreference,
    UserPrivacySettings,
    ViewingEvent,
)
from app.schemas import (
    DeliveryOptimizeRequest,
    DeliveryOptimizeResponse,
    EdgeRegisterIn,
    ErasureRequest,
    FairnessReportIn,
    FairnessReportOut,
    FeedbackSummaryItem,
    FeedbackSummaryResponse,
    ForecastQuery,
    IngestBatchIn,
    IngestEventIn,
    ModelMetricIn,
    ModelMetricOut,
    NetworkProbeIn,
    PlaybackFeedback,
    PreferenceIn,
    PreferenceOut,
    PrivacyConsentIn,
    PrivacyConsentOut,
    QualityRequest,
    QualityResponse,
    RecommendedItem,
    RecommendationsResponse,
    TransparencyResponse,
    UserInteractionIn,
    ViewingEventIn,
)
from app.services.abr import build_quality_from_request
from app.services.forecaster import forecast_network
from app.services.quality_rl import agent

_INTERACTION_METRIC_LABELS = frozenset(
    {
        "playback_quality_feedback",
        "click",
        "search",
        "skip",
        "rate",
        "explicit_dislike",
        "not_interested",
        "manifest_switch",
    }
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="AI-Enhanced Streaming Control Plane", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/privacy/transparency", response_model=TransparencyResponse)
async def privacy_transparency():
    return TransparencyResponse(
        policy_version=settings.privacy_policy_version,
        policy_url=settings.privacy_policy_url,
        data_protection_contact=settings.data_protection_contact,
        data_categories=[
            "Playback and quality preferences (user_preferences)",
            "Viewing history and ratings (viewing_events)",
            "Feedback and UI interactions (user_interactions)",
            "Network telemetry used for ABR (network_samples; often aggregated)",
            "Operational ingest from servers/devices (ingestion_events)",
        ],
        processing_purposes=[
            "Adaptive streaming quality and delivery optimization",
            "Content recommendations and personalization",
            "Capacity forecasting and infrastructure analytics",
            "Model training and evaluation when consented",
        ],
        user_rights_summary=[
            "Access / portability: GET /privacy/export/{user_id}",
            "Erasure: POST /privacy/erasure with confirm=true",
            "Consent controls: PUT /privacy/consent",
            "Opt out of personalization, analytics, or model-training datasets independently",
        ],
        encryption_notes=[
            "In transit: terminate TLS at your reverse proxy or API gateway (HTTPS).",
            "At rest: use encrypted PostgreSQL (e.g. cloud KMS, volume encryption).",
            "Do not log raw payloads containing secrets; rotate DATABASE_URL credentials.",
        ],
        compliance_notes=[
            "Defaults allow processing until users opt out via user_privacy_settings (tune for strict opt-in jurisdictions).",
            "Map retention windows and legal basis to your GDPR/CCPA register; this API does not auto-delete by age.",
            "Training export excludes users who set consent_model_training=false; device ingest excludes consent_analytics=false.",
        ],
        api_endpoints={
            "transparency": "/privacy/transparency",
            "consent_get": "/privacy/consent/{user_id}",
            "consent_put": "/privacy/consent",
            "export": "/privacy/export/{user_id}",
            "erasure": "/privacy/erasure",
            "fairness_reports": "/ethics/fairness-reports",
        },
    )


@app.put("/privacy/consent", response_model=PrivacyConsentOut)
async def put_privacy_consent(body: PrivacyConsentIn, session: AsyncSession = Depends(get_session)):
    row = await session.get(UserPrivacySettings, body.user_id)
    if row is None:
        row = UserPrivacySettings(
            user_id=body.user_id,
            consent_personalization=(
                body.consent_personalization if body.consent_personalization is not None else True
            ),
            consent_analytics=body.consent_analytics if body.consent_analytics is not None else True,
            consent_model_training=body.consent_model_training if body.consent_model_training is not None else True,
            policy_version_ack=body.policy_version_ack,
        )
        session.add(row)
    else:
        if body.consent_personalization is not None:
            row.consent_personalization = body.consent_personalization
        if body.consent_analytics is not None:
            row.consent_analytics = body.consent_analytics
        if body.consent_model_training is not None:
            row.consent_model_training = body.consent_model_training
        if body.policy_version_ack is not None:
            row.policy_version_ack = body.policy_version_ack
    await session.commit()
    await session.refresh(row)
    return PrivacyConsentOut(
        user_id=row.user_id,
        consent_personalization=row.consent_personalization,
        consent_analytics=row.consent_analytics,
        consent_model_training=row.consent_model_training,
        policy_version_ack=row.policy_version_ack,
        updated_at=row.updated_at,
    )


@app.get("/privacy/consent/{user_id}", response_model=PrivacyConsentOut)
async def get_privacy_consent(user_id: str, session: AsyncSession = Depends(get_session)):
    row = await session.get(UserPrivacySettings, user_id)
    if row is None:
        return PrivacyConsentOut(
            user_id=user_id,
            consent_personalization=True,
            consent_analytics=True,
            consent_model_training=True,
            policy_version_ack=None,
            updated_at=None,
        )
    return PrivacyConsentOut(
        user_id=row.user_id,
        consent_personalization=row.consent_personalization,
        consent_analytics=row.consent_analytics,
        consent_model_training=row.consent_model_training,
        policy_version_ack=row.policy_version_ack,
        updated_at=row.updated_at,
    )


@app.get("/privacy/export/{user_id}")
async def privacy_export_user(user_id: str, session: AsyncSession = Depends(get_session)):
    from app.services.privacy import export_portable_data

    PRIVACY_EXPORT_REQUESTS.inc()
    return await export_portable_data(session, user_id)


@app.post("/privacy/erasure")
async def privacy_erasure(body: ErasureRequest, session: AsyncSession = Depends(get_session)):
    from app.services.privacy import erase_user_data
    from app.services.recommendations.hybrid_engine import invalidate_cache

    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail="Erasure is destructive. Resend with confirm=true after user verification.",
        )
    await erase_user_data(session, body.user_id)
    await session.commit()
    agent.forget_user(body.user_id)
    invalidate_cache()
    PRIVACY_ERASURE_REQUESTS.inc()
    return {"ok": True, "erased_user_id": body.user_id}


@app.post("/ethics/fairness-report", response_model=FairnessReportOut)
async def post_fairness_report(body: FairnessReportIn, session: AsyncSession = Depends(get_session)):
    row = FairnessAuditRecord(
        model_name=body.model_name,
        slice_name=body.slice_name,
        metric_name=body.metric_name,
        metric_value=body.metric_value,
        extra=body.extra,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    FAIRNESS_AUDIT_INGEST.labels(model_name=body.model_name).inc()
    return FairnessReportOut(
        id=row.id,
        model_name=row.model_name,
        slice_name=row.slice_name,
        metric_name=row.metric_name,
        metric_value=row.metric_value,
        extra=row.extra,
        created_at=row.created_at,
    )


@app.get("/ethics/fairness-reports", response_model=list[FairnessReportOut])
async def list_fairness_reports(
    session: AsyncSession = Depends(get_session),
    model_name: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=500),
):
    stmt = select(FairnessAuditRecord)
    if model_name:
        stmt = stmt.where(FairnessAuditRecord.model_name == model_name)
    stmt = stmt.order_by(FairnessAuditRecord.id.desc()).limit(limit)
    res = await session.execute(stmt)
    rows = res.scalars().all()
    return [
        FairnessReportOut(
            id=r.id,
            model_name=r.model_name,
            slice_name=r.slice_name,
            metric_name=r.metric_name,
            metric_value=r.metric_value,
            extra=r.extra,
            created_at=r.created_at,
        )
        for r in rows
    ]


@app.get("/metrics")
async def metrics():
    body, ctype = metrics_response()
    return Response(content=body, media_type=ctype)


@app.put("/users/{user_id}/preferences", response_model=PreferenceOut)
async def put_preferences(
    user_id: str,
    body: PreferenceIn,
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(UserPreference, user_id)
    if row is None:
        row = UserPreference(user_id=user_id)
        session.add(row)
    row.preferred_max_height = body.preferred_max_height
    row.buffering_tolerance_sec = body.buffering_tolerance_sec
    await session.commit()
    await session.refresh(row)
    return PreferenceOut(
        user_id=row.user_id,
        preferred_max_height=row.preferred_max_height,
        buffering_tolerance_sec=row.buffering_tolerance_sec,
    )


@app.get("/users/{user_id}/preferences", response_model=PreferenceOut)
async def get_preferences(user_id: str, session: AsyncSession = Depends(get_session)):
    row = await session.get(UserPreference, user_id)
    if row is None:
        row = UserPreference(user_id=user_id)
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return PreferenceOut(
        user_id=row.user_id,
        preferred_max_height=row.preferred_max_height,
        buffering_tolerance_sec=row.buffering_tolerance_sec,
    )


@app.post("/network/probe")
async def ingest_probe(body: NetworkProbeIn, session: AsyncSession = Depends(get_session)):
    sample = NetworkSample(
        bandwidth_mbps=body.bandwidth_mbps,
        latency_ms=body.latency_ms,
        congestion=body.congestion,
        source=body.source,
    )
    session.add(sample)
    await session.commit()
    NETWORK_PROBE_INGEST.inc()
    LAST_BANDWIDTH_MBPS.set(body.bandwidth_mbps)
    LAST_LATENCY_MS.set(body.latency_ms)
    return {"stored": True}


@app.post("/ingest/event")
async def ingest_event(body: IngestEventIn, session: AsyncSession = Depends(get_session)):
    from app.services.integration_hooks import apply_ingest_side_effects

    session.add(
        IngestionEvent(
            source=body.source,
            origin_id=body.origin_id,
            event_type=body.event_type,
            payload=body.payload,
            client_ts=body.client_ts,
        )
    )
    await apply_ingest_side_effects(session, body)
    await session.commit()
    INGESTION_EVENTS.labels(source=body.source).inc()
    return {"stored": True}


@app.post("/ingest/batch")
async def ingest_batch(body: IngestBatchIn, session: AsyncSession = Depends(get_session)):
    from app.services.integration_hooks import apply_ingest_side_effects

    for ev in body.events:
        session.add(
            IngestionEvent(
                source=ev.source,
                origin_id=ev.origin_id,
                event_type=ev.event_type,
                payload=ev.payload,
                client_ts=ev.client_ts,
            )
        )
        await apply_ingest_side_effects(session, ev)
        INGESTION_EVENTS.labels(source=ev.source).inc()
    await session.commit()
    return {"stored": len(body.events)}


@app.post("/edges/register")
async def register_edge(body: EdgeRegisterIn, session: AsyncSession = Depends(get_session)):
    row = await session.get(EdgeNode, body.edge_id)
    if row is None:
        session.add(
            EdgeNode(
                edge_id=body.edge_id,
                region=body.region,
                base_url=body.base_url,
                capacity_units=body.capacity_units,
            )
        )
    else:
        row.region = body.region
        row.base_url = body.base_url
        row.capacity_units = body.capacity_units
        row.healthy = True
    await session.commit()
    return {"ok": True, "edge_id": body.edge_id}


@app.post("/delivery/optimize", response_model=DeliveryOptimizeResponse)
async def delivery_optimize(body: DeliveryOptimizeRequest, session: AsyncSession = Depends(get_session)):
    from app.services.delivery_router import build_delivery_plan

    t0 = time.perf_counter()
    plan = await build_delivery_plan(session, body)
    QUALITY_DECISIONS.labels(recommended_height=str(plan.quality.height)).inc()
    DECISION_LATENCY.observe(time.perf_counter() - t0)
    if body.fast_path:
        ABR_FAST_PATH_DECISIONS.inc()
    return plan


@app.post("/quality/recommend", response_model=QualityResponse)
async def recommend_quality(body: QualityRequest, session: AsyncSession = Depends(get_session)):
    t0 = time.perf_counter()
    resp, _risk = await build_quality_from_request(session, body)
    QUALITY_DECISIONS.labels(recommended_height=str(resp.height)).inc()
    DECISION_LATENCY.observe(time.perf_counter() - t0)
    if body.fast_path:
        ABR_FAST_PATH_DECISIONS.inc()
    return resp


@app.post("/quality/feedback")
async def quality_feedback(body: PlaybackFeedback, session: AsyncSession = Depends(get_session)):
    from app.services.privacy import require_personalization

    await require_personalization(session, body.user_id)
    agent.learn(
        user_id=body.user_id,
        stall_seconds=body.stall_seconds,
        played_height=body.played_height,
        ideal_height=body.ideal_height,
    )
    return {"ok": True}


@app.post("/feedback/interaction")
async def feedback_interaction(body: UserInteractionIn, session: AsyncSession = Depends(get_session)):
    from app.services.privacy import require_personalization
    from app.services.recommendations.hybrid_engine import invalidate_cache

    await require_personalization(session, body.user_id)

    session.add(
        UserInteraction(
            user_id=body.user_id,
            interaction_type=body.interaction_type,
            content_id=body.content_id,
            payload=body.payload,
        )
    )
    await session.commit()

    metric_label = body.interaction_type if body.interaction_type in _INTERACTION_METRIC_LABELS else "custom"
    USER_INTERACTIONS.labels(interaction_type=metric_label).inc()

    if body.interaction_type == "playback_quality_feedback":
        try:
            agent.learn(
                user_id=body.user_id,
                stall_seconds=float(body.payload.get("stall_seconds", 0)),
                played_height=int(body.payload["played_height"]),
                ideal_height=int(body.payload["ideal_height"]),
            )
        except (KeyError, TypeError, ValueError):
            pass

    if body.interaction_type in ("explicit_dislike", "not_interested", "rate", "rating"):
        invalidate_cache()

    return {"ok": True}


@app.get("/analytics/export/training")
async def export_training_snapshot(
    session: AsyncSession = Depends(get_session),
    streams: str = Query(default="network,viewing,ingestion,interactions"),
    limit_per_stream: int = Query(default=5000, ge=1, le=100_000),
):
    from app.services.training_export import collect_training_records

    TRAINING_EXPORT_REQUESTS.inc()
    stream_list = [s.strip() for s in streams.split(",") if s.strip()]
    rows = await collect_training_records(session, stream_list, limit_per_stream)

    def ndjson_iter():
        for row in rows:
            yield json.dumps(row, default=str) + "\n"

    return StreamingResponse(ndjson_iter(), media_type="application/x-ndjson")


@app.get("/analytics/feedback/summary", response_model=FeedbackSummaryResponse)
async def feedback_summary(session: AsyncSession = Depends(get_session)):
    viewing_total = int(
        (await session.execute(select(func.count()).select_from(ViewingEvent))).scalar_one() or 0
    )
    user_interactions_total = int(
        (await session.execute(select(func.count()).select_from(UserInteraction))).scalar_one() or 0
    )
    network_samples_total = int(
        (await session.execute(select(func.count()).select_from(NetworkSample))).scalar_one() or 0
    )
    ingestion_events_total = int(
        (await session.execute(select(func.count()).select_from(IngestionEvent))).scalar_one() or 0
    )
    cnt = func.count().label("cnt")
    res = await session.execute(
        select(UserInteraction.interaction_type, cnt)
        .group_by(UserInteraction.interaction_type)
        .order_by(cnt.desc())
    )
    by_type = [FeedbackSummaryItem(interaction_type=r[0], count=int(r[1])) for r in res.all()]
    return FeedbackSummaryResponse(
        interactions_by_type=by_type,
        viewing_events_total=viewing_total,
        user_interactions_total=user_interactions_total,
        network_samples_total=network_samples_total,
        ingestion_events_total=ingestion_events_total,
    )


@app.post("/models/metrics", response_model=ModelMetricOut)
async def ingest_model_metric(body: ModelMetricIn, session: AsyncSession = Depends(get_session)):
    row = TrainingQualityMetric(
        model_name=body.model_name,
        version=body.version,
        metric_name=body.metric_name,
        metric_value=body.metric_value,
        extra=body.extra,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    MODEL_METRIC_INGEST.labels(model_name=body.model_name).inc()
    return ModelMetricOut(
        id=row.id,
        model_name=row.model_name,
        version=row.version,
        metric_name=row.metric_name,
        metric_value=row.metric_value,
        extra=row.extra,
        created_at=row.created_at,
    )


@app.get("/models/metrics", response_model=list[ModelMetricOut])
async def list_model_metrics(
    session: AsyncSession = Depends(get_session),
    model_name: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=500),
):
    stmt = select(TrainingQualityMetric)
    if model_name:
        stmt = stmt.where(TrainingQualityMetric.model_name == model_name)
    stmt = stmt.order_by(TrainingQualityMetric.id.desc()).limit(limit)
    res = await session.execute(stmt)
    rows = res.scalars().all()
    return [
        ModelMetricOut(
            id=r.id,
            model_name=r.model_name,
            version=r.version,
            metric_name=r.metric_name,
            metric_value=r.metric_value,
            extra=r.extra,
            created_at=r.created_at,
        )
        for r in rows
    ]


async def _load_interaction_rows(session: AsyncSession) -> list[tuple[str, str, float, float | None]]:
    opt_out_personal = select(UserPrivacySettings.user_id).where(
        UserPrivacySettings.consent_personalization.is_(False)
    )
    res = await session.execute(
        select(
            ViewingEvent.user_id,
            ViewingEvent.content_id,
            func.coalesce(func.sum(ViewingEvent.watch_seconds), 0.0),
            func.avg(ViewingEvent.rating),
        )
        .where(~ViewingEvent.user_id.in_(opt_out_personal))
        .group_by(ViewingEvent.user_id, ViewingEvent.content_id)
    )
    out: list[tuple[str, str, float, float | None]] = []
    for uid, cid, wsum, ravg in res.all():
        rating = float(ravg) if ravg is not None else None
        if rating is not None and math.isnan(rating):
            rating = None
        out.append((uid, cid, float(wsum), rating))
    return sorted(out, key=lambda x: (x[0], x[1]))


@app.post("/viewing/event")
async def record_viewing_event(body: ViewingEventIn, session: AsyncSession = Depends(get_session)):
    from app.services.privacy import require_personalization
    from app.services.recommendations.hybrid_engine import invalidate_cache

    await require_personalization(session, body.user_id)

    session.add(
        ViewingEvent(
            user_id=body.user_id,
            content_id=body.content_id,
            watch_seconds=body.watch_seconds,
            rating=body.rating,
        )
    )
    await session.commit()
    invalidate_cache()
    return {"stored": True}


@app.get("/recommendations/{user_id}", response_model=RecommendationsResponse)
async def get_recommendations(
    user_id: str,
    session: AsyncSession = Depends(get_session),
    top_k: int = Query(default=10, ge=1, le=100),
    exclude_watched: bool = Query(default=True),
):
    from app.services.privacy import require_personalization
    from app.services.recommendations import hybrid_engine

    await require_personalization(session, user_id)

    if not hybrid_engine.deps_available():
        raise HTTPException(
            status_code=503,
            detail="Recommendations unavailable: install scikit-learn, scipy, and PyTorch (see requirements-analysis.txt).",
        )

    rows = await _load_interaction_rows(session)
    state, cache_hit, fit_sec = await asyncio.to_thread(hybrid_engine.fit_or_get_cached, rows)
    if not cache_hit and fit_sec > 0:
        RECOMMENDATION_FIT_SECONDS.observe(fit_sec)

    exclude: set[str] = set()
    if exclude_watched:
        res = await session.execute(select(ViewingEvent.content_id).where(ViewingEvent.user_id == user_id))
        exclude = {r[0] for r in res.all()}

    raw = hybrid_engine.recommend_for_user(state, user_id, top_k, exclude_watched=exclude)
    fallback = "popularity" if user_id not in state.user_index or state.mlp is None else "hybrid"
    RECOMMENDATION_REQUESTS.labels(fallback=fallback).inc()

    items = [RecommendedItem(content_id=c, score=round(s, 4), source=src) for c, s, src in raw]
    notes = {
        "collaborative_filtering": "TruncatedSVD matrix factorization (scikit-learn)",
        "deep_learning": "PyTorch MLP on concatenated user/item latent factors",
        "tensorflow": "tf.keras.layers.Dense stack on same latent inputs is a drop-in swap",
    }
    return RecommendationsResponse(user_id=user_id, items=items, model_notes=notes)


def _unlink_safe(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


@app.post("/content/analyze", response_model=ContentAnalysisResponse)
async def content_analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    transcript: str | None = Form(default=None),
):
    from app.services.content_analysis.pipeline import analysis_available, analyze_media

    if not analysis_available():
        raise HTTPException(
            status_code=503,
            detail="Content analysis unavailable: install PyTorch, torchvision, scikit-learn, Pillow, and ffmpeg.",
        )

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    background_tasks.add_task(_unlink_safe, path)
    try:
        with open(path, "wb") as out:
            shutil.copyfileobj(video.file, out)
    except Exception:
        _unlink_safe(path)
        raise

    t0 = time.perf_counter()
    try:
        result = await asyncio.to_thread(analyze_media, path, transcript)
        CONTENT_ANALYSIS_REQUESTS.labels(status="ok").inc()
        return result
    except Exception as exc:
        CONTENT_ANALYSIS_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        CONTENT_ANALYSIS_SECONDS.observe(time.perf_counter() - t0)


@app.post("/network/forecast")
async def network_forecast(q: ForecastQuery, session: AsyncSession = Depends(get_session)):
    res = await session.execute(
        select(NetworkSample.bandwidth_mbps, NetworkSample.latency_ms)
        .order_by(NetworkSample.id.desc())
        .limit(128)
    )
    rows = list(reversed(res.all()))
    bw_hist = [float(r[0]) for r in rows]
    lat_hist = [float(r[1]) for r in rows]
    fc = forecast_network(bw_hist, lat_hist, horizon=q.horizon)
    FORECAST_REQUESTS.inc()
    return {
        "horizon_steps": fc.horizon_steps,
        "predicted_bandwidth_mbps": fc.predicted_bandwidth_mbps,
        "predicted_latency_ms": fc.predicted_latency_ms,
        "bottleneck_risk": fc.bottleneck_risk,
    }

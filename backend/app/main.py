import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Response
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import engine, get_session
from app.metrics_prom import (
    DECISION_LATENCY,
    FORECAST_REQUESTS,
    LAST_BANDWIDTH_MBPS,
    LAST_LATENCY_MS,
    NETWORK_PROBE_INGEST,
    QUALITY_DECISIONS,
    metrics_response,
)
from app.models import Base, NetworkSample, UserPreference
from app.schemas import (
    ForecastQuery,
    NetworkProbeIn,
    PlaybackFeedback,
    PreferenceIn,
    PreferenceOut,
    QualityRequest,
    QualityResponse,
)
from app.services.forecaster import forecast_network
from app.services.quality_rl import agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="AI-Enhanced Streaming Control Plane", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


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


@app.post("/quality/recommend", response_model=QualityResponse)
async def recommend_quality(body: QualityRequest, session: AsyncSession = Depends(get_session)):
    t0 = time.perf_counter()
    prefs = await session.get(UserPreference, body.user_id)
    if prefs is None:
        prefs = UserPreference(user_id=body.user_id)
        session.add(prefs)
        await session.commit()
        await session.refresh(prefs)

    risk = None
    if body.use_forecast:
        res = await session.execute(
            select(NetworkSample.bandwidth_mbps, NetworkSample.latency_ms)
            .order_by(NetworkSample.id.desc())
            .limit(64)
        )
        rows = list(reversed(res.all()))
        bw_hist = [float(r[0]) for r in rows]
        lat_hist = [float(r[1]) for r in rows]
        if bw_hist:
            fc = forecast_network(bw_hist, lat_hist)
            risk = fc.bottleneck_risk
            FORECAST_REQUESTS.inc()

    choice = agent.recommend(
        user_id=body.user_id,
        bandwidth_mbps=body.bandwidth_mbps,
        latency_ms=body.latency_ms,
        congestion=body.congestion,
        user_max_height=prefs.preferred_max_height,
        buffering_tolerance_sec=prefs.buffering_tolerance_sec,
        forecast_bottleneck_risk=risk,
    )
    QUALITY_DECISIONS.labels(recommended_height=str(choice.height)).inc()
    DECISION_LATENCY.observe(time.perf_counter() - t0)
    return QualityResponse(
        rungs=choice.rungs,
        height=choice.height,
        target_bitrate_mbps=choice.target_bitrate_mbps,
        policy=choice.policy,
    )


@app.post("/quality/feedback")
async def quality_feedback(body: PlaybackFeedback):
    agent.learn(
        user_id=body.user_id,
        stall_seconds=body.stall_seconds,
        played_height=body.played_height,
        ideal_height=body.ideal_height,
    )
    return {"ok": True}


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

"""Adaptive bitrate decision: user prefs + optional forecast + RL agent."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.metrics_prom import FORECAST_REQUESTS
from app.models import NetworkSample, UserPreference
from app.schemas import QualityRequest, QualityResponse
from app.services.forecaster import forecast_network
from app.services.quality_rl import agent


async def build_quality_response(
    session: AsyncSession,
    user_id: str,
    bandwidth_mbps: float,
    latency_ms: float,
    congestion: float,
    use_forecast: bool,
) -> tuple[QualityResponse, float | None]:
    prefs = await session.get(UserPreference, user_id)
    if prefs is None:
        prefs = UserPreference(user_id=user_id)
        session.add(prefs)
        await session.commit()
        await session.refresh(prefs)

    risk: float | None = None
    if use_forecast:
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
        user_id=user_id,
        bandwidth_mbps=bandwidth_mbps,
        latency_ms=latency_ms,
        congestion=congestion,
        user_max_height=prefs.preferred_max_height,
        buffering_tolerance_sec=prefs.buffering_tolerance_sec,
        forecast_bottleneck_risk=risk,
    )
    return (
        QualityResponse(
            rungs=choice.rungs,
            height=choice.height,
            target_bitrate_mbps=choice.target_bitrate_mbps,
            policy=choice.policy,
        ),
        risk,
    )


async def build_quality_from_request(session: AsyncSession, body: QualityRequest) -> tuple[QualityResponse, float | None]:
    return await build_quality_response(
        session,
        body.user_id,
        body.bandwidth_mbps,
        body.latency_ms,
        body.congestion,
        body.use_forecast,
    )

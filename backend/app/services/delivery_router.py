"""Edge selection and delivery hints using forecast risk + infra state (cache/LB/P2P/routing placeholders)."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EdgeNode
from app.schemas import DeliveryOptimizeRequest, DeliveryOptimizeResponse
from app.services.abr import build_quality_response


async def _choose_edge(
    session: AsyncSession,
    client_region: str | None,
    bottleneck_risk: float | None,
) -> dict[str, str | None]:
    res = await session.execute(select(EdgeNode).where(EdgeNode.healthy.is_(True)))
    edges = list(res.scalars().all())
    if not edges:
        return {"edge_id": None, "base_url": None, "region": None}

    def score(e: EdgeNode) -> float:
        headroom = float(e.capacity_units) - float(e.current_load)
        if client_region and e.region == client_region:
            headroom += 50.0
        if bottleneck_risk is not None:
            headroom -= float(bottleneck_risk) * 35.0
        return headroom

    best = max(edges, key=score)
    return {
        "edge_id": best.edge_id,
        "base_url": best.base_url,
        "region": best.region,
    }


async def build_delivery_plan(session: AsyncSession, body: DeliveryOptimizeRequest) -> DeliveryOptimizeResponse:
    quality, risk = await build_quality_response(
        session,
        body.user_id,
        body.bandwidth_mbps,
        body.latency_ms,
        body.congestion,
        body.use_forecast,
        body.fast_path,
    )
    edge = await _choose_edge(session, body.client_region, risk)

    if risk is not None and risk > 0.55:
        cache_hint = "Prefer longer segment TTL and larger edge cache; warm popular manifests."
        routing_hint = "Steer users to regional edge; enable backup path if SD-WAN metrics degrade."
    else:
        cache_hint = "Standard CDN / Varnish policy; refresh hot titles on schedule."
        routing_hint = "Balanced anycast; monitor via infra ingest."

    hints = {
        "abr": "Use ladder from quality.height / target_bitrate_mbps with DASH or HLS.",
        "transcoding": "Align FFmpeg renditions with control-plane ladder; edge serves packaged output.",
        "cache": cache_hint,
        "load_balancing": "HAProxy / LB weights from infra edge_load events (see POST /ingest/event).",
        "p2p": "Optional WebRTC or libp2p assist under high edge load — not enforced here.",
        "routing": routing_hint,
    }
    return DeliveryOptimizeResponse(
        quality=quality,
        edge=edge,
        forecast_bottleneck_risk=risk,
        delivery_hints=hints,
    )

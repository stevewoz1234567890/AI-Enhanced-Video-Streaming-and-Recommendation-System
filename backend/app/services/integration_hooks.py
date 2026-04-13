"""Side effects from centralized ingestion (probes, edge telemetry) feeding ABR and forecasting."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.metrics_prom import LAST_BANDWIDTH_MBPS, LAST_LATENCY_MS, NETWORK_PROBE_INGEST
from app.models import EdgeNode, NetworkSample
from app.schemas import IngestEventIn


async def apply_ingest_side_effects(session: AsyncSession, body: IngestEventIn) -> None:
    if body.event_type == "network_probe":
        p = body.payload
        try:
            bw = float(p["bandwidth_mbps"])
            lat = float(p["latency_ms"])
            cg = float(p.get("congestion", 0.0))
            src = str(p.get("source", body.source))[:32]
            session.add(
                NetworkSample(
                    bandwidth_mbps=bw,
                    latency_ms=lat,
                    congestion=cg,
                    source=src,
                )
            )
            NETWORK_PROBE_INGEST.inc()
            LAST_BANDWIDTH_MBPS.set(bw)
            LAST_LATENCY_MS.set(lat)
        except (KeyError, TypeError, ValueError):
            pass

    if body.source == "infra" and body.event_type == "edge_load":
        edge = await session.get(EdgeNode, body.origin_id)
        if edge is not None:
            if "current_load" in body.payload:
                edge.current_load = float(body.payload["current_load"])
            if "healthy" in body.payload:
                edge.healthy = bool(body.payload["healthy"])
            if "capacity_units" in body.payload:
                edge.capacity_units = float(body.payload["capacity_units"])
